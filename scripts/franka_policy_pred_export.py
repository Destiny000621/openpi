"""Export a replay-ready episode built from REAL pi0.5-SFT policy predictions.

Complement to franka_rot6d_roundtrip_export.py: that script pushed ground-truth
actions through the serving transforms (identity model) to test the
representation; THIS one runs the actual trained checkpoint on the demo
episode's recorded observations and exports what the POLICY predicts, stitched
at the deployment cadence, in the raw-episode format ``pixi run replay``
consumes.

Teacher-forced open-loop stitching (the closest offline analog of deployment):
    anchors a = 0, C, 2C, ...   (C = steps consumed per chunk; the deployed
                                 client re-infers after executing 15 of 50 —
                                 open_loop_horizon 50 x (1 - threshold 0.7))
    obs(a)  = dataset images (camera1 -> side, camera0 -> wrist) + 10-D state
              + the training prompt — i.e., the observations of a robot that
              tracked the demo perfectly
    chunk   = policy.infer(obs)["actions"]  (50, 10) ABSOLUTE (server output
              transforms already applied — full serve parity via
              policy_config.create_trained_policy)
    stitched[a : a+C] = chunk[:C]           (latest_only hard switch)

Interpretation of the replay:
  * good insertions  -> the model can fit its training episode; live failures
    are closed-loop / visual-generalization effects, not the action head.
  * misses           -> the model cannot even reproduce a training episode's
    insertion precision from ground-truth observations (fit/perception limit).
Chunk-boundary jumps in the stitched stream are the model's prediction error
made visible — their size is reported (replay.py aborts on steps > --max-step).

Usage (this machine, RTX 5090):
    export HF_LEROBOT_HOME=<dir whose local/<repo> is the dataset>   # see below
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python \
        scripts/franka_policy_pred_export.py --episode 5

If local/double_cable_100_r6_v21 is not in your lerobot home, symlink it:
    mkdir -p /tmp/lerobot_home/local
    ln -s /home/boyuan/Desktop/openpi/double_cable_100_r6_v21 \
          /tmp/lerobot_home/local/double_cable_100_r6_v21
    export HF_LEROBOT_HOME=/tmp/lerobot_home
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config

XYZ, ROT6D, GRIP = slice(0, 3), slice(3, 9), 9
GRIP_THR = 0.4


def _rot6d_to_rot(rot6d: np.ndarray) -> Rotation:
    """Verbatim port of avantbot franka_ee_client._rot6d_to_rot (deployed parser)."""
    r6 = np.asarray(rot6d, dtype=np.float64)
    c0, c1 = r6[:3].copy(), r6[3:6].copy()
    n0 = float(np.linalg.norm(c0))
    if n0 < 1e-6:
        raise ValueError(f"degenerate rot6d action: |c0| = {n0:.2e}")
    c0 /= n0
    c1 -= c0 * float(c0 @ c1)
    n1 = float(np.linalg.norm(c1))
    if n1 < 1e-6:
        raise ValueError(f"degenerate rot6d action: c1 collinear with c0 (|c1_perp| = {n1:.2e})")
    c1 /= n1
    return Rotation.from_matrix(np.stack([c0, c1, np.cross(c0, c1)], axis=1))


def _vec10_to_pose7(vec10: np.ndarray) -> np.ndarray:
    out = np.empty((len(vec10), 7))
    for i, row in enumerate(vec10):
        q = _rot6d_to_rot(row[ROT6D]).as_quat()  # xyzw
        out[i, 0], out[i, 1:4], out[i, 4:7] = q[3], q[:3], row[XYZ]
    return out


def _continuity_fix(quat: np.ndarray) -> np.ndarray:
    dots = np.einsum("td,td->t", quat[1:], quat[:-1])
    flips = np.cumprod(np.where(dots < 0, -1.0, 1.0))
    out = quat.copy()
    out[1:] *= flips[:, None]
    return out


def _crossings(g: np.ndarray, thr: float = GRIP_THR) -> int:
    above = g >= thr
    return int(np.abs(np.diff(above.astype(int))).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config-name", default="pi05_franka_double_cable_100_r6_rawrot")
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("/home/boyuan/.cache/openpi/hf/pi05_franka_double_cable_100_r6_rawrot_8k"))
    ap.add_argument("--episode", type=int, default=5)
    ap.add_argument("--consume", type=int, default=15,
                    help="Steps executed per chunk before re-inference (deployed cadence: 15 of 50).")
    ap.add_argument("--raw-root", type=Path,
                    default=Path("/home/boyuan/Desktop/Haply_Franka/data_log/double_lan_insertion_uniform"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/home/boyuan/Desktop/Haply_Franka/data_log/_replay_from_lerobot"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: PLC0415 — after env is set

    cfg = _config.get_config(args.config_name)
    if getattr(cfg.data, "action_representation", None) != "rot6d10":
        raise SystemExit(f"{args.config_name} is not a rot6d10 config.")
    prompt = cfg.data.default_prompt
    print(f"loading policy from {args.checkpoint_dir} (serve parity) ...")
    policy = _policy_config.create_trained_policy(cfg, args.checkpoint_dir)

    ds = LeRobotDataset(cfg.data.repo_id)
    start = int(ds.episode_data_index["from"][args.episode])
    end = int(ds.episode_data_index["to"][args.episode])
    T = end - start
    print(f"episode {args.episode}: {T} frames; inferring every {args.consume} steps "
          f"-> {int(np.ceil(T / args.consume))} chunks")

    # GT actions/states straight from the parquet — ds[i] would decode two video
    # frames per row; only the 71 anchor observations need images.
    import pyarrow.parquet as pq  # noqa: PLC0415

    info = json.loads((Path(ds.root) / "meta" / "info.json").read_text())
    rel = info["data_path"].format(episode_chunk=args.episode // int(info["chunks_size"]),
                                   episode_index=args.episode)
    cols = pq.read_table(Path(ds.root) / rel, columns=["action", "observation.state"]).to_pydict()
    gt = np.asarray(cols["action"], dtype=np.float64)[:T]
    states = np.asarray(cols["observation.state"], dtype=np.float64)[:T]

    pred = np.full((T, 10), np.nan)
    boundary_jump = []
    for a in range(0, T, args.consume):
        item = ds[start + a]
        obs = {
            "observation/image": np.asarray(item["observation.images.camera1"]),        # side -> base_0_rgb
            "observation/wrist_image": np.asarray(item["observation.images.camera0"]),  # wrist -> left_wrist_0_rgb
            "observation/state": np.asarray(item["observation.state"], dtype=np.float32),
            "prompt": prompt,
        }
        chunk = np.asarray(policy.infer(obs)["actions"], dtype=np.float64)  # (50, 10) absolute
        n = min(args.consume, T - a, len(chunk))
        pred[a : a + n] = chunk[:n]
        if a > 0:
            boundary_jump.append(float(np.linalg.norm(pred[a, XYZ] - pred[a - 1, XYZ])))
        if (a // args.consume) % 10 == 0:
            print(f"  chunk @ t={a} ({a/30:.1f}s)")

    assert not np.isnan(pred).any()

    # ---- prediction-vs-GT report (the model's fit on this training episode) ----
    dxyz = np.linalg.norm(pred[:, XYZ] - gt[:, XYZ], axis=1) * 1e3
    r_pred = Rotation.concatenate([_rot6d_to_rot(r) for r in pred[:, ROT6D]])
    r_gt = Rotation.concatenate([_rot6d_to_rot(r) for r in gt[:, ROT6D]])
    drot = np.degrees((r_pred.inv() * r_gt).magnitude())
    dgrip = np.abs(pred[:, GRIP] - gt[:, GRIP])
    ins = (gt[:, 1] < 0.10) & (gt[:, 2] < 0.365)
    grasp = (gt[:, 1] > 0.14) & (gt[:, 2] < 0.365)
    print("\n[policy prediction vs GT actions]  (teacher-forced, executed prefix only)")
    for name, m in (("all", np.ones(T, bool)), ("insert-phase", ins), ("grasp-phase", grasp)):
        if m.any():
            print(f"  {name:12s}: xyz mean {dxyz[m].mean():6.2f} / p95 {np.percentile(dxyz[m],95):6.2f} mm"
                  f"   rot mean {drot[m].mean():5.2f} / p95 {np.percentile(drot[m],95):5.2f} deg"
                  f"   grip mean {dgrip[m].mean():.4f} rad")
    steps = np.linalg.norm(np.diff(pred[:, XYZ], axis=0), axis=1) * 1e3
    print(f"  chunk-boundary xyz jump: mean {np.mean(boundary_jump)*1e3:.1f} / max {np.max(boundary_jump)*1e3:.1f} mm"
          f"   (within-chunk step p99 {np.percentile(steps,99):.1f} mm; replay guard --max-step is 80 mm)")
    print(f"  gripper transitions: pred {_crossings(pred[:, GRIP])} vs GT {_crossings(gt[:, GRIP])}"
          f"   pred grip range [{pred[:, GRIP].min():.3f}, {pred[:, GRIP].max():.3f}] rad")

    # ---- replay export (same layout as the roundtrip/lerobot_to_replay dirs) ----
    target_pose = _vec10_to_pose7(pred)
    ee_pose = _vec10_to_pose7(states)  # GT measured pose, for --source ee_pose / reference
    target_pose[:, :4] = _continuity_fix(target_pose[:, :4])
    ee_pose[:, :4] = _continuity_fix(ee_pose[:, :4])

    ts_out, raw_note = np.arange(T) / 30.0, "uniform 30 Hz timestamps"
    if args.raw_root.exists():
        raws = sorted(p.parent for p in args.raw_root.rglob("arm0_states.npz"))
        if args.episode < len(raws):
            cand = raws[args.episode]
            tgt_raw = np.asarray(np.load(cand / "arm0_actions.npz")["target_pose"], dtype=np.float64)
            if (len(tgt_raw) >= T
                    and np.abs(gt[0, :3] - tgt_raw[0, 4:7]).max() < 1e-6
                    and np.abs(gt[T - 1, :3] - tgt_raw[T - 1, 4:7]).max() < 1e-6):
                ts_out = np.load(cand / "timestamps.npy").astype(np.float64)[:T]
                raw_note = f"raw loop timestamps from {cand}"

    out = args.out_dir / f"episode_{Path(cfg.data.repo_id).name}_{args.episode:06d}_policy_pred_{args.checkpoint_dir.name}"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists — pass --overwrite to replace it.")
    out.mkdir(parents=True, exist_ok=True)
    joint_targets = np.full((T, 8), np.nan)
    joint_targets[:, 7] = np.clip(pred[:, GRIP], 0.0, 0.7929)  # replay reads col 7 (raw knuckle rad)
    np.save(out / "timestamps.npy", ts_out)
    np.savez(out / "arm0_actions.npz", target_pose=target_pose, joint_targets=joint_targets)
    np.savez(out / "arm0_states.npz", ee_pose=ee_pose)
    (out / "metadata.json").write_text(json.dumps({
        "arms": ["arm0"],
        "source": "policy_prediction",
        "config_name": args.config_name,
        "checkpoint": str(args.checkpoint_dir),
        "source_dataset": cfg.data.repo_id,
        "source_episode_index": args.episode,
        "consume_per_chunk": args.consume,
        "timestamps": raw_note,
        "note": "REAL pi0.5-SFT inference on the demo episode's recorded observations "
                "(teacher-forced anchors every {c} ticks, latest_only stitching), decoded "
                "rot6d->quat with the deployed client's Gram-Schmidt. Replaying this shows "
                "what the policy would command if the robot tracked the demo perfectly.".format(c=args.consume),
    }, indent=2))
    print(f"\nWrote {T} ticks ({float(ts_out[-1]-ts_out[0]):.1f}s) -> {out}")
    print("\n⚠ This trajectory is MODEL OUTPUT, not a recorded demo. Dry-run first, keep")
    print("  --max-step at its 80 mm default, start at --speed 0.5, hand on the e-stop:")
    print(f"  pixi run replay -- --episode-dir {out} --dry-run")
    print(f"  pixi run replay -- --episode-dir {out} --speed 0.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
