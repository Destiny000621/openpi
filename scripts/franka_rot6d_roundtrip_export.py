"""Export a replay-ready episode whose actions round-tripped the pi0.5 rot6d serving chain.

Answers, on the real robot: "did the 8-D (quat pose) -> 10-D (rot6d) conversion
used by a rot6d10 TrainConfig change the actions?" — by pushing a training
episode's ground-truth actions through EXACTLY the transform chain a served
(perfect) policy output traverses, then exporting the result in the raw-episode
format Haply_Franka's ``pixi run replay`` consumes. If this episode replays as
perfectly as the raw recording, the representation round trip is exonerated and
the failure lives in the model; if it degrades, the conversion chain is the bug.

Emulated chain (per inference anchor ``a``, mirroring openpi serving exactly):
    train  : FrankaInputs rot6d state clip (franka_policy.py:80-88)
             -> DeltaActions(mask=make_bool_mask(9,-1))  xyz+rot6d rel-to-state
             -> Normalize(quantile)                       -> float32 model input
    model  : identity — the "policy" predicts the ground-truth normalized chunk
    serve  : Unnormalize(quantile) -> AbsoluteActions(mask) -> slice [:, :10]
             (the exact output order of policy_config.create_trained_policy;
             note Unnormalize round-trips the STATE too before AbsoluteActions
             adds it back, just like Policy.infer does)
    client : Gram-Schmidt rot6d -> quaternion — a verbatim port of
             ``_rot6d_to_rot`` from avantbot's franka_ee_client.py (the
             deployed r6-schema parser)
Norm stats come from the CHECKPOINT's baked assets — the same norm_stats.json
``serve_policy`` loads via policy_config.create_trained_policy.

The export also cross-checks, numerically:
  * round trip vs. the dataset's own 10-D actions (transform chain ~ identity),
  * round trip vs. the RAW recording's original target_pose quaternions
    (the true end-to-end 8D -> 10D -> 8D error, in deg / mm / rad),
  * that the checkpoint stats really carry the rot6d identity override
    (normalize_rot6d=False), and that the stored rot6d columns are orthonormal.

Usage (openpi venv):
    uv run python scripts/franka_rot6d_roundtrip_export.py \
        --config-name pi05_franka_double_cable_100_r6_rawrot \
        --dataset /home/boyuan/Desktop/openpi/double_cable_100_r6_v21 \
        --episode 5

Then replay on the robot (Haply_Franka; read replay.py's warnings first):
    pixi run replay -- --episode-dir data_log/_replay_from_lerobot/<printed> --dry-run
    pixi run replay -- --episode-dir <printed> --speed 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.shared import normalize as _normalize

XYZ = slice(0, 3)
ROT6D = slice(3, 9)
GRIP = 9


def _load_parquet(dataset: Path, episode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (action (T,10) f32, state (T,10) f32, timestamp (T,) f64) for one episode."""
    info = json.loads((dataset / "meta" / "info.json").read_text())
    rel = info["data_path"].format(episode_chunk=episode // int(info["chunks_size"]), episode_index=episode)
    tbl = pq.read_table(dataset / rel, columns=["action", "observation.state", "timestamp"])
    cols = tbl.to_pydict()
    action = np.asarray(cols["action"], dtype=np.float32)
    state = np.asarray(cols["observation.state"], dtype=np.float32)
    ts = np.asarray(cols["timestamp"], dtype=np.float64).reshape(-1)
    return action, state, ts


def _rot6d_to_rot(rot6d: np.ndarray) -> Rotation:
    """Verbatim port of avantbot franka_ee_client._rot6d_to_rot (the deployed parser).

    [c0(3), c1(3)] (first two rotation-matrix COLUMNS) -> scipy Rotation via
    Gram-Schmidt; raises on degenerate input.
    """
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
    """[x,y,z, r6(6), ...] (T,10+) -> [qw,qx,qy,qz,x,y,z] (T,7) via the client parser."""
    out = np.empty((len(vec10), 7))
    for i, row in enumerate(vec10):
        q_xyzw = _rot6d_to_rot(row[ROT6D]).as_quat()
        out[i, 0] = q_xyzw[3]
        out[i, 1:4] = q_xyzw[:3]
        out[i, 4:7] = row[XYZ]
    return out


def _continuity_fix(quat: np.ndarray) -> np.ndarray:
    """Remove frame-to-frame hemisphere jumps (converter's _continuity_fix)."""
    dots = np.einsum("td,td->t", quat[1:], quat[:-1])
    flips = np.cumprod(np.where(dots < 0, -1.0, 1.0))
    out = quat.copy()
    out[1:] *= flips[:, None]
    return out


def _geodesic_deg(r_a: Rotation, r_b: Rotation) -> np.ndarray:
    """Per-frame angle between two Rotation stacks, in degrees."""
    return np.degrees((r_a.inv() * r_b).magnitude())


def _raw_gripper(store: dict, prefer_key: str, joint_key: str) -> np.ndarray | None:
    """Gripper column (T,), mirroring the converter's _gripper source priority."""
    if prefer_key in store:
        return np.asarray(store[prefer_key], dtype=np.float64).reshape(-1)
    if joint_key in store and np.asarray(store[joint_key]).ndim == 2:
        return np.asarray(store[joint_key], dtype=np.float64)[:, 7]
    return None


def roundtrip_serving_chain(
    action: np.ndarray, state: np.ndarray, norm_stats: dict, anchor_stride: int
) -> np.ndarray:
    """Push GT absolute actions through delta->normalize->[model]->unnormalize->absolute.

    ``anchor_stride`` mimics the client's open_loop_horizon: one "inference" per
    stride, its chunk delta'd against (and re-anchored to) that tick's state.
    With ground-truth actions the result is anchor-independent up to float32
    rounding — which is exactly what this script measures.
    """
    mask = _transforms.make_bool_mask(9, -1)
    delta = _transforms.DeltaActions(mask)
    absolute = _transforms.AbsoluteActions(mask)
    norm = _transforms.Normalize(norm_stats, use_quantiles=True)  # pi05: use_quantile_norm=True (config.py:189)
    unnorm = _transforms.Unnormalize(norm_stats, use_quantiles=True)

    out = np.empty_like(action, dtype=np.float64)
    for a in range(0, len(action), anchor_stride):
        chunk = action[a : a + anchor_stride]
        data = {"state": state[a].copy(), "actions": chunk.astype(np.float32).copy()}
        data = norm(delta(data))
        # ---- model boundary: a perfect policy emits exactly these values ----
        data = {k: np.asarray(v, dtype=np.float32) for k, v in data.items()}
        data = absolute(unnorm(data))  # Policy.infer output order (policy.py:92-102)
        out[a : a + anchor_stride] = data["actions"][:, :10]  # FrankaOutputs slice
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config-name", default="pi05_franka_double_cable_100_r6_rawrot")
    ap.add_argument("--dataset", type=Path, default=Path("/home/boyuan/Desktop/openpi/double_cable_100_r6_v21"))
    ap.add_argument("--episode", type=int, default=5)
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/home/boyuan/.cache/openpi/hf/pi05_franka_double_cable_100_r6_rawrot_8k"),
        help="Served checkpoint; its baked assets/<repo_id>/norm_stats.json is used (serve parity).",
    )
    ap.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/boyuan/Desktop/Haply_Franka/data_log/double_lan_insertion_uniform"),
        help="Raw session root for true timestamps + end-to-end error vs the original quats.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/boyuan/Desktop/Haply_Franka/data_log/_replay_from_lerobot"),
    )
    ap.add_argument("--anchor-stride", type=int, default=16, help="Ticks per emulated inference (client open_loop_horizon).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # --- config guardrails: this test is only meaningful for the rot6d10 recipe ---
    cfg = _config.get_config(args.config_name)
    dc = cfg.data
    if getattr(dc, "action_representation", None) != "rot6d10" or getattr(dc, "state_dim", None) != 10:
        raise SystemExit(f"{args.config_name} is not a rot6d10/state_dim=10 config.")
    if not getattr(dc, "use_delta_joint_actions", False):
        raise SystemExit(f"{args.config_name} does not use delta actions; nothing to round-trip.")
    if args.dataset.name != Path(dc.repo_id).name:
        print(f"WARNING: dataset dir '{args.dataset.name}' != config repo_id '{dc.repo_id}'")

    # --- norm stats: the exact file serve_policy loads for this checkpoint ---
    stats_dir = args.checkpoint_dir / "assets" / dc.repo_id
    norm_stats = _normalize.load(stats_dir)
    a_q01, a_q99 = np.asarray(norm_stats["actions"].q01), np.asarray(norm_stats["actions"].q99)
    rot_identity = bool(np.all(a_q01[ROT6D] == -1.0) and np.all(a_q99[ROT6D] == 1.0))
    print(f"norm stats : {stats_dir}")
    print(f"  action q01[3:9]={a_q01[ROT6D]}  q99[3:9]={a_q99[ROT6D]}")
    print(f"  rot6d identity override baked in: {rot_identity}"
          f"  (config normalize_rot6d={getattr(dc, 'normalize_rot6d', None)})")
    if getattr(dc, "normalize_rot6d", True) is False and not rot_identity:
        print("  *** MISMATCH: config says raw rot6d but the baked stats normalize it — serving would distort rotations! ***")

    # --- load the episode ---
    action, state, ts_parquet = _load_parquet(args.dataset, args.episode)
    T = len(action)
    print(f"episode    : {args.dataset.name} #{args.episode}  ({T} frames)")

    # Stored rot6d must be (float32-) exact rotation-matrix columns.
    c0, c1 = action[:, 3:6].astype(np.float64), action[:, 6:9].astype(np.float64)
    print(f"rot6d sanity: |c0|-1 max {np.abs(np.linalg.norm(c0, axis=1) - 1).max():.2e}, "
          f"|c1|-1 max {np.abs(np.linalg.norm(c1, axis=1) - 1).max():.2e}, "
          f"|c0.c1| max {np.abs((c0 * c1).sum(axis=1)).max():.2e}")

    # --- FrankaInputs state clip (franka_policy.py:80-88), then the serving chain ---
    state_clipped = np.concatenate(
        [state[:, :3], np.clip(state[:, 3:9], -1.0, 1.0), state[:, 9:]], axis=-1
    ).astype(np.float32)
    rt = roundtrip_serving_chain(action, state_clipped, norm_stats, args.anchor_stride)

    # --- transform-chain error vs the dataset's own actions (should be ~f32 eps) ---
    d = rt - action.astype(np.float64)
    rot_rt = Rotation.concatenate([_rot6d_to_rot(r) for r in rt[:, ROT6D]])
    rot_ds = Rotation.concatenate([_rot6d_to_rot(r) for r in action[:, ROT6D].astype(np.float64)])
    print("\n[chain vs dataset]  (delta+quantile-norm round trip; expect ~1e-7)")
    print(f"  xyz   max {np.abs(d[:, XYZ]).max():.3e} m")
    print(f"  rot6d max {np.abs(d[:, ROT6D]).max():.3e}   geodesic max {_geodesic_deg(rot_rt, rot_ds).max():.3e} deg")
    print(f"  grip  max {np.abs(d[:, GRIP]).max():.3e} rad")

    # --- client-side parse to pose7 (Gram-Schmidt -> quat), replay layout ---
    target_pose = _vec10_to_pose7(rt)
    ee_pose = _vec10_to_pose7(state.astype(np.float64))
    target_pose[:, :4] = _continuity_fix(target_pose[:, :4])
    ee_pose[:, :4] = _continuity_fix(ee_pose[:, :4])

    # --- end-to-end error vs the RAW recording (the original 8-D commands) ---
    ts_out, raw_dir, raw_note = ts_parquet, None, "parquet uniform timestamps"
    if args.raw_root.exists():
        raws = sorted(p.parent for p in args.raw_root.rglob("arm0_states.npz"))
        if args.episode < len(raws):
            cand = raws[args.episode]
            raw_ac = dict(np.load(cand / "arm0_actions.npz"))
            raw_st = dict(np.load(cand / "arm0_states.npz"))
            tgt_raw = np.asarray(raw_ac["target_pose"], dtype=np.float64)
            e_first = float(np.abs(action[0, :3].astype(np.float64) - tgt_raw[0, 4:7]).max())
            e_last = float(np.abs(action[T - 1, :3].astype(np.float64) - tgt_raw[T - 1, 4:7]).max())
            if e_first < 1e-6 and e_last < 1e-6 and len(tgt_raw) >= T:
                raw_dir = cand
                rot_raw = Rotation.from_quat(tgt_raw[:T][:, [1, 2, 3, 0]])  # wxyz -> scipy xyzw
                geo = _geodesic_deg(rot_rt, rot_raw)
                dxyz = np.linalg.norm(rt[:, XYZ] - tgt_raw[:T, 4:7], axis=1) * 1000.0
                print(f"\n[end-to-end vs raw]  {cand.name}  (8D -> rot6d -> transforms -> 8D)")
                print(f"  rot geodesic: mean {geo.mean():.4f} deg  max {geo.max():.4f} deg")
                print(f"  xyz         : mean {dxyz.mean():.5f} mm  max {dxyz.max():.5f} mm")
                grip_raw = _raw_gripper(raw_ac, "gripper_target", "joint_targets")
                if grip_raw is not None:
                    print(f"  gripper     : max {np.abs(rt[:, GRIP] - grip_raw[:T]).max():.3e} rad")
                st_raw = np.asarray(raw_st["ee_pose"], dtype=np.float64)[:T]
                geo_s = _geodesic_deg(
                    Rotation.concatenate([_rot6d_to_rot(r) for r in state[:, ROT6D].astype(np.float64)]),
                    Rotation.from_quat(st_raw[:, [1, 2, 3, 0]]),
                )
                print(f"  state ee rot: max {geo_s.max():.4f} deg  (same chain on observation.state)")
                # Faithful pacing + diffability: reuse the raw loop timestamps and
                # the raw stream's starting hemisphere.
                ts_out = np.load(cand / "timestamps.npy").astype(np.float64)[:T]
                raw_note = f"raw loop timestamps from {cand}"
                if float(np.dot(target_pose[0, :4], tgt_raw[0, :4])) < 0:
                    target_pose[:, :4] *= -1.0
                if float(np.dot(ee_pose[0, :4], st_raw[0, :4])) < 0:
                    ee_pose[:, :4] *= -1.0
            else:
                print(f"\nWARNING: raw candidate {cand.name} does not content-match "
                      f"(first {e_first:.2e}, last {e_last:.2e}, len {len(tgt_raw)} vs {T}) — "
                      "skipping raw comparison, keeping parquet timestamps.")
        else:
            print(f"\nWARNING: raw root has only {len(raws)} episodes; no index {args.episode}.")
    else:
        print(f"\nWARNING: raw root {args.raw_root} not found — parquet timestamps, no end-to-end check.")

    # --- write the replay-compatible episode dir (lerobot_to_replay.py layout) ---
    out = args.out_dir / f"episode_{args.dataset.name}_{args.episode:06d}_rot6d_roundtrip"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists — pass --overwrite to replace it.")
    out.mkdir(parents=True, exist_ok=True)
    joint_targets = np.full((T, 8), np.nan)
    joint_targets[:, 7] = rt[:, GRIP]  # replay.py reads the gripper cmd from col 7
    np.save(out / "timestamps.npy", ts_out)
    np.savez(out / "arm0_actions.npz", target_pose=target_pose, joint_targets=joint_targets)
    np.savez(out / "arm0_states.npz", ee_pose=ee_pose)
    (out / "metadata.json").write_text(json.dumps({
        "arms": ["arm0"],
        "source": "rot6d_roundtrip",
        "config_name": args.config_name,
        "norm_stats": str(stats_dir),
        "source_dataset": str(args.dataset),
        "source_episode_index": args.episode,
        "raw_episode": str(raw_dir) if raw_dir else None,
        "timestamps": raw_note,
        "anchor_stride": args.anchor_stride,
        "note": "GT actions pushed through the exact pi0.5 serving transform chain "
                "(FrankaInputs clip -> DeltaActions(9,-1) -> Normalize -> [identity model] "
                "-> Unnormalize -> AbsoluteActions -> slice10) with the checkpoint's baked "
                "norm stats, then rot6d->quat via avantbot franka_ee_client._rot6d_to_rot. "
                "Replaying this isolates the 8D->10D->8D representation round trip.",
    }, indent=2))

    print(f"\nWrote {T} ticks ({float(ts_out[-1] - ts_out[0]):.1f}s) -> {out}")
    print(f"  timestamps  : {raw_note}")
    print(f"  gripper cmd : min {np.nanmin(joint_targets[:, 7]):.4f}  max {np.nanmax(joint_targets[:, 7]):.4f} rad")
    print("\nReplay it (read src/collector/replay.py's warnings first):")
    print(f"  pixi run replay -- --episode-dir {out} --dry-run")
    print(f"  pixi run replay -- --episode-dir {out} --speed 0.5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
