"""Open-loop action-error eval for single-arm Franka rot6d10 checkpoints.

Ranks checkpoints by how well they predict recorded action chunks, and reports
errors in **physical units** (mm / degrees / gripper radians) rather than only
normalized MSE — normalized numbers are not comparable across dims when
`normalize_rot6d=False` (rot6d bypasses normalization while xyz/gripper do not),
and they are not comparable across configs whose norm-stats differ.

Metrics per checkpoint (averaged over sampled frames x chunk steps):
  * xyz     : L2 position error of the predicted EE target, in mm
  * rot     : geodesic angle between predicted and recorded EE orientation, in deg
  * grip    : absolute gripper error, in knuckle radians (range 0..0.7929)
  * nmse    : mean squared error in the model's normalized space (training-loss
              comparable, reported for continuity with eval_open_loop.py)
Errors are also broken down by position within the 50-step action chunk, which
shows how fast open-loop error grows with horizon.

Reconstruction (mirrors the serving path): the model predicts deltas w.r.t. the
query state, so absolute pose = state + delta for xyz and rot6d; rot6d is then
orthonormalized with Gram-Schmidt exactly as the robot client does.

NOTE on "held-out": a config trained on all episodes of its dataset has no
held-out split — evaluating on that dataset measures action FIT, which is still
the right signal for ranking checkpoints of the same run. Point --repo-id at a
*different* recording session for a generalization signal (expect worse numbers
if the scene geometry differs).

Usage:
    uv run python scripts/eval_open_loop_franka.py \
        --config-name pi05_franka_double_cable_100_r6_rawrot \
        --repo-id local/double_cable_100_r6_v21 --episodes 0-99 \
        --checkpoint <ckpt_dir>/2000 --checkpoint <ckpt_dir>/4000 \
        --checkpoint <ckpt_dir>/5999
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.shared import nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REAL_DIM = 10  # rot6d10 action: [x,y,z, r6_0..r6_5, gripper]
XYZ = slice(0, 3)
ROT6D = slice(3, 9)
GRIP = 9


def parse_episodes(spec: str, total: int) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [e for e in out if e >= total]
    if bad:
        raise SystemExit(f"episodes out of range (dataset has {total}): {bad[:5]}")
    return sorted(set(out))


def build_eval_frames(data_config, action_horizon, repo_id, episodes, stride, max_per_ep):
    """Sample frame indices from the requested episodes, with the full transform chain applied."""
    delta = {
        "action": [t / LeRobotDatasetMetadata(repo_id).fps for t in range(action_horizon)],
    }
    meta = LeRobotDatasetMetadata(repo_id)
    ds = LeRobotDataset(repo_id, delta_timestamps=delta)
    dc = dataclasses.replace(data_config, repo_id=repo_id)
    ds_t = _data_loader.transform_dataset(ds, dc, skip_norm_stats=False)

    frame_idxs: list[int] = []
    for ep in episodes:
        start = int(ds.episode_data_index["from"][ep])
        end = int(ds.episode_data_index["to"][ep])
        last = end - action_horizon  # need a full chunk of ground truth
        if last <= start:
            continue
        frame_idxs.extend(list(range(start, last, stride))[:max_per_ep])
    return ds_t, frame_idxs


def collate(items: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack(xs), *items)


def _gram_schmidt(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3). Columns c0, c1 orthonormalized; c2 = c0 x c1."""
    c0 = rot6d[..., 0:3].astype(np.float64)
    c1 = rot6d[..., 3:6].astype(np.float64)
    c0 = c0 / np.clip(np.linalg.norm(c0, axis=-1, keepdims=True), 1e-9, None)
    c1 = c1 - c0 * np.sum(c0 * c1, axis=-1, keepdims=True)
    c1 = c1 / np.clip(np.linalg.norm(c1, axis=-1, keepdims=True), 1e-9, None)
    c2 = np.cross(c0, c1)
    return np.stack([c0, c1, c2], axis=-1)


def _geodesic_deg(r_a: np.ndarray, r_b: np.ndarray) -> np.ndarray:
    """Angle between two rotation matrices, in degrees."""
    rel = np.einsum("...ji,...jk->...ik", r_a, r_b)  # r_a^T @ r_b
    trace = np.trace(rel, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def _unnorm_quantile(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """Invert Normalize._normalize_quantile: (x+1)/2 * (q99-q01) + q01."""
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def evaluate_checkpoint(config, norm_stats, ckpt: str, ds_t, frame_idxs, batch_size, horizon):
    model = config.model.load(_model.restore_params(Path(ckpt) / "params", dtype=jnp.bfloat16))
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    rng = jax.random.key(0)

    a_q01 = np.asarray(norm_stats["actions"].q01)[:REAL_DIM]
    a_q99 = np.asarray(norm_stats["actions"].q99)[:REAL_DIM]

    # accumulators: totals and per-chunk-step
    sq_norm = np.zeros(REAL_DIM, dtype=np.float64)
    xyz_mm, rot_deg, grip_err = [], [], []
    step_xyz = np.zeros(horizon, dtype=np.float64)
    step_rot = np.zeros(horizon, dtype=np.float64)
    n_frames = 0

    for s in range(0, len(frame_idxs), batch_size):
        batch = collate([ds_t[i] for i in frame_idxs[s : s + batch_size]])
        gt = np.asarray(batch.pop("actions"))[..., :REAL_DIM]  # (B,H,10) normalized
        obs = _model.Observation.from_dict(batch)
        rng, key = jax.random.split(rng)
        pred = np.asarray(sample_actions(key, obs))[..., :REAL_DIM]

        sq_norm += ((pred - gt).astype(np.float64) ** 2).sum(axis=(0, 1))

        # --- physical units ---
        # xyz + gripper: quantile-unnormalize the deltas (rot6d dims are raw under
        # normalize_rot6d=False; unnormalizing them with identity stats is a no-op).
        pred_p = _unnorm_quantile(pred.astype(np.float64), a_q01, a_q99)
        gt_p = _unnorm_quantile(gt.astype(np.float64), a_q01, a_q99)

        state = np.asarray(obs.state)[..., :REAL_DIM].astype(np.float64)  # (B,10)
        st_rot = state[:, None, ROT6D]  # (B,1,6) broadcast over the chunk

        # deltas are w.r.t. the query state -> absolute pose = state + delta
        d_xyz = np.linalg.norm(pred_p[..., XYZ] - gt_p[..., XYZ], axis=-1) * 1000.0  # mm
        r_pred = _gram_schmidt(st_rot + pred_p[..., ROT6D])
        r_gt = _gram_schmidt(st_rot + gt_p[..., ROT6D])
        d_rot = _geodesic_deg(r_pred, r_gt)  # (B,H) deg
        d_grip = np.abs(pred_p[..., GRIP] - gt_p[..., GRIP])

        xyz_mm.append(d_xyz.ravel())
        rot_deg.append(d_rot.ravel())
        grip_err.append(d_grip.ravel())
        step_xyz += d_xyz.sum(axis=0)
        step_rot += d_rot.sum(axis=0)
        n_frames += d_xyz.shape[0]

    xyz_mm = np.concatenate(xyz_mm)
    rot_deg = np.concatenate(rot_deg)
    grip_err = np.concatenate(grip_err)
    n_pairs = n_frames * horizon
    return {
        "nmse": float((sq_norm / n_pairs).mean()),
        "xyz_mm": float(xyz_mm.mean()),
        "xyz_mm_p90": float(np.percentile(xyz_mm, 90)),
        "rot_deg": float(rot_deg.mean()),
        "rot_deg_p90": float(np.percentile(rot_deg, 90)),
        "grip": float(grip_err.mean()),
        "step_xyz": step_xyz / max(n_frames, 1),
        "step_rot": step_rot / max(n_frames, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True)
    ap.add_argument("--repo-id", required=True, help="dataset to evaluate on")
    ap.add_argument("--episodes", required=True, help="'0-99' or '0-9,50-59' (inclusive ranges)")
    ap.add_argument("--checkpoint", action="append", required=True, help="checkpoint step dir (repeatable)")
    ap.add_argument("--stride", type=int, default=60, help="sample every Nth frame within an episode")
    ap.add_argument("--max-per-ep", type=int, default=6, help="cap frames per episode")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise SystemExit(f"No norm_stats for '{args.config_name}'. Run compute_norm_stats first.")
    horizon = config.model.action_horizon

    meta = LeRobotDatasetMetadata(args.repo_id)
    eps = parse_episodes(args.episodes, meta.total_episodes)
    ds_t, frame_idxs = build_eval_frames(data_config, horizon, args.repo_id, eps, args.stride, args.max_per_ep)
    logger.info(
        f"Config={args.config_name}  eval_repo={args.repo_id}  episodes={len(eps)}  "
        f"frames={len(frame_idxs)}  horizon={horizon}"
    )
    if not frame_idxs:
        raise SystemExit("No evaluable frames (episodes too short for the action horizon?).")

    rows = []
    for ckpt in args.checkpoint:
        m = evaluate_checkpoint(config, data_config.norm_stats, ckpt, ds_t, frame_idxs, args.batch_size, horizon)
        name = Path(ckpt).parent.parent.name + "/" + Path(ckpt).name
        rows.append((name, m))
        logger.info(
            f"{ckpt}: xyz={m['xyz_mm']:.2f}mm rot={m['rot_deg']:.2f}deg grip={m['grip']:.4f} nmse={m['nmse']:.5f}"
        )

    rows.sort(key=lambda r: r[1]["xyz_mm"])
    print("\n=== open-loop action error (lower = better) ===")
    print(f"  eval set: {args.repo_id}  episodes {args.episodes}  ({len(frame_idxs)} frames x {horizon} steps)")
    print(f"  {'checkpoint':30s} {'xyz mm':>8s} {'p90':>7s} {'rot deg':>8s} {'p90':>7s} {'grip rad':>9s} {'nmse':>8s}")
    for name, m in rows:
        print(
            f"  {name:30s} {m['xyz_mm']:8.2f} {m['xyz_mm_p90']:7.2f} {m['rot_deg']:8.2f} "
            f"{m['rot_deg_p90']:7.2f} {m['grip']:9.4f} {m['nmse']:8.5f}"
        )
    print(f"\n  best by mean xyz error: {rows[0][0]}")

    print("\n=== error growth across the action chunk (mean over frames) ===")
    probes = [0, horizon // 4, horizon // 2, 3 * horizon // 4, horizon - 1]
    print(f"  {'checkpoint':30s} " + " ".join(f"{'s' + str(p):>13s}" for p in probes))
    for name, m in rows:
        cells = " ".join(f"{m['step_xyz'][p]:6.1f}mm/{m['step_rot'][p]:4.1f}d" for p in probes)
        print(f"  {name:30s} {cells}")


if __name__ == "__main__":
    main()
