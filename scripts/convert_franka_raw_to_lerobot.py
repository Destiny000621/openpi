"""Convert raw single-arm Franka FR3 recordings to an EE-space LeRobot v2.1 dataset.

Why EE-space (not joint-space): in these GELLO/Haply recordings the arm columns of
``joint_targets`` are DEAD (std=0 — never stamped by the teleop/CRISP stack); only
the gripper column carries signal. The real arm command lives in ``target_pose``
(absolute EE target pose, stamped by the runner in both Cartesian and joint-IK mode).

Representation (8-D — target_pose + gripper, used directly):
    state  = [ qw, qx, qy, qz, x, y, z,  gripper ]   from ee_pose + gripper state
    action = [ qw, qx, qy, qz, x, y, z,  gripper ]   from target_pose + gripper action

We keep the raw quaternion (measured: 0 sign-flips across the whole dataset, so it is
already continuous — no need for 6D). Quaternions are canonicalized per episode to a
single hemisphere (qw>=0 at frame 0) so the same physical orientation is represented
consistently across episodes. xyz are world-frame meters; gripper is raw knuckle
radians in [0, 0.7929]. Poses are stored ABSOLUTE; the TrainConfig applies
DeltaActions(mask=(-4,3,-1)) at train time: xyz -> delta vs state, quaternion +
gripper stay absolute.

Why not avantbot's convert_lerobot + convert_v3_to_v21.py: avantbot emits LeRobot v3.0
(flattened per-dim columns, many episodes per parquet); openpi's lerobot 0.1.0 needs
v2.1 (one parquet/episode, fixed_size_list columns). The symlink-only v3->v21 script
fixes neither, and would carry the dead joint_targets. This script reads the raw
recordings and writes clean EE-space v2.1 directly.

Raw layout (one dir per session, one subdir per episode):
    <session>/episode_*/
        arm0_states.npz     # ee_pose (T,7)=[qw,qx,qy,qz,x,y,z]; gripper: gripper_pos or joint_pos[:,7]
        arm0_actions.npz    # target_pose (T,7)=[qw,qx,qy,qz,x,y,z]; gripper: gripper_target or joint_targets[:,7]
        camera0.mp4         # wrist camera
        camera1.mp4         # side / third-person camera
        metadata.json, SUCCESS

Usage:
    export HF_LEROBOT_HOME=/mnt/localssd/Sichang/lerobot_home
    uv run python scripts/convert_franka_raw_to_lerobot.py \
        --input-dir "/mnt/localssd/Sichang/Autel Haply Dataset" \
        --repo-id local/lan_insertion_v21 \
        --task "insert the LAN cable" \
        --fps 30 --success-only
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_NAMES = ["qw", "qx", "qy", "qz", "x", "y", "z", "gripper"]  # 8-D


def _canon_quat(pose7: np.ndarray) -> np.ndarray:
    """Copy of [qw,qx,qy,qz,x,y,z] with the quaternion flipped to the qw>=0
    hemisphere (per-episode constant sign; preserves within-episode continuity)."""
    pose = pose7.copy()
    if pose[0, 0] < 0:  # decide hemisphere from the first frame
        pose[:, :4] *= -1.0
    return pose


def _gripper(store, prefer_key: str, joint_key: str) -> np.ndarray:
    """Gripper column (T,1): prefer an explicit key, else the last joint slot."""
    if prefer_key in store:
        g = np.asarray(store[prefer_key], dtype=np.float32).reshape(-1, 1)
    else:
        g = np.asarray(store[joint_key], dtype=np.float32)[:, 7:8]
    return g


def _find_episodes(input_dir: Path, success_only: bool) -> list[Path]:
    eps = sorted(p.parent for p in input_dir.rglob("arm0_states.npz"))
    if success_only:
        kept = [ep for ep in eps if (ep / "SUCCESS").exists()]
        logger.info("success-only: kept %d / %d episodes", len(kept), len(eps))
        eps = kept
    if not eps:
        raise FileNotFoundError(f"No episodes (arm0_states.npz) found under {input_dir}")
    return eps


def _read_video(path: Path, max_side: int = 640) -> np.ndarray:
    """Decode an mp4 to (N, H, W, 3) uint8 RGB, downscaling so the longest side is
    <= max_side (aspect preserved, even dims). pi0.5 resizes every image to 224x224,
    so storing HD frames is pure waste and dominates conversion time."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    dst = None  # (w, h) computed once from the first frame
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if dst is None:
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            dst = (max(2, int(round(w * scale)) & ~1), max(2, int(round(h * scale)) & ~1))
        if (dst[0], dst[1]) != (frame.shape[1], frame.shape[0]):
            frame = cv2.resize(frame, dst, interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Decoded 0 frames from {path}")
    return np.stack(frames)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True, help="Dataset root (contains session dirs).")
    ap.add_argument("--repo-id", type=str, required=True, help="e.g. local/lan_insertion_v21")
    ap.add_argument("--task", type=str, default=None, help="Instruction; defaults to metadata task_instruction.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--success-only", action="store_true", help="Only convert episodes with a SUCCESS marker.")
    ap.add_argument("--overwrite", action="store_true", help="Delete an existing dataset at the target first.")
    args = ap.parse_args()

    out_root = HF_LEROBOT_HOME / args.repo_id
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_root)

    episodes = _find_episodes(args.input_dir, args.success_only)

    probe = _read_video(episodes[0] / "camera0.mp4")[0]
    height, width = int(probe.shape[0]), int(probe.shape[1])
    logger.info("Camera resolution: %dx%d", height, width)

    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "observation.images.camera0": {  # wrist
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.camera1": {  # side / third-person
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type="fr3",
        features=features,
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    total_frames = 0
    for ep_idx, ep in enumerate(episodes):
        meta = json.loads((ep / "metadata.json").read_text())
        task = args.task or meta.get("task_instruction", "franka task")

        st = np.load(ep / "arm0_states.npz")
        ac = np.load(ep / "arm0_actions.npz")
        # State: ee_pose (canonicalized) + gripper state.
        state = np.concatenate(
            [_canon_quat(st["ee_pose"].astype(np.float32)), _gripper(st, "gripper_pos", "joint_pos")], axis=1
        )
        # Action: target_pose (canonicalized) + gripper action.
        action = np.concatenate(
            [_canon_quat(ac["target_pose"].astype(np.float32)), _gripper(ac, "gripper_target", "joint_targets")],
            axis=1,
        )
        cam0 = _read_video(ep / "camera0.mp4")  # wrist
        cam1 = _read_video(ep / "camera1.mp4")  # side

        t = min(len(state), len(action), len(cam0), len(cam1))
        if len({len(state), len(action), len(cam0), len(cam1)}) != 1:
            logger.warning(
                "ep %s length mismatch state=%d action=%d cam0=%d cam1=%d -> %d",
                ep.name, len(state), len(action), len(cam0), len(cam1), t,
            )

        for i in range(t):
            dataset.add_frame(
                {
                    "observation.state": state[i],
                    "action": action[i],
                    "observation.images.camera0": cam0[i],
                    "observation.images.camera1": cam1[i],
                    "task": task,
                }
            )
        dataset.save_episode()
        total_frames += t
        logger.info("[%d/%d] %s -> %d frames (task=%r)", ep_idx + 1, len(episodes), ep.name, t, task)

    logger.info("Done. %d episodes / %d frames at %s", len(episodes), total_frames, out_root)


if __name__ == "__main__":
    main()
