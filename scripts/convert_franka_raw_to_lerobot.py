"""Convert raw single-arm Franka FR3 recordings to an EE-space LeRobot v2.1 dataset.

Why EE-space actions (not joint-space): in these GELLO/Haply recordings the arm
columns of ``joint_targets`` are DEAD (std=0 — never stamped by the teleop/CRISP
stack); only the gripper column carries signal. The real arm command lives in
``target_pose`` (absolute EE target pose, stamped by the runner in both Cartesian
and joint-IK mode).

Two representations, selected by --rep:

--rep rot6d10 (default; RLinf-style relative-EEF training):
    action (10-D) = [ x, y, z, r6_0..r6_5, gripper ]   from target_pose + gripper target
    state  (10-D) = [ x, y, z, r6_0..r6_5, gripper ]   from ee_pose + gripper state
    r6 = first two columns of the rotation matrix (Zhou et al. 6D), concatenated
    [c0x,c0y,c0z, c1x,c1y,c1z]. Rotation-sign-invariant (R(-q)=R(q)), so NO quat
    canonicalization is needed in this mode. Train-time DeltaActions(mask=(9,-1))
    makes xyz+rot6d relative-to-state; gripper stays absolute. Recover the
    rotation at inference by Gram-Schmidt on the two 3-vectors.

--rep quat8 (legacy; reproduces the _s29 datasets):
    action (8-D) = [ qw, qx, qy, qz, x, y, z,  gripper ]        target_pose + gripper target
    state (29-D) = [ qw, qx, qy, qz, x, y, z,  gripper,         ee_pose + gripper (dims 0-7)
                     j0..j6,                                    joint_pos arm columns (dims 8-14)
                     j0_vel..j6_vel, gripper_vel,               joint_vel (dims 15-22)
                     fx, fy, fz, tx, ty, tz ]                   wrench (dims 23-28)

The first 8 state dims mirror the action layout — the train-time
DeltaActions(mask=(-4,3,-1)) subtracts state[:8] from the action chunk, so
[quat, xyz, gripper] MUST stay at the front of the state. The extra proprio
(arm joint positions, joint velocities incl. gripper, external wrench) rides
behind it; all of it is present in 77/77 recorded episodes.

Quaternion canonicalization: the driver emits per-sample quats whose sign jumps
hemisphere whenever the physical qw crosses 0 — and this task's orientation sits
at qw~0, so the dataset has persistent within-episode sign flips, episodes split
across both hemispheres, and state/action streams that disagree. We repair all
three: (1) enforce sign continuity along each stream, (2) flip the action stream
to agree with the state stream, (3) flip whole episodes to a dataset-level
reference orientation (computed in a cheap npz-only pre-pass, saved to
quat_reference.json for the deploy client). Do NOT canonicalize by sign(qw) —
qw~0 makes that decision noise.

xyz are world-frame meters; gripper is raw knuckle radians in [0, 0.7929].
Poses are stored ABSOLUTE; the TrainConfig applies DeltaActions(mask=(-4,3,-1))
at train time: xyz -> delta vs state, quaternion + gripper stay absolute.

Why not avantbot's convert_lerobot + convert_v3_to_v21.py: avantbot emits LeRobot v3.0
(flattened per-dim columns, many episodes per parquet); openpi's lerobot 0.1.0 needs
v2.1 (one parquet/episode, fixed_size_list columns). The symlink-only v3->v21 script
fixes neither, and would carry the dead joint_targets. This script reads the raw
recordings and writes clean EE-space v2.1 directly.

Raw layout (one dir per session, one subdir per episode):
    <session>/episode_*/
        arm0_states.npz     # ee_pose (T,7)=[qw,qx,qy,qz,x,y,z], joint_pos (T,8),
                            # joint_vel (T,8), joint_eff (T,7), wrench (T,6);
                            # gripper: gripper_pos or joint_pos[:,7]
        arm0_actions.npz    # target_pose (T,7)=[qw,qx,qy,qz,x,y,z]; gripper:
                            # gripper_target or joint_targets[:,7]
        camera0.mp4         # wrist camera
        camera1.mp4         # side / third-person camera
        metadata.json, SUCCESS

Usage:
    export HF_LEROBOT_HOME=/mnt/localssd/Sichang/lerobot_home
    uv run python scripts/convert_franka_raw_to_lerobot.py \
        --input-dir "/mnt/localssd/Sichang/Autel Haply Dataset" \
        --repo-id local/lan_insertion_s29_v21 \
        --task "insert the LAN cable" \
        --fps 30 --success-only

Use a fresh repo id for the 29-D schema (the _s29 convention) — do not reuse or
--overwrite a pre-29-D 8-D dataset name; old datasets/checkpoints are
incompatible (FrankaInputs rejects 8-D states).
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
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# quat8 layout; also the first 8 dims of the 29-D state (see module docstring).
ACTION_NAMES = ["qw", "qx", "qy", "qz", "x", "y", "z", "gripper"]  # 8-D
STATE_NAMES = [
    *ACTION_NAMES,                                                  # ee_pose + gripper (0-7)
    *[f"j{i}" for i in range(7)],                                   # joint_pos arm (8-14)
    *[f"j{i}_vel" for i in range(7)], "gripper_vel",                # joint_vel (15-22)
    "fx", "fy", "fz", "tx", "ty", "tz",                             # wrench (23-28)
]  # 29-D

# rot6d10 layout; state and action share it exactly (see module docstring).
ROT6D_NAMES = ["x", "y", "z", *[f"r6_{i}" for i in range(6)], "gripper"]  # 10-D


def _pose7_to_xyz_rot6d(pose7: np.ndarray) -> np.ndarray:
    """[qw,qx,qy,qz,x,y,z] (T,7) -> [x,y,z, rot6d(6)] (T,9). Sign-invariant."""
    quat_xyzw = pose7[:, [1, 2, 3, 0]].astype(np.float64)  # scipy is scalar-last
    rot = Rotation.from_quat(quat_xyzw).as_matrix()  # (T,3,3)
    rot6d = np.concatenate([rot[:, :, 0], rot[:, :, 1]], axis=1)  # first two columns
    return np.concatenate([pose7[:, 4:7], rot6d.astype(np.float32)], axis=1).astype(np.float32)


def _continuity_fix(quat: np.ndarray) -> np.ndarray:
    """Remove recorded hemisphere jumps: flip the sign from each frame where the
    stream's consecutive 4-D dot goes negative (a >90-deg jump in quat space is a
    sign flip, not physical motion at 30 fps)."""
    dots = np.einsum("td,td->t", quat[1:], quat[:-1])
    flips = np.cumprod(np.where(dots < 0, -1.0, 1.0))
    out = quat.copy()
    out[1:] *= flips[:, None]
    return out


def _compute_quat_reference(episodes: list[Path]) -> np.ndarray:
    """Dataset-level reference orientation (unit quat) from an npz-only pre-pass.

    Per episode: continuity-fix ee_pose quats and take their normalized mean.
    Episode means are hemisphere-aligned to the first episode before averaging,
    so the reference is well-defined even though raw episodes sit in both
    hemispheres. Whole episodes are later flipped to agree with this reference.
    """
    means = []
    for ep in episodes:
        q = _continuity_fix(np.load(ep / "arm0_states.npz")["ee_pose"].astype(np.float64)[:, :4])
        m = q.mean(axis=0)
        means.append(m / np.linalg.norm(m))
    ref = means[0]
    acc = np.zeros(4)
    for m in means:
        acc += m if np.dot(m, ref) >= 0 else -m
    return acc / np.linalg.norm(acc)


def _canonicalize_quats(
    state_quat: np.ndarray, action_quat: np.ndarray, quat_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Continuity-fix both streams, make the action stream agree with the state
    stream, then flip the whole episode onto the reference hemisphere. Returns
    the canonicalized (state_quat, action_quat)."""
    sq = _continuity_fix(state_quat)
    aq = _continuity_fix(action_quat)
    if np.dot(aq[0], sq[0]) < 0:  # target leads measured pose by mm — same hemisphere
        aq = -aq
    if np.dot(sq.mean(axis=0), quat_ref) < 0:
        sq, aq = -sq, -aq
    return sq, aq


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
    ap.add_argument("--repo-id", type=str, required=True, help="e.g. local/lan_insertion_s29_v21")
    ap.add_argument("--task", type=str, default=None, help="Instruction; defaults to metadata task_instruction.")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--success-only", action="store_true", help="Only convert episodes with a SUCCESS marker.")
    ap.add_argument("--overwrite", action="store_true", help="Delete an existing dataset at the target first.")
    ap.add_argument(
        "--rep", choices=["rot6d10", "quat8"], default="rot6d10",
        help="Action/state representation (see module docstring). Default rot6d10.",
    )
    ap.add_argument(
        "--include-joints", action="store_true",
        help="rot6d10 only: append the 7 arm joint_pos columns to the state "
        "(17-D state [xyz, rot6d, gripper, j0..j6]; action stays 10-D).",
    )
    ap.add_argument(
        "--episode-list", type=Path, default=None,
        help="Optional file with one episode dir (path or name) per line; only those are converted.",
    )
    args = ap.parse_args()

    out_root = HF_LEROBOT_HOME / args.repo_id
    if out_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_root} exists. Pass --overwrite to replace it.")
        shutil.rmtree(out_root)

    episodes = _find_episodes(args.input_dir, args.success_only)
    if args.episode_list is not None:
        wanted = {Path(line.strip()).name for line in args.episode_list.read_text().splitlines() if line.strip()}
        episodes = [ep for ep in episodes if ep.name in wanted]
        missing = wanted - {ep.name for ep in episodes}
        if missing:
            raise FileNotFoundError(f"{len(missing)} episodes from --episode-list not found: {sorted(missing)[:5]} ...")
        logger.info("episode-list: converting %d / %d listed episodes", len(episodes), len(wanted))

    quat_ref = None
    if args.rep == "quat8":  # rot6d is rotation-sign-invariant; no canonicalization needed
        quat_ref = _compute_quat_reference(episodes)
        logger.info("Quaternion reference (canonical hemisphere): %s", np.round(quat_ref, 4).tolist())

    probe = _read_video(episodes[0] / "camera0.mp4")[0]
    height, width = int(probe.shape[0]), int(probe.shape[1])
    logger.info("Camera resolution: %dx%d", height, width)

    if args.include_joints and args.rep != "rot6d10":
        raise SystemExit("--include-joints is only supported with --rep rot6d10")
    if args.rep == "rot6d10":
        state_names = ROT6D_NAMES + ([f"j{i}" for i in range(7)] if args.include_joints else [])
        state_schema = {"dtype": "float32", "shape": (len(state_names),), "names": state_names}
        action_schema = {"dtype": "float32", "shape": (10,), "names": ROT6D_NAMES}
    else:
        state_schema = {"dtype": "float32", "shape": (29,), "names": STATE_NAMES}
        action_schema = {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES}
    features = {
        "observation.state": state_schema,
        "action": action_schema,
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
        ee_pose = st["ee_pose"].astype(np.float32)
        target_pose = ac["target_pose"].astype(np.float32)
        if args.rep == "rot6d10":
            # State/action (10-D each): [xyz, rot6d, gripper]. rot6d is quat-sign
            # invariant, so the raw (flip-ridden) quats are safe to use directly.
            state_parts = [_pose7_to_xyz_rot6d(ee_pose), _gripper(st, "gripper_pos", "joint_pos")]
            if args.include_joints:
                state_parts.append(st["joint_pos"].astype(np.float32)[:, :7])
            state = np.concatenate(state_parts, axis=1)
            action = np.concatenate(
                [_pose7_to_xyz_rot6d(target_pose), _gripper(ac, "gripper_target", "joint_targets")], axis=1
            )
        else:
            # One canonicalization decision per episode, shared by state and action.
            state_quat, action_quat = _canonicalize_quats(ee_pose[:, :4], target_pose[:, :4], quat_ref)
            # State (29-D): [quat, xyz, gripper | joint_pos arm | joint_vel | wrench].
            state = np.concatenate(
                [
                    state_quat.astype(np.float32),
                    ee_pose[:, 4:7],
                    _gripper(st, "gripper_pos", "joint_pos"),
                    st["joint_pos"].astype(np.float32)[:, :7],
                    st["joint_vel"].astype(np.float32),
                    st["wrench"].astype(np.float32),
                ],
                axis=1,
            )
            # Action (8-D): target_pose (canonicalized) + gripper action.
            action = np.concatenate(
                [action_quat.astype(np.float32), target_pose[:, 4:7], _gripper(ac, "gripper_target", "joint_targets")],
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

    if quat_ref is not None:
        # The deploy client needs the same hemisphere convention for its live ee_pose
        # quats: flip the observed quat if dot(q, quat_reference) < 0. (quat8 only —
        # rot6d needs no convention.)
        (out_root / "quat_reference.json").write_text(
            json.dumps({"quat_reference_wxyz": quat_ref.tolist()}, indent=2)
        )
    logger.info("Done. %d episodes / %d frames at %s", len(episodes), total_frames, out_root)


if __name__ == "__main__":
    main()
