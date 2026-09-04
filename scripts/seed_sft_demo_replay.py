"""Push APPROVED SFT demo insert segments into the SubRL replay (dc100 dataset).

Consumes the human-approved verdicts from mine_sft_insert_seeds.py
(franka_sft_seed_review/segments.json) and seeds each verified segment of the
first --max-episodes episodes as RLTTransition windows:

  - z_rl / ref_chunk : recomputed by the FROZEN policy per window anchor (~every
    15 ticks), inputs rebuilt exactly like the live client (10-D dataset state,
    side camera resize_with_pad 224, RAW wrist frame for the server-side crop).
  - action_chunk     : the honest EXECUTED residual of the teleop demo vs pi0.5's
    chunk — clip((demo_target_xyz - ref_chunk_xyz) / 0.01, ±1) per row (rot and
    gripper residual rows are 0, matching the live bound); rows past the segment
    end are 0. Demos become "expert corrections to pi0.5" for success-BC.
  - rewards          : terminal window's last executed row = 1; success=1 on all
    windows; done on the terminal window.

Run with the SERVE and LEARNER stopped:
    cd ~/Desktop/openpi
    SUBRL_RETURN_EMBED=1 SUBRL_EMBED_WEIGHTS=1,4,1 \
    SUBRL_RLT_SRC=~/Desktop/openpi-RLT/rlt_online_rl/src \
    uv run python scripts/seed_sft_demo_replay.py
"""

import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
import requests
import tyro

PROMPT = "Unplug the two cables from the right router, then insert them into the left router"
Z, C, A = 2048, 50, 10
BOUND_XYZ = 0.01


@dataclasses.dataclass
class Args:
    dataset: str = "~/Desktop/openpi/double_cable_100_r6_v21"
    review: str = "~/Desktop/SubRL/data/franka_sft_seed_review/segments.json"
    journal: str = "~/subrl_runs/dryrun_v0/replay_journal.pkl"
    config: str = "pi05_franka_double_cable_100_r6_rawrot_wcrop"
    dir: str = "~/.cache/openpi/hf/pi05_franka_double_cable_100_wcrop_10k"
    max_episodes: int = 50  # user 2026-09-03: the first 50 demo episodes are enough
    anchor_stride: int = 15
    replay_port: int = 9102
    episode_id_base: int = 100  # keep seed ids clear of the eval seeds (0-3)


def _decode_anchor_frames(video: pathlib.Path, start: int, end: int, stride: int) -> np.ndarray:
    """Decode ONLY the anchor frames of [start, end] in one ffmpeg pass (AV1)."""
    sel = f"between(n\\,{start}\\,{end})*not(mod(n-{start}\\,{stride}))"
    proc = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-c:v", "libdav1d", "-i", str(video),
            "-vf", f"select={sel}", "-vsync", "0",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True,
        capture_output=True,
    )
    w, h = 640, 360
    frames = np.frombuffer(proc.stdout, np.uint8)
    n = len(frames) // (h * w * 3)
    return frames[: n * h * w * 3].reshape(n, h, w, 3)


def main(args: Args) -> None:  # noqa: PLR0915
    assert os.environ.get("SUBRL_RETURN_EMBED") == "1", "run with SUBRL_RETURN_EMBED=1"
    assert os.environ.get("SUBRL_EMBED_WEIGHTS"), "run with SUBRL_EMBED_WEIGHTS (e.g. 1,4,1)"
    ds = pathlib.Path(args.dataset).expanduser()
    journal = pathlib.Path(args.journal).expanduser()
    journal.parent.mkdir(parents=True, exist_ok=True)
    segs = [
        s
        for s in json.loads(pathlib.Path(args.review).expanduser().read_text())
        if s.get("verified") and s.get("episode", 9999) < args.max_episodes
    ]
    print(f"{len(segs)} approved segments (episodes < {args.max_episodes})")
    assert segs, "nothing to push"

    from openpi_client import image_tools

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), pathlib.Path(args.dir).expanduser()
    )

    sys.path.insert(0, str(pathlib.Path("~/Desktop/openpi/packages/openpi-client/src").expanduser()))
    from openpi_client import msgpack_numpy

    replay = subprocess.Popen(
        [
            str(pathlib.Path.home() / "venvs/subrl/bin/python"),
            str(pathlib.Path.home() / "Desktop/SubRL/real_robot/services/serve_replay.py"),
            "--port", str(args.replay_port), "--journal", str(journal),
        ]
    )
    base = f"http://127.0.0.1:{args.replay_port}"
    try:
        for _ in range(30):
            try:
                if requests.get(f"{base}/healthz", timeout=1).ok:
                    break
            except Exception:  # noqa: BLE001, S110, PERF203
                time.sleep(0.5)
        else:
            raise RuntimeError("replay did not come up")
        start_size = requests.get(f"{base}/stats", timeout=5).json()["size"]

        total = 0
        for c in segs:
            ep = c["episode"]
            t0, t1 = c["start_tick"], c["end_tick"]
            df = pd.read_parquet(
                ds / f"data/chunk-000/episode_{ep:06d}.parquet",
                columns=["observation.state", "action"],
            )
            st = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
            act = np.stack(df["action"].to_numpy()).astype(np.float32)
            exec_xyz = st[:, :3] + act[:, :3]  # absolute demo target per tick

            wrist = _decode_anchor_frames(
                ds / f"videos/chunk-000/observation.images.camera0/episode_{ep:06d}.mp4",
                t0, t1, args.anchor_stride,
            )
            side = _decode_anchor_frames(
                ds / f"videos/chunk-000/observation.images.camera1/episode_{ep:06d}.mp4",
                t0, t1, args.anchor_stride,
            )
            anchors = list(range(t0, t1 + 1, args.anchor_stride))[: min(len(wrist), len(side))]

            feats = []
            for k, t in enumerate(anchors):
                out = policy.infer(
                    {
                        "observation/state": st[t],
                        "observation/image": image_tools.resize_with_pad(side[k], 224, 224),
                        "observation/wrist_image": wrist[k],
                        "prompt": PROMPT,
                    }
                )
                ref = np.asarray(out["actions"], np.float32)[:C, :A]
                res = np.zeros((C, A), np.float32)
                for i in range(C):
                    ti = t + i
                    if ti > t1:
                        break
                    res[i, :3] = np.clip((exec_xyz[ti] - ref[i, :3]) / BOUND_XYZ, -1.0, 1.0)
                feats.append(
                    {
                        "z_rl": np.asarray(out["image_embedding"], np.float32),
                        "proprio": st[t],
                        "ref_chunk": ref,
                        "action_chunk": res,
                    }
                )
            windows = []
            for k, f in enumerate(feats):
                last = k == len(feats) - 1
                nxt = feats[k + 1] if not last else f
                rew = np.zeros(C, np.float32)
                if last:
                    rew[max(0, min(t1 - anchors[k], C - 1))] = 1.0
                windows.append(
                    {
                        "z_rl": f["z_rl"],
                        "proprio": f["proprio"],
                        "ref_chunk": f["ref_chunk"],
                        "action_chunk": f["action_chunk"],
                        "rewards": rew,
                        "done": bool(last),
                        "next_z_rl": nxt["z_rl"],
                        "next_proprio": nxt["proprio"],
                        "next_ref_chunk": nxt["ref_chunk"],
                        "source": 1,
                        "collection_phase": "online_rl",
                        "success": 1,
                        "intervention_flag": False,
                        "episode_id": args.episode_id_base + ep,
                        "step_id": k,
                    }
                )
            r = requests.post(
                f"{base}/extend",
                data=msgpack_numpy.Packer().pack({"transitions": windows}),
                headers={"Content-Type": "application/octet-stream"},
                timeout=60,
            )
            r.raise_for_status()
            total += len(windows)
            mean_res = float(np.mean([np.abs(w["action_chunk"][:, :3]).mean() for w in windows]))
            print(
                f"  seeded ep{ep:03d} t{t0}-{t1}: {len(windows)} windows "
                f"(mean |xyz residual| {mean_res:.2f})",
                flush=True,
            )
        stats = requests.get(f"{base}/stats", timeout=5).json()
        print(
            f"\nSEEDED {len(segs)} demo episodes / {total} windows "
            f"(replay size {start_size} -> {stats['size']}, journal {journal})"
        )
    finally:
        replay.terminate()
        try:
            replay.wait(timeout=10)
        except subprocess.TimeoutExpired:
            replay.kill()


if __name__ == "__main__":
    main(tyro.cli(Args))
