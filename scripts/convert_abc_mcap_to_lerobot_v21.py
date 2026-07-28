# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mcap", "mcap-protobuf-support", "pyarrow", "tyro"]
# ///
"""Convert ABC-130k (XDOF) YAM MCAP episodes into a LeRobot v2.1 dataset for openpi.

Input:  <root>/episode_<uuid>/episode.mcap   (ABC-130k release format, see the
        dataset's docs/YAM_DATA_FORMAT.md)
Output: LeRobot v2.1 layout matching local/vials_4_30fps_180_v21:
        data/chunk-XXX/episode_NNNNNN.parquet
        videos/chunk-XXX/observation.images.{head,left_wrist,right_wrist}_camera/episode_NNNNNN.mp4
        meta/{info.json, episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, stats.json}

Alignment: fixed 30 Hz tick clock over [max stream start, min stream end],
causal floor matching (latest message at or before each tick) — same scheme as
the official abc-repo exporter (export_mcap.py).

State/action layout (14-D, matches limb OpenPIObsTransform state_keys):
  [left joints 0-5, left gripper, right joints 0-5, right gripper]
Gripper is the ABC normalized aperture (0 = closed, 1 = open).

Video: all cameras letterboxed to 640x480 h264 CFR 30 fps. ZED-X stereo top
episodes pick one eye deterministically (sha1(episode_id) parity, same rule
as the abc exporter); RealSense mono episodes use /top-camera directly.

Usage:
    uv run python scripts/convert_abc_mcap_to_lerobot_v21.py \
        --root /mnt/localssd/Sichang/abc_earbuds/raw/data/train/<task> \
        --out /mnt/localssd/Sichang/lerobot_home/local/<name>_v21 \
        --workers 32

Idempotent: episodes whose parquet + all three mp4s already exist are skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import traceback
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro

FPS = 30
TICK_NS = 33_333_333  # int(1e9 / 30)
OUT_W, OUT_H = 640, 480
CHUNK_SIZE = 1000

STATE_TOPICS = [("/left-arm-state", 6), ("/left-ee-state", 1), ("/right-arm-state", 6), ("/right-ee-state", 1)]
ACTION_TOPICS = [("/left-arm-action", 6), ("/left-ee-action", 1), ("/right-arm-action", 6), ("/right-ee-action", 1)]
WRIST_CAMERAS = [
    ("observation.images.left_wrist_camera", "/left-wrist-camera"),
    ("observation.images.right_wrist_camera", "/right-wrist-camera"),
]
TOP_KEY = "observation.images.head_camera"
DIM_NAMES = [f"{s}_joint_{i}" for s in ("left",) for i in range(6)] + ["left_gripper"] \
    + [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]

X264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-g", str(FPS), "-bf", "0",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "2"]


@dataclass
class Config:
    root: Path  # directory containing episode_<uuid>/ dirs
    out: Path  # output LeRobot v2.1 dataset root
    workers: int = 16
    max_episodes: int | None = None  # cap for smoke tests
    task_name: str | None = None  # override; default read from episode metadata


def floor_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    """Index of the latest source message at or before each target tick."""
    return np.clip(np.searchsorted(source_ts, target_ts, side="right") - 1, 0, len(source_ts) - 1)


def probe(path: str, *entries: str) -> list[int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", *entries, "-of", "csv=p=0", path],
        capture_output=True, text=True,
    ).stdout.strip()
    return [int(x) for x in out.split(",")]


def demuxer(fmt: str) -> str:
    """ffmpeg raw Annex-B demuxer name for a CompressedVideo.format value."""
    return {"h264": "h264", "h265": "hevc", "hevc": "hevc"}[fmt.lower()]


def encode_aligned(raw_path: str, fmt: str, width: int, height: int, needed: np.ndarray, out_path: str) -> None:
    """Decode an Annex-B stream, emit frame needed[i] at tick i (duplicating as required),
    letterbox to OUT_WxOUT_H, and encode CFR 30fps h264."""
    frame_bytes = width * height * 3
    vf = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=bicubic,"
          f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2")
    dec = subprocess.Popen(
        ["ffmpeg", "-f", demuxer(fmt), "-i", raw_path,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "error", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(FPS), "-i", "-", "-vf", vf, *X264, "-v", "error", out_path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    src_idx, frame = -1, None
    try:
        for wanted in needed:
            while src_idx < wanted:
                raw = dec.stdout.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                src_idx, frame = src_idx + 1, raw
            if frame is None:
                raise RuntimeError(f"decoder produced no frames (wanted index {wanted})")
            enc.stdin.write(frame)
    finally:
        dec.stdout.close(); dec.terminate(); dec.wait()
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")


def stat_dict(arr: np.ndarray) -> dict:
    """Per-episode stats for a (T, D) or (T,) array, lists like lerobot v2.1 expects."""
    a = arr.reshape(len(arr), -1).astype(np.float64)
    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "count": [len(a)],
    }


IMAGE_STAT_PLACEHOLDER = {
    "min": [[[0.0]], [[0.0]], [[0.0]]],
    "max": [[[255.0]], [[255.0]], [[255.0]]],
    "mean": [[[128.0]], [[128.0]], [[128.0]]],
    "std": [[[75.0]], [[75.0]], [[75.0]]],
}


def convert_episode(job: tuple[str, int, str]) -> dict | None:
    """Convert one episode.mcap. Returns metadata dict or None on skip/failure."""
    mcap_path, ep_idx, out_root = job
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    out = Path(out_root)
    chunk = ep_idx // CHUNK_SIZE
    pq_path = out / "data" / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
    vid_paths = {k: out / "videos" / f"chunk-{chunk:03d}" / k / f"episode_{ep_idx:06d}.mp4"
                 for k in [TOP_KEY, *(k for k, _ in WRIST_CAMERAS)]}
    ep_id = Path(mcap_path).parent.name

    try:
        cams: dict[str, list] = {}
        scalars: dict[str, list] = {}
        task_name = None
        cam_topics = {"/top-camera", "/top-left-camera", "/top-right-camera"} | {t for _, t in WRIST_CAMERAS}
        scalar_topics = {t for t, _ in STATE_TOPICS + ACTION_TOPICS}
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, channel, message, decoded in reader.iter_decoded_messages():
                if channel.topic in cam_topics:
                    cams.setdefault(channel.topic, []).append((message.log_time, decoded.data, decoded.format))
                elif channel.topic in scalar_topics:
                    scalars.setdefault(channel.topic, []).append(
                        (message.log_time, np.asarray(decoded.position, dtype=np.float64)))
                elif channel.topic == "/instruction":
                    task_name = decoded.data
        for msgs in (*cams.values(), *scalars.values()):
            msgs.sort(key=lambda x: x[0])

        missing = [t for t in scalar_topics if t not in scalars]
        if missing:
            return {"skip": f"{ep_id}: missing topics {missing}"}

        # Resolve top camera: stereo pick one eye deterministically, else mono.
        if "/top-left-camera" in cams and "/top-right-camera" in cams:
            top_topic = ("/top-left-camera" if hashlib.sha1(ep_id.encode()).digest()[0] % 2 == 0
                         else "/top-right-camera")
        elif "/top-camera" in cams:
            top_topic = "/top-camera"
        else:
            return {"skip": f"{ep_id}: no top camera"}
        active = [(TOP_KEY, top_topic)] + [(k, t) for k, t in WRIST_CAMERAS]
        for k, t in active:
            if t not in cams:
                return {"skip": f"{ep_id}: missing camera {t}"}

        # Fixed 30 Hz tick clock over the overlap of all used streams.
        streams = [cams[t] for _, t in active] + [scalars[t] for t, _ in STATE_TOPICS + ACTION_TOPICS]
        t0 = max(s[0][0] for s in streams)
        t_end = min(s[-1][0] for s in streams)
        ticks = np.arange(t0 + TICK_NS, t_end + 1, TICK_NS, dtype=np.int64)
        num_steps = len(ticks)
        if num_steps < 10:
            return {"skip": f"{ep_id}: too short ({num_steps} steps)"}

        # State / action arrays.
        def resample(topics: list[tuple[str, int]]) -> np.ndarray:
            parts = []
            for topic, dim in topics:
                msgs = scalars[topic]
                ts = np.array([t for t, _ in msgs], dtype=np.int64)
                vals = np.stack([v[:dim] for _, v in msgs])
                if vals.shape[1] != dim:
                    raise RuntimeError(f"{topic}: expected dim {dim}, got {vals.shape[1]}")
                parts.append(vals[floor_indices(ts, ticks)])
            return np.concatenate(parts, axis=-1).astype(np.float32)

        state = resample(STATE_TOPICS)
        action = resample(ACTION_TOPICS)

        # Videos.
        if not pq_path.exists() or not all(p.exists() for p in vid_paths.values()):
            for k, topic in active:
                msgs = cams[topic]
                fmt = msgs[0][2] or "h264"
                suffix = ".h264" if demuxer(fmt) == "h264" else ".hevc"
                cam_ts = np.array([t for t, _, _ in msgs], dtype=np.int64)
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(b"".join(data for _, data, _ in msgs))
                try:
                    width, height = probe(tmp.name, "-show_entries", "stream=width,height")
                    (n_frames,) = probe(tmp.name, "-count_packets", "-show_entries", "stream=nb_read_packets")
                    if n_frames > 0 and n_frames != len(cam_ts):  # chunks != frames; respace timestamps
                        cam_ts = np.linspace(cam_ts[0], cam_ts[-1], n_frames, dtype=np.int64)
                    vid_paths[k].parent.mkdir(parents=True, exist_ok=True)
                    tmp_out = str(vid_paths[k]) + ".part.mp4"
                    encode_aligned(tmp.name, fmt, width, height, floor_indices(cam_ts, ticks), tmp_out)
                    (n_out,) = probe(tmp_out, "-count_frames", "-show_entries", "stream=nb_read_frames")
                    if n_out != num_steps:
                        raise RuntimeError(f"{k}: {n_out} frames, expected {num_steps}")
                    os.replace(tmp_out, vid_paths[k])
                finally:
                    os.unlink(tmp.name)

        # Parquet (index/task_index are provisional 0; finalize pass rewrites them).
        table = pa.table({
            "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
            "episode_index": pa.array(np.full(num_steps, ep_idx, dtype=np.int64)),
            "frame_index": pa.array(np.arange(num_steps, dtype=np.int64)),
            "timestamp": pa.array((np.arange(num_steps) / FPS).astype(np.float32)),
            "index": pa.array(np.zeros(num_steps, dtype=np.int64)),
            "task_index": pa.array(np.zeros(num_steps, dtype=np.int64)),
        })
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, pq_path)

        return {
            "episode_index": ep_idx,
            "episode_id": ep_id,
            "length": num_steps,
            "task": task_name,
            "stats": {
                "observation.state": stat_dict(state),
                "action": stat_dict(action),
            },
        }
    except Exception:
        return {"skip": f"{ep_id}: ERROR\n{traceback.format_exc()}"}


def main(cfg: Config) -> None:
    mcaps = sorted(cfg.root.glob("episode_*/episode.mcap"))
    if cfg.max_episodes:
        mcaps = mcaps[: cfg.max_episodes]
    print(f"{len(mcaps)} episodes under {cfg.root}")
    cfg.out.mkdir(parents=True, exist_ok=True)

    jobs = [(str(p), i, str(cfg.out)) for i, p in enumerate(mcaps)]
    with Pool(cfg.workers) as pool:
        results = pool.imap_unordered(convert_episode, jobs)
        done, skipped = [], []
        for i, r in enumerate(results):
            if r is None or "skip" in (r or {}):
                skipped.append(r["skip"] if r else "None")
                print(f"[SKIP {i + 1}/{len(jobs)}] {(r or {}).get('skip', '')[:200]}", flush=True)
            else:
                done.append(r)
                print(f"[OK {i + 1}/{len(jobs)}] ep{r['episode_index']:06d} {r['episode_id']}: "
                      f"{r['length']} steps", flush=True)

    if not done:
        raise SystemExit("no episodes converted")
    done.sort(key=lambda r: r["episode_index"])

    # ── Re-pack episode indices to be contiguous (skips leave holes) ──────
    remap = {r["episode_index"]: new for new, r in enumerate(done)}
    for r in done:
        old, new = r["episode_index"], remap[r["episode_index"]]
        if old == new:
            continue
        for kind, key in [("data", None), *[("videos", k) for k in
                          [TOP_KEY, *(k for k, _ in WRIST_CAMERAS)]]]:
            oc, nc = old // CHUNK_SIZE, new // CHUNK_SIZE
            if kind == "data":
                src = cfg.out / "data" / f"chunk-{oc:03d}" / f"episode_{old:06d}.parquet"
                dst = cfg.out / "data" / f"chunk-{nc:03d}" / f"episode_{new:06d}.parquet"
            else:
                src = cfg.out / "videos" / f"chunk-{oc:03d}" / key / f"episode_{old:06d}.mp4"
                dst = cfg.out / "videos" / f"chunk-{nc:03d}" / key / f"episode_{new:06d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
        r["episode_index"] = new

    # ── Finalize parquets: global index + episode_index + task_index ──────
    task_name = cfg.task_name or next((r["task"] for r in done if r["task"]), "unknown task")
    global_idx = 0
    for r in done:
        ep_idx = r["episode_index"]
        p = cfg.out / "data" / f"chunk-{ep_idx // CHUNK_SIZE:03d}" / f"episode_{ep_idx:06d}.parquet"
        t = pq.read_table(p)
        n = t.num_rows
        t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index",
                         pa.array(np.full(n, ep_idx, dtype=np.int64)))
        t = t.set_column(t.schema.get_field_index("index"), "index",
                         pa.array(np.arange(global_idx, global_idx + n, dtype=np.int64)))
        pq.write_table(t, p)
        r["global_start"] = global_idx
        global_idx += n
    total_frames = global_idx

    # ── meta/ ──────────────────────────────────────────────────────────────
    meta = cfg.out / "meta"
    meta.mkdir(exist_ok=True)

    with open(meta / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_name}) + "\n")

    with open(meta / "episodes.jsonl", "w") as f_ep, open(meta / "episodes_stats.jsonl", "w") as f_st:
        for r in done:
            n = r["length"]
            f_ep.write(json.dumps({"episode_index": r["episode_index"], "tasks": [task_name],
                                   "length": n}) + "\n")
            stats = dict(r["stats"])
            for k in [TOP_KEY, *(k for k, _ in WRIST_CAMERAS)]:
                stats[k] = {**IMAGE_STAT_PLACEHOLDER, "count": [n]}
            stats["episode_index"] = stat_dict(np.full(n, r["episode_index"], dtype=np.float64))
            stats["frame_index"] = stat_dict(np.arange(n, dtype=np.float64))
            stats["timestamp"] = stat_dict(np.arange(n, dtype=np.float64) / FPS)
            stats["index"] = stat_dict(np.arange(r["global_start"], r["global_start"] + n, dtype=np.float64))
            stats["task_index"] = stat_dict(np.zeros(n, dtype=np.float64))
            f_st.write(json.dumps({"episode_index": r["episode_index"], "stats": stats}) + "\n")

    # Aggregate stats.json (state/action only, like the vial dataset).
    agg = {}
    for key in ("observation.state", "action"):
        mins = np.min([r["stats"][key]["min"] for r in done], axis=0)
        maxs = np.max([r["stats"][key]["max"] for r in done], axis=0)
        counts = np.array([r["stats"][key]["count"][0] for r in done], dtype=np.float64)
        means = np.array([r["stats"][key]["mean"] for r in done])
        stds = np.array([r["stats"][key]["std"] for r in done])
        w = counts / counts.sum()
        mean = (means * w[:, None]).sum(axis=0)
        var = ((stds**2 + (means - mean) ** 2) * w[:, None]).sum(axis=0)
        agg[key] = {"min": mins.tolist(), "max": maxs.tolist(), "mean": mean.tolist(),
                    "std": np.sqrt(var).tolist(), "count": [int(counts.sum())]}
    (meta / "stats.json").write_text(json.dumps(agg, indent=2))

    n_eps = len(done)
    feat_vec = {"dtype": "float32", "shape": [14], "names": DIM_NAMES, "fps": FPS}
    feat_scalar = lambda dt: {"dtype": dt, "shape": [1], "names": None, "fps": FPS}  # noqa: E731
    feat_video = {
        "dtype": "video", "shape": [OUT_H, OUT_W, 3], "names": ["height", "width", "channels"],
        "video_info": {"video.fps": FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                       "video.is_depth_map": False, "has_audio": False},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "yam",
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 3 * n_eps,
        "total_chunks": (n_eps + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": feat_vec,
            "action": feat_vec,
            "timestamp": feat_scalar("float32"),
            "frame_index": feat_scalar("int64"),
            "episode_index": feat_scalar("int64"),
            "index": feat_scalar("int64"),
            "task_index": feat_scalar("int64"),
            TOP_KEY: feat_video,
            **{k: feat_video for k, _ in WRIST_CAMERAS},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))

    # Episode-id map for traceability back to the HF uuids.
    (meta / "episode_ids.json").write_text(json.dumps(
        {r["episode_index"]: r["episode_id"] for r in done}, indent=1))

    print(f"\nDone: {n_eps} episodes, {total_frames} frames -> {cfg.out}")
    print(f"Skipped {len(skipped)}:")
    for s in skipped[:20]:
        print("  " + s.splitlines()[0])
    if skipped:
        (meta / "skipped.json").write_text(json.dumps(skipped, indent=1))


if __name__ == "__main__":
    main(tyro.cli(Config))
