"""Convert a LeRobot v3.0 dataset to v2.1 layout (for old lerobot consumers like openpi).

Layout differences:
  v3.0: data/chunk-000/file-NNN.parquet, videos/{cam}/chunk-000/file-NNN.mp4,
        meta/episodes/chunk-000/file-000.parquet (per-ep stats inline)
  v2.1: data/chunk-000/episode_NNNNNN.parquet, videos/{cam}/chunk-000/episode_NNNNNN.mp4,
        meta/episodes.jsonl, meta/episodes_stats.jsonl, meta/tasks.jsonl

Symlinks parquets/videos in place; only writes new metadata files. Idempotent.

Usage:
    uv run python scripts/convert_v3_to_v21.py \
        --src ~/.cache/huggingface/lerobot/ttotmoon/place_the_vial_into_the_stand_1to4 \
        --dst /mnt/localssd/sunlingfeng/datasets/place_the_vial_v21
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from loguru import logger


def _to_serializable(v):
    if isinstance(v, np.ndarray):
        # Image stats land here as object arrays of ndarrays — flatten one level.
        if v.dtype == object:
            v = np.array([_to_serializable(x) for x in v])
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, list):
        return [_to_serializable(x) for x in v]
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="v3.0 dataset root")
    parser.add_argument("--dst", type=Path, required=True, help="output v2.1 dataset root")
    args = parser.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    # ── meta/info.json ────────────────────────────────────────────────────
    src_info = json.loads((src / "meta/info.json").read_text())
    info = dict(src_info)
    info["codebase_version"] = "v2.1"
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["total_chunks"] = 1
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2))
    logger.info(f"Wrote {dst}/meta/info.json (codebase=v2.1)")

    # ── meta/stats.json (top-level aggregate, already in v3.0 cache) ────
    if (src / "meta/stats.json").exists():
        shutil.copy2(src / "meta/stats.json", dst / "meta/stats.json")
        logger.info("Copied meta/stats.json")

    # ── meta/tasks.jsonl ────────────────────────────────────────────────
    # In v3.0, tasks.parquet uses `task` as the row index and `task_index` as a column.
    tasks_df = pq.read_table(src / "meta/tasks.parquet").to_pandas().reset_index()
    with open(dst / "meta/tasks.jsonl", "w") as f:
        for _, row in tasks_df.iterrows():
            f.write(json.dumps({"task_index": int(row["task_index"]), "task": str(row["task"])}) + "\n")
    logger.info(f"Wrote meta/tasks.jsonl ({len(tasks_df)} tasks)")

    # ── meta/episodes.jsonl + meta/episodes_stats.jsonl ────────────────
    # Scan all shards under meta/episodes/chunk-*/file-*.parquet — datasets with
    # >chunks_size episodes (v3 default 1000) get rotated across multiple files.
    ep_shards = sorted((src / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not ep_shards:
        raise FileNotFoundError(f"No meta/episodes/chunk-*/file-*.parquet under {src}")
    import pandas as pd

    ep_df = pd.concat([pq.read_table(p).to_pandas() for p in ep_shards], ignore_index=True)
    ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)
    logger.info(f"Loaded {len(ep_df)} episodes from {len(ep_shards)} metadata shard(s)")
    n_eps = len(ep_df)

    feature_keys = [k for k in info["features"].keys() if k not in {"index", "frame_index", "episode_index", "task_index", "timestamp"}]
    # Include scalar columns too — lerobot 0.1.0 expects stats for all feature-like columns.
    extra_stat_keys = ["episode_index", "frame_index", "timestamp", "index", "task_index"]

    with open(dst / "meta/episodes.jsonl", "w") as f_ep, \
         open(dst / "meta/episodes_stats.jsonl", "w") as f_stats:
        for _, row in ep_df.iterrows():
            ep_idx = int(row["episode_index"])
            tasks = [str(t) for t in row["tasks"]]
            length = int(row["length"])
            f_ep.write(json.dumps({
                "episode_index": ep_idx,
                "tasks": tasks,
                "length": length,
            }) + "\n")

            stats = {}
            for key in [*feature_keys, *extra_stat_keys]:
                s = {}
                for stat_name in ("min", "max", "mean", "std", "count"):
                    col = f"stats/{key}/{stat_name}"
                    if col in ep_df.columns:
                        s[stat_name] = _to_serializable(row[col])
                if s:
                    stats[key] = s
            f_stats.write(json.dumps({"episode_index": ep_idx, "stats": stats}) + "\n")

    logger.info(f"Wrote meta/episodes.jsonl + meta/episodes_stats.jsonl ({n_eps} episodes)")

    # ── data/chunk-000/episode_NNNNNN.parquet (symlink from file-NNN.parquet) ──
    for ep_idx in range(n_eps):
        chunk = int(ep_df.iloc[ep_idx]["data/chunk_index"])
        fidx = int(ep_df.iloc[ep_idx]["data/file_index"])
        src_pq = src / f"data/chunk-{chunk:03d}/file-{fidx:03d}.parquet"
        dst_pq = dst / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        if not dst_pq.exists():
            dst_pq.symlink_to(src_pq)
    logger.info(f"Symlinked {n_eps} parquet files (data/chunk-XXX/episode_NNNNNN.parquet)")

    # ── videos/chunk-XXX/{cam}/episode_NNNNNN.mp4 (symlink) ──────────────
    cam_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    for cam in cam_keys:
        (dst / f"videos/chunk-000/{cam}").mkdir(parents=True, exist_ok=True)
        for ep_idx in range(n_eps):
            chunk = int(ep_df.iloc[ep_idx][f"videos/{cam}/chunk_index"])
            fidx = int(ep_df.iloc[ep_idx][f"videos/{cam}/file_index"])
            src_mp4 = src / f"videos/{cam}/chunk-{chunk:03d}/file-{fidx:03d}.mp4"
            dst_mp4 = dst / f"videos/chunk-{chunk:03d}/{cam}/episode_{ep_idx:06d}.mp4"
            if not dst_mp4.exists():
                dst_mp4.symlink_to(src_mp4)
    logger.info(f"Symlinked videos for {len(cam_keys)} cameras × {n_eps} episodes")

    logger.info(f"Done. v2.1 dataset at: {dst}")


if __name__ == "__main__":
    main()
