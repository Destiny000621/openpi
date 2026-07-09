"""Merge multiple LeRobot v2.1 datasets (with optional per-episode selection) into one.

openpi's LeRobotAlohaDataConfig takes a single repo_id, so to train on a combination
of datasets you first materialize a single merged v2.1 dataset.

What it does:
  - Concatenates selected episodes from each source, re-indexed 0..N-1 in the order given.
  - Keeps only columns common to ALL sources (e.g. drops `phase`/`correction_index` that
    only one dataset has) so the per-episode parquets share one schema.
  - Rewrites each episode's `episode_index`, global `index`, and `task_index` columns.
    Tasks are unified across sources (union, first-seen order) and remapped per frame.
  - Symlinks the video files (resolved to the real mp4, so no symlink-to-symlink chains).
  - Writes v2.1 meta: info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl.
    (lerobot 0.1.0 aggregates dataset stats from episodes_stats.jsonl for v2.1, so no
    meta/stats.json is required.)

All sources must share fps, robot_type, and the shapes of the common features.

Usage:
    uv run python scripts/merge_lerobot_v21.py \
        --dst   /mnt/localssd/Sichang/lerobot_home/local/vials_4_aug_8ml46_v21 \
        --src   /mnt/localssd/Sichang/lerobot_home/local/vials_4_30fps_180_v21 --episodes all \
        --src   /mnt/localssd/Sichang/lerobot_home/local/8ml_vial_place_30fps_v21 \
                --episodes 0-40,165,167,168,172,177
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Columns that index the dataset; rewritten per output episode.
EPISODE_INDEX_COL = "episode_index"
GLOBAL_INDEX_COL = "index"
TASK_INDEX_COL = "task_index"


def parse_episodes(spec: str, total: int) -> list[int]:
    """Parse 'all' or a comma list of ints/ranges like '0-40,165,167' (ranges inclusive)."""
    if spec.strip() == "all":
        return list(range(total))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))  # inclusive
        else:
            out.append(int(part))
    # de-dup preserving order
    seen: set[int] = set()
    result = []
    for e in out:
        if e in seen:
            continue
        if not (0 <= e < total):
            raise ValueError(f"episode {e} out of range [0, {total})")
        seen.add(e)
        result.append(e)
    return result


def load_jsonl(path: Path) -> dict[int, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[int(d["episode_index"])] = d
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=Path, required=True, help="output merged v2.1 dataset root")
    parser.add_argument("--src", type=Path, action="append", required=True, help="source v2.1 dataset root (repeatable)")
    parser.add_argument(
        "--episodes",
        action="append",
        required=True,
        help="episode selection for the matching --src: 'all' or e.g. '0-40,165,167' (repeatable, paired by order)",
    )
    args = parser.parse_args()

    if len(args.src) != len(args.episodes):
        raise ValueError(f"got {len(args.src)} --src but {len(args.episodes)} --episodes; they must pair 1:1")

    srcs = [s.resolve() for s in args.src]
    dst = args.dst.resolve()

    # ── load source infos, determine common features/columns ──────────────
    infos = [json.loads((s / "meta/info.json").read_text()) for s in srcs]
    fps = {i["fps"] for i in infos}
    robots = {i.get("robot_type") for i in infos}
    if len(fps) != 1:
        raise ValueError(f"sources disagree on fps: {fps}")
    if len(robots) != 1:
        raise ValueError(f"sources disagree on robot_type: {robots}")

    common_features = set(infos[0]["features"])
    for i in infos[1:]:
        common_features &= set(i["features"])
    dropped = set().union(*[set(i["features"]) for i in infos]) - common_features
    if dropped:
        logger.info(f"Dropping non-common features: {sorted(dropped)}")
    # sanity: common features must share shape/dtype across sources
    for k in common_features:
        shapes = {tuple(i["features"][k].get("shape") or []) for i in infos}
        dtypes = {i["features"][k].get("dtype") for i in infos}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError(f"feature '{k}' differs across sources: shapes={shapes} dtypes={dtypes}")

    video_keys = [k for k in infos[0]["features"] if infos[0]["features"][k].get("dtype") == "video" and k in common_features]
    # parquet columns = common, non-video features
    keep_cols = [k for k in infos[0]["features"] if k in common_features and infos[0]["features"][k].get("dtype") != "video"]

    # ── unified task list ──────────────────────────────────────────────────
    global_tasks: list[str] = []
    task_to_global: dict[str, int] = {}

    def task_global_index(task: str) -> int:
        if task not in task_to_global:
            task_to_global[task] = len(global_tasks)
            global_tasks.append(task)
        return task_to_global[task]

    # per-source local task_index -> global index
    src_task_remap: list[dict[int, int]] = []
    for s in srcs:
        local = {}
        for line in (s / "meta/tasks.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            local[int(d["task_index"])] = task_global_index(str(d["task"]))
        src_task_remap.append(local)

    # ── prepare output dirs ────────────────────────────────────────────────
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for cam in video_keys:
        (dst / "videos" / "chunk-000" / cam).mkdir(parents=True, exist_ok=True)

    ep_meta_lines: list[str] = []
    ep_stats_lines: list[str] = []
    new_idx = 0
    global_offset = 0
    total_frames = 0

    for s, info, remap, ep_spec in zip(srcs, infos, src_task_remap, args.episodes, strict=True):
        # which episodes from this source
        sel = parse_episodes(ep_spec, info["total_episodes"])
        src_eps = load_jsonl(s / "meta/episodes.jsonl")
        src_stats = load_jsonl(s / "meta/episodes_stats.jsonl")
        logger.info(f"{s.name}: selecting {len(sel)} episodes -> output idx {new_idx}..{new_idx + len(sel) - 1}")

        for src_ep in sel:
            src_pq = s / f"data/chunk-000/episode_{src_ep:06d}.parquet"
            table = pq.read_table(src_pq)

            # keep only common, non-video columns, in canonical order
            table = table.select(keep_cols)
            n = table.num_rows

            # guard: one episode per parquet
            uniq = set(table.column(EPISODE_INDEX_COL).to_pylist())
            if uniq != {src_ep}:
                raise ValueError(f"{src_pq} holds episode_index {uniq}, expected {{{src_ep}}} (not 1-episode-per-file)")

            # rewrite index columns
            new_task_idx = [remap[t] for t in table.column(TASK_INDEX_COL).to_pylist()]
            table = table.set_column(
                table.schema.get_field_index(EPISODE_INDEX_COL), EPISODE_INDEX_COL, pa.array([new_idx] * n, pa.int64())
            )
            table = table.set_column(
                table.schema.get_field_index(GLOBAL_INDEX_COL),
                GLOBAL_INDEX_COL,
                pa.array(list(range(global_offset, global_offset + n)), pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index(TASK_INDEX_COL), TASK_INDEX_COL, pa.array(new_task_idx, pa.int64())
            )

            pq.write_table(table, dst / f"data/chunk-000/episode_{new_idx:06d}.parquet")

            # videos: symlink resolved real path
            for cam in video_keys:
                src_mp4 = s / f"videos/chunk-000/{cam}/episode_{src_ep:06d}.mp4"
                real = Path(os.path.realpath(src_mp4))
                link = dst / f"videos/chunk-000/{cam}/episode_{new_idx:06d}.mp4"
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(real)

            # meta lines
            ep_entry = src_eps[src_ep]
            length = int(ep_entry["length"])
            ep_meta_lines.append(json.dumps({"episode_index": new_idx, "tasks": ep_entry["tasks"], "length": length}))
            stats_entry = src_stats[src_ep]
            ep_stats_lines.append(json.dumps({"episode_index": new_idx, "stats": stats_entry["stats"]}))

            global_offset += n
            total_frames += n
            new_idx += 1

    total_episodes = new_idx

    # ── meta/tasks.jsonl ───────────────────────────────────────────────────
    with open(dst / "meta/tasks.jsonl", "w") as f:
        for ti, task in enumerate(global_tasks):
            f.write(json.dumps({"task_index": ti, "task": task}) + "\n")

    # ── meta/episodes.jsonl + episodes_stats.jsonl ─────────────────────────
    (dst / "meta/episodes.jsonl").write_text("\n".join(ep_meta_lines) + "\n")
    (dst / "meta/episodes_stats.jsonl").write_text("\n".join(ep_stats_lines) + "\n")

    # ── meta/info.json ─────────────────────────────────────────────────────
    info = dict(infos[0])
    info["codebase_version"] = "v2.1"
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    info["features"] = {k: v for k, v in infos[0]["features"].items() if k in common_features}
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(global_tasks)
    info["total_videos"] = total_episodes * len(video_keys)
    info["total_chunks"] = 1
    info["chunks_size"] = max(total_episodes, infos[0].get("chunks_size", 1000))
    info["splits"] = {"train": f"0:{total_episodes}"}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2))

    logger.info(
        f"Done. Merged dataset at {dst}\n"
        f"  episodes={total_episodes}  frames={total_frames}  tasks={len(global_tasks)} {global_tasks}"
    )


if __name__ == "__main__":
    main()
