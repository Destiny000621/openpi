"""Split multi-box LeRobot v2.1 recordings into one logical episode per box.

The curated cardboard-box dataset can contain several physical box assemblies in
one source episode. OpenPI samples action chunks inside LeRobot episode
boundaries, so training directly on those source episodes can produce chunks
that cross from the end of one box to the reset/start of the next box.

This script materializes frame-accurate logical episodes from a reviewed JSON
manifest. It rewrites the tabular episode/index/timestamp fields, replaces the
last action with a no-op absolute EEF target, trims both videos, and records
source provenance. Run it separately for train/validation/test. The validator
requires every logical box from one source episode to stay in the same split.

Manifest (``end_frame`` is exclusive)::

    {
      "version": 1,
      "source_repo_id": "byang11259/cardboard_box_tcp_curated",
      "segments": [
        {
          "source_episode": 0,
          "box_index": 0,
          "start_frame": 120,
          "end_frame": 2540,
          "split": "train",
          "task": "Assemble the cardboard box and put it into the bin"
        }
      ]
    }

Example::

    uv run python scripts/split_cardboard_box_lerobot_v21.py \
      --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
      --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_logical_train" \
      --manifest configs/cardboard_box_segments.json \
      --split train \
      --action-horizon 50
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
import itertools
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

DEFAULT_TASK = "Assemble the cardboard box and put it into the bin"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class Segment:
    """One physical cardboard-box trajectory inside a source episode."""

    source_episode: int
    box_index: int
    start_frame: int
    end_frame: int
    split: str
    task: str | None = None

    @property
    def length(self) -> int:
        return self.end_frame - self.start_frame


def _load_jsonl_by_index(path: Path, key: str = "episode_index") -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        index = int(record[key])
        if index in records:
            raise ValueError(f"Duplicate {key}={index} in {path}:{line_number}")
        records[index] = record
    return records


def load_manifest(path: Path) -> tuple[str | None, list[Segment]]:
    """Load and minimally type-check a logical-box segmentation manifest."""

    payload = json.loads(path.read_text())
    if int(payload.get("version", -1)) != MANIFEST_VERSION:
        raise ValueError(f"{path}: expected manifest version {MANIFEST_VERSION}, got {payload.get('version')}")
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError(f"{path}: 'segments' must be a non-empty list")

    required = {"source_episode", "box_index", "start_frame", "end_frame", "split"}
    segments: list[Segment] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: segment {index} must be an object")
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"{path}: segment {index} is missing {sorted(missing)}")
        segments.append(
            Segment(
                source_episode=int(raw["source_episode"]),
                box_index=int(raw["box_index"]),
                start_frame=int(raw["start_frame"]),
                end_frame=int(raw["end_frame"]),
                split=str(raw["split"]),
                task=str(raw["task"]) if raw.get("task") is not None else None,
            )
        )

    source_repo_id = payload.get("source_repo_id")
    return str(source_repo_id) if source_repo_id is not None else None, segments


def validate_segments(
    segments: list[Segment],
    episode_lengths: dict[int, int],
    *,
    action_horizon: int,
) -> None:
    """Validate ranges, logical IDs, overlap, and grouped data splitting."""

    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")

    source_splits: dict[int, set[str]] = defaultdict(set)
    source_ranges: dict[int, list[Segment]] = defaultdict(list)
    seen_box_ids: set[tuple[int, int]] = set()

    for segment in segments:
        if segment.source_episode not in episode_lengths:
            raise ValueError(f"Unknown source_episode={segment.source_episode}")
        source_length = episode_lengths[segment.source_episode]
        if not 0 <= segment.start_frame < segment.end_frame <= source_length:
            raise ValueError(
                f"Invalid range for source episode {segment.source_episode}, box {segment.box_index}: "
                f"[{segment.start_frame}, {segment.end_frame}) with source length {source_length}"
            )
        if segment.length < action_horizon + 1:
            raise ValueError(
                f"Source episode {segment.source_episode}, box {segment.box_index} has only {segment.length} frames; "
                f"require at least action_horizon + 1 = {action_horizon + 1}"
            )
        if not segment.split.strip():
            raise ValueError(f"Empty split for source episode {segment.source_episode}, box {segment.box_index}")
        box_id = (segment.source_episode, segment.box_index)
        if box_id in seen_box_ids:
            raise ValueError(f"Duplicate logical box id {box_id}")
        seen_box_ids.add(box_id)
        source_splits[segment.source_episode].add(segment.split)
        source_ranges[segment.source_episode].append(segment)

    split_leaks = {episode: sorted(splits) for episode, splits in source_splits.items() if len(splits) != 1}
    if split_leaks:
        raise ValueError(
            "Every logical box from a source episode must stay in one split to prevent collection-session leakage: "
            f"{split_leaks}"
        )

    for source_episode, ranges in source_ranges.items():
        ordered = sorted(ranges, key=lambda item: item.start_frame)
        for previous, current in itertools.pairwise(ordered):
            if previous.end_frame > current.start_frame:
                raise ValueError(
                    f"Overlapping segments in source episode {source_episode}: "
                    f"box {previous.box_index} [{previous.start_frame}, {previous.end_frame}) and "
                    f"box {current.box_index} [{current.start_frame}, {current.end_frame})"
                )


def _format_dataset_path(template: str, episode_index: int, chunks_size: int, *, video_key: str | None = None) -> str:
    fields: dict[str, Any] = {
        "episode_chunk": episode_index // chunks_size,
        "episode_index": episode_index,
    }
    if video_key is not None:
        fields["video_key"] = video_key
    return template.format(**fields)


def _replace_column(table: pa.Table, name: str, values: Any) -> pa.Table:
    field_index = table.schema.get_field_index(name)
    if field_index < 0:
        return table
    column_type = table.schema.field(field_index).type
    return table.set_column(field_index, name, pa.array(values, type=column_type))


def _as_numeric_matrix(table: pa.Table, name: str) -> np.ndarray:
    values = np.asarray(table.column(name).to_pylist())
    if values.dtype.kind not in "biufc":
        raise TypeError(f"{name} must be numeric, got dtype {values.dtype}")
    return values


def slice_and_rewrite_table(
    source_table: pa.Table,
    segment: Segment,
    *,
    output_episode: int,
    global_index_offset: int,
    output_task_index: int,
    fps: float,
    action_tolerance: float = 1e-5,
) -> pa.Table:
    """Slice a logical episode and reset every LeRobot indexing column."""

    required_columns = {
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "index",
        "task_index",
        "timestamp",
    }
    missing = required_columns - set(source_table.column_names)
    if missing:
        raise ValueError(f"Source table is missing required columns: {sorted(missing)}")

    frame_indices = np.asarray(source_table.column("frame_index").to_pylist(), dtype=np.int64)
    expected = np.arange(source_table.num_rows, dtype=np.int64)
    if not np.array_equal(frame_indices, expected):
        raise ValueError("Source frame_index must be contiguous 0..N-1 before logical episode slicing")

    table = source_table.slice(segment.start_frame, segment.length)
    states = _as_numeric_matrix(table, "observation.state")
    actions = _as_numeric_matrix(table, "action")
    if states.shape != actions.shape:
        raise ValueError(f"state/action shape mismatch in logical segment: {states.shape} vs {actions.shape}")
    if states.ndim != 2 or states.shape[1] != 16:
        raise ValueError(f"Expected raw dual-EEF state/action shape [T, 16], got {states.shape}")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError("Logical segment state/action values must all be finite")
    if len(states) > 1 and not np.allclose(actions[:-1], states[1:], atol=action_tolerance, rtol=0.0):
        max_error = float(np.max(np.abs(actions[:-1] - states[1:])))
        raise ValueError(
            "Dataset does not satisfy action[t] == observation.state[t+1] inside the selected segment; "
            f"maximum error is {max_error:.6g}"
        )

    # The source action at the final selected frame points beyond this logical
    # episode. Replace it with an absolute no-op target so LeRobot's terminal
    # action padding cannot cross into the next physical box.
    actions[-1] = states[-1]
    table = _replace_column(table, "action", actions.tolist())

    length = table.num_rows
    table = _replace_column(table, "episode_index", [output_episode] * length)
    table = _replace_column(table, "frame_index", range(length))
    table = _replace_column(table, "index", range(global_index_offset, global_index_offset + length))
    table = _replace_column(table, "task_index", [output_task_index] * length)
    table = _replace_column(table, "timestamp", np.arange(length, dtype=np.float64) / fps)
    if "task" in table.column_names:
        table = _replace_column(table, "task", [segment.task or DEFAULT_TASK] * length)
    if "next.done" in table.column_names:
        dones = [False] * length
        dones[-1] = True
        table = _replace_column(table, "next.done", dones)
    return table


def _numpy_to_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, dict):
        return {key: _numpy_to_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_numpy_to_json(item) for item in value]
    return value


def compute_table_stats(table: pa.Table, features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute LeRobot-compatible stats for numeric, non-video columns."""

    stats: dict[str, dict[str, Any]] = {}
    for key, feature in features.items():
        if key not in table.column_names or feature.get("dtype") in {"string", "image", "video"}:
            continue
        values = np.asarray(table.column(key).to_pylist())
        if values.dtype.kind not in "biufc":
            continue
        keepdims = values.ndim == 1
        stats[key] = _numpy_to_json(
            {
                "min": np.min(values, axis=0, keepdims=keepdims),
                "max": np.max(values, axis=0, keepdims=keepdims),
                "mean": np.mean(values, axis=0, keepdims=keepdims),
                "std": np.std(values, axis=0, keepdims=keepdims),
                "count": np.asarray([len(values)], dtype=np.int64),
            }
        )
    return stats


def _find_ffmpeg(explicit_binary: str | None) -> str:
    if explicit_binary:
        return explicit_binary
    if binary := shutil.which("ffmpeg"):
        return binary
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("ffmpeg is required to trim logical-episode videos") from exc


def trim_video(
    source: Path,
    destination: Path,
    *,
    start_frame: int,
    end_frame: int,
    ffmpeg_binary: str,
    crf: int,
) -> None:
    """Frame-accurately trim and re-encode one policy video."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    video_filter = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    subprocess.run(command, check=True)


def count_video_frames(path: Path) -> int:
    """Decode a video to verify exact logical-episode frame count."""

    import av

    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def _write_derived_readme(
    destination: Path,
    *,
    source_repo_id: str | None,
    split: str,
    num_segments: int,
    manifest_path: Path,
) -> None:
    source = source_repo_id or "local source dataset"
    text = f"""# Logical cardboard-box episodes ({split})

Generated from `{source}` by `scripts/split_cardboard_box_lerobot_v21.py`.

- Split: `{split}`
- Logical episodes (one physical box each): {num_segments}
- Boundary manifest: `meta/logical_segments_manifest.json` (copied from `{manifest_path.name}`)
- Final action of each logical episode: absolute no-op (`action[-1] = observation.state[-1]`)
- Videos: frame-accurately trimmed and re-encoded
- Episode stats: numeric fields recomputed; cropped-video stats intentionally omitted

Do not mix logical episodes from the same source episode across train,
validation, and test. See `meta/logical_segments.jsonl` for provenance.
"""
    (destination / "README.md").write_text(text)


def materialize(args: argparse.Namespace) -> None:
    source = args.src.resolve()
    destination = args.dst.resolve()
    info = json.loads((source / "meta/info.json").read_text())
    source_episodes = _load_jsonl_by_index(source / "meta/episodes.jsonl")
    source_repo_id, all_segments = load_manifest(args.manifest)
    episode_lengths = {index: int(record["length"]) for index, record in source_episodes.items()}
    validate_segments(all_segments, episode_lengths, action_horizon=args.action_horizon)

    if source_repo_id and source_repo_id != info.get("repo_id") and info.get("repo_id") is not None:
        logger.warning("Manifest source_repo_id=%s differs from info.json repo_id=%s", source_repo_id, info["repo_id"])

    segments = [segment for segment in all_segments if segment.split == args.split]
    if not segments:
        raise ValueError(f"Manifest contains no segments for split {args.split!r}")
    segments.sort(key=lambda segment: (segment.source_episode, segment.start_frame, segment.box_index))

    logger.info(
        "Validated %d total logical boxes; materializing %d boxes for split %s",
        len(all_segments),
        len(segments),
        args.split,
    )
    if args.dry_run:
        return
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination}")

    features = info["features"]
    video_keys = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    chunks_size = int(info.get("chunks_size", 1000))
    data_template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_template = info.get(
        "video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    )
    ffmpeg_binary = _find_ffmpeg(args.ffmpeg_bin)

    tasks: list[str] = []
    task_to_index: dict[str, int] = {}

    def task_index(task: str) -> int:
        if task not in task_to_index:
            task_to_index[task] = len(tasks)
            tasks.append(task)
        return task_to_index[task]

    episode_lines: list[str] = []
    episode_stats_lines: list[str] = []
    provenance_lines: list[str] = []
    global_index_offset = 0

    for output_episode, segment in enumerate(segments):
        task = segment.task
        if task is None:
            source_tasks = source_episodes[segment.source_episode].get("tasks", [])
            task = str(source_tasks[0]) if source_tasks else DEFAULT_TASK
        resolved_segment = Segment(**{**asdict(segment), "task": task})
        output_task_index = task_index(task)

        source_data_path = source / _format_dataset_path(data_template, segment.source_episode, chunks_size)
        source_table = pq.read_table(source_data_path)
        output_table = slice_and_rewrite_table(
            source_table,
            resolved_segment,
            output_episode=output_episode,
            global_index_offset=global_index_offset,
            output_task_index=output_task_index,
            fps=float(info["fps"]),
        )

        output_data_path = destination / _format_dataset_path(
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            output_episode,
            chunks_size,
        )
        output_data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(output_table, output_data_path)

        for video_key in video_keys:
            source_video_path = source / _format_dataset_path(
                video_template, segment.source_episode, chunks_size, video_key=video_key
            )
            output_video_path = destination / _format_dataset_path(
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                output_episode,
                chunks_size,
                video_key=video_key,
            )
            trim_video(
                source_video_path,
                output_video_path,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame,
                ffmpeg_binary=ffmpeg_binary,
                crf=args.video_crf,
            )
            decoded_frames = count_video_frames(output_video_path)
            if decoded_frames != segment.length:
                raise ValueError(
                    f"{output_video_path} contains {decoded_frames} frames; expected exactly {segment.length}"
                )

        logical_stats = compute_table_stats(output_table, features)
        # Video stats from the full source episode would be wrong after
        # cropping/re-encoding, so omit them. This OpenPI adapter computes its
        # training norm stats only from transformed state/actions.

        episode_lines.append(json.dumps({"episode_index": output_episode, "tasks": [task], "length": segment.length}))
        episode_stats_lines.append(json.dumps({"episode_index": output_episode, "stats": logical_stats}))
        provenance_lines.append(
            json.dumps(
                {
                    "episode_index": output_episode,
                    **asdict(resolved_segment),
                    "source_repo_id": source_repo_id,
                }
            )
        )
        global_index_offset += segment.length
        logger.info(
            "Logical episode %d <- source %d box %d frames [%d, %d)",
            output_episode,
            segment.source_episode,
            segment.box_index,
            segment.start_frame,
            segment.end_frame,
        )

    metadata_dir = destination / "meta"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": index, "task": task}) + "\n" for index, task in enumerate(tasks))
    )
    (metadata_dir / "episodes.jsonl").write_text("\n".join(episode_lines) + "\n")
    (metadata_dir / "episodes_stats.jsonl").write_text("\n".join(episode_stats_lines) + "\n")
    (metadata_dir / "logical_segments.jsonl").write_text("\n".join(provenance_lines) + "\n")
    shutil.copy2(args.manifest, metadata_dir / "logical_segments_manifest.json")

    output_info = dict(info)
    output_info["codebase_version"] = "v2.1"
    output_info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    output_info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    output_info["total_episodes"] = len(segments)
    output_info["total_frames"] = global_index_offset
    output_info["total_tasks"] = len(tasks)
    output_info["total_videos"] = len(segments) * len(video_keys)
    output_info["total_chunks"] = math.ceil(len(segments) / chunks_size)
    output_info["splits"] = {args.split: f"0:{len(segments)}"}
    (metadata_dir / "info.json").write_text(json.dumps(output_info, indent=2) + "\n")
    _write_derived_readme(
        destination,
        source_repo_id=source_repo_id,
        split=args.split,
        num_segments=len(segments),
        manifest_path=args.manifest,
    )
    logger.info("Wrote %s with %d logical episodes and %d frames", destination, len(segments), global_index_offset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="source LeRobot v2.1 dataset root")
    parser.add_argument("--dst", type=Path, required=True, help="empty destination for one materialized split")
    parser.add_argument("--manifest", type=Path, required=True, help="reviewed JSON logical-box boundary manifest")
    parser.add_argument("--split", required=True, help="manifest split to materialize, e.g. train, val, or test")
    parser.add_argument("--action-horizon", type=int, default=50, help="minimum logical episode length guard")
    parser.add_argument("--video-crf", type=int, default=18, help="H.264 constant-rate factor for trimmed videos")
    parser.add_argument("--ffmpeg-bin", help="explicit ffmpeg executable; defaults to PATH or imageio-ffmpeg")
    parser.add_argument("--dry-run", action="store_true", help="validate the complete manifest without writing files")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    materialize(parse_args())


if __name__ == "__main__":
    main()
