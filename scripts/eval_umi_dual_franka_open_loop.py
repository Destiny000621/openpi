"""Rank dual-Franka UMI pi0.5 checkpoints on held-out logical episodes.

Errors are computed in the exact normalized 20-D action space used for
training. Use a validation dataset materialized from source episodes that are
absent from the training split; see ``split_cardboard_box_lerobot_v21.py``.

Example:

    uv run python scripts/eval_umi_dual_franka_open_loop.py \
      --config-name pi05_umi_dual_franka_cardboard_box_relative \
      --repo-id local/cardboard_box_tcp_curated_logical_validation \
      --episodes all \
      --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_relative/v1/3000 \
      --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_relative/v1/4999
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np

import openpi.models.model as _model
from openpi.shared import nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logger = logging.getLogger(__name__)

NATIVE_ACTION_DIM = 20
TRANSLATION_DIMS = (0, 1, 2, 10, 11, 12)
ROTATION_6D_DIMS = (*range(3, 9), *range(13, 19))
GRIPPER_DIMS = (9, 19)


def parse_episodes(spec: str, total: int) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(total))
    result: list[int] = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            lower, upper = part.split("-", maxsplit=1)
            result.extend(range(int(lower), int(upper) + 1))
        else:
            result.append(int(part))
    result = list(dict.fromkeys(result))
    invalid = [episode for episode in result if not 0 <= episode < total]
    if invalid:
        raise ValueError(f"Episodes {invalid} are outside [0, {total})")
    return result


def build_eval_frames(
    data_config: _config.DataConfig,
    action_horizon: int,
    repo_id: str,
    episodes: list[int],
    stride: int,
    max_per_episode: int,
):
    """Create a raw chunked dataset and select only full chunks."""

    metadata = LeRobotDatasetMetadata(repo_id)
    delta_timestamps = {
        key: [step / metadata.fps for step in range(action_horizon)] for key in data_config.action_sequence_keys
    }
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    episode_index = dataset.episode_data_index
    frame_indices: list[int] = []
    for episode in episodes:
        start = int(episode_index["from"][episode])
        end = int(episode_index["to"][episode])
        last_start = end - action_horizon
        if last_start < start:
            logger.warning("Skipping logical episode %d: shorter than action horizon", episode)
            continue
        frame_indices.extend(list(range(start, last_start + 1, stride))[:max_per_episode])
    return dataset, frame_indices


def _collate(items: list[dict]) -> dict:
    return jax.tree.map(lambda *values: np.stack(values), *items)


def load_checkpoint_norm_stats(data_config: _config.DataConfig, checkpoint: Path):
    if data_config.asset_id is None:
        raise ValueError("Training config must define an asset_id for checkpoint normalization stats")
    norm_stats = _checkpoints.load_norm_stats(checkpoint / "assets", data_config.asset_id)
    if norm_stats is None:
        raise FileNotFoundError(f"Checkpoint {checkpoint} does not contain normalization stats")
    return norm_stats


def assert_matching_norm_stats(reference, candidate, *, reference_checkpoint: Path, candidate_checkpoint: Path) -> None:
    if set(reference) != set(candidate):
        raise ValueError(
            f"Checkpoint normalization keys differ: {reference_checkpoint} has {sorted(reference)}, "
            f"{candidate_checkpoint} has {sorted(candidate)}"
        )
    for key in sorted(reference):
        for field in ("mean", "std", "q01", "q99"):
            reference_value = getattr(reference[key], field)
            candidate_value = getattr(candidate[key], field)
            if reference_value is None or candidate_value is None:
                matches = reference_value is None and candidate_value is None
            else:
                matches = np.array_equal(np.asarray(reference_value), np.asarray(candidate_value))
            if not matches:
                raise ValueError(
                    "Cannot rank normalized MSE across checkpoints with different normalization stats: "
                    f"{reference_checkpoint} and {candidate_checkpoint} differ at {key}.{field}"
                )


def evaluate_checkpoint(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    norm_stats,
    checkpoint: Path,
    raw_dataset,
    repo_id: str,
    frame_indices: list[int],
    batch_size: int,
) -> np.ndarray:
    eval_data_config = dataclasses.replace(data_config, repo_id=repo_id, norm_stats=norm_stats)
    dataset = _data_loader.transform_dataset(raw_dataset, eval_data_config)
    model = config.model.load(_model.restore_params(checkpoint / "params", dtype=jnp.bfloat16))
    sample_actions = nnx_utils.module_jit(model.sample_actions)
    random_key = jax.random.key(0)
    squared_error_sum = np.zeros(NATIVE_ACTION_DIM, dtype=np.float64)
    sample_count = 0

    for start in range(0, len(frame_indices), batch_size):
        batch = _collate([dataset[index] for index in frame_indices[start : start + batch_size]])
        ground_truth = np.asarray(batch.pop("actions"))[..., :NATIVE_ACTION_DIM]
        observation = _model.Observation.from_dict(batch)
        random_key, sample_key = jax.random.split(random_key)
        prediction = np.asarray(sample_actions(sample_key, observation))[..., :NATIVE_ACTION_DIM]
        difference = prediction.astype(np.float64) - ground_truth.astype(np.float64)
        squared_error_sum += np.square(difference).sum(axis=(0, 1))
        sample_count += difference.shape[0] * difference.shape[1]
    return squared_error_sum / sample_count


def _mean_for_dims(mse: np.ndarray, dims: tuple[int, ...]) -> float:
    return float(mse[np.asarray(dims)].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--repo-id", required=True, help="held-out logical-episode LeRobot repo")
    parser.add_argument("--episodes", default="all", help="'all', '0-9', or comma/range combinations")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-per-episode", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    metadata = LeRobotDatasetMetadata(args.repo_id)
    episodes = parse_episodes(args.episodes, metadata.total_episodes)
    dataset, frame_indices = build_eval_frames(
        data_config,
        config.model.action_horizon,
        args.repo_id,
        episodes,
        args.stride,
        args.max_per_episode,
    )
    if not frame_indices:
        raise SystemExit("No full action chunks found in the requested logical episodes")

    logger.info(
        "config=%s repo=%s logical_episodes=%d frames=%d horizon=%d",
        args.config_name,
        args.repo_id,
        len(episodes),
        len(frame_indices),
        config.model.action_horizon,
    )
    rows = []
    reference_norm_stats = None
    reference_checkpoint = None
    for checkpoint in args.checkpoint:
        norm_stats = load_checkpoint_norm_stats(data_config, checkpoint)
        if reference_norm_stats is None:
            reference_norm_stats = norm_stats
            reference_checkpoint = checkpoint
        else:
            assert reference_checkpoint is not None
            assert_matching_norm_stats(
                reference_norm_stats,
                norm_stats,
                reference_checkpoint=reference_checkpoint,
                candidate_checkpoint=checkpoint,
            )
        mse = evaluate_checkpoint(
            config,
            data_config,
            norm_stats,
            checkpoint,
            dataset,
            args.repo_id,
            frame_indices,
            args.batch_size,
        )
        rows.append(
            (
                str(checkpoint),
                float(mse.mean()),
                _mean_for_dims(mse, TRANSLATION_DIMS),
                _mean_for_dims(mse, ROTATION_6D_DIMS),
                _mean_for_dims(mse, GRIPPER_DIMS),
            )
        )
    rows.sort(key=lambda row: row[1])

    print("\nNormalized open-loop action MSE (lower is better)")
    print(f"{'checkpoint':65s} {'all':>10s} {'xyz':>10s} {'rot6d':>10s} {'grip':>10s}")
    for checkpoint, overall, translation, rotation, gripper in rows:
        print(f"{checkpoint:65.65s} {overall:10.6f} {translation:10.6f} {rotation:10.6f} {gripper:10.6f}")


if __name__ == "__main__":
    main()
