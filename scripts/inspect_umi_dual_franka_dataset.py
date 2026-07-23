"""CPU preflight for a dual-Franka UMI LeRobot dataset.

This inspects raw LeRobot fields, verifies ``action[t] == state[t+1]``, runs
the selected OpenPI data transform without normalization/tokenization, and
round-trips model-space 20-D actions back to raw absolute 16-D TCP targets.
It works with either one-box logical episodes or the original long episodes.
For long episodes it cannot infer physical box boundaries, so it does not
certify that a sampled 50-step chunk stays within one box instance.

Example:

    uv run python scripts/inspect_umi_dual_franka_dataset.py \
      --config-name pi05_umi_dual_franka_cardboard_box_relative \
      --repo-id local/cardboard_box_tcp_curated_logical_train
"""

from __future__ import annotations

import argparse
import dataclasses
import logging

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np

from openpi import transforms as _transforms
from openpi.policies import umi_dual_franka_policy as umi_policy
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logger = logging.getLogger(__name__)


def _episode_ranges(dataset: LeRobotDataset) -> list[tuple[int, int]]:
    episode_index = dataset.episode_data_index
    return [
        (int(episode_index["from"][episode]), int(episode_index["to"][episode]))
        for episode in range(len(episode_index["from"]))
    ]


def _sample_full_chunk_indices(
    dataset: LeRobotDataset,
    *,
    action_horizon: int,
    num_samples: int,
    seed: int,
) -> list[int]:
    candidates: list[int] = []
    for start, end in _episode_ranges(dataset):
        candidates.extend(range(start, max(start, end - action_horizon)))
    if not candidates:
        raise ValueError("No recorded episode is long enough to contain one full action chunk")
    generator = np.random.default_rng(seed)
    count = min(num_samples, len(candidates))
    return generator.choice(candidates, size=count, replace=False).tolist()


def _rotation_error_degrees(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    # Float32 trace-based SO(3) angles suffer severe cancellation near zero and
    # can report ~0.05 degree errors for an otherwise exact round trip.
    actual_matrix = umi_policy.quaternion_xyzw_to_matrix(np.asarray(actual, dtype=np.float64))
    expected_matrix = umi_policy.quaternion_xyzw_to_matrix(np.asarray(expected, dtype=np.float64))
    relative = np.einsum("...ji,...jk->...ik", actual_matrix, expected_matrix)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _action_next_state_error(raw_states: np.ndarray, raw_actions: np.ndarray, *, horizon: int) -> float:
    expected_state_shape = (horizon + 1, umi_policy.RAW_STATE_DIM)
    expected_action_shape = (horizon, umi_policy.RAW_STATE_DIM)
    if raw_states.shape != expected_state_shape or raw_actions.shape != expected_action_shape:
        raise ValueError(
            "Raw state/action chunks must have shapes "
            f"{expected_state_shape} and {expected_action_shape}, got {raw_states.shape} and {raw_actions.shape}"
        )
    return float(np.max(np.abs(raw_actions - raw_states[1:])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-name",
        default="pi05_umi_dual_franka_cardboard_box_relative",
    )
    parser.add_argument(
        "--repo-id",
        default="local/cardboard_box_tcp_curated_logical_train",
    )
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    data_config = dataclasses.replace(data_config, repo_id=args.repo_id)
    metadata = LeRobotDatasetMetadata(args.repo_id)
    horizon = train_config.model.action_horizon
    delta_timestamps = {
        key: [step / metadata.fps for step in range(horizon)] for key in data_config.action_sequence_keys
    }
    raw_single = LeRobotDataset(args.repo_id)
    raw_chunked = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
    transformed = _data_loader.TransformedDataset(
        raw_chunked,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs],
    )
    decode_actions = _transforms.compose(data_config.data_transforms.outputs)
    indices = _sample_full_chunk_indices(
        raw_single,
        action_horizon=horizon,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    transformed_states = []
    transformed_actions = []
    max_next_state_error = 0.0
    max_position_roundtrip_error = 0.0
    max_rotation_roundtrip_error = 0.0
    max_gripper_roundtrip_error = 0.0

    for index in indices:
        raw_rows = raw_single.hf_dataset.select(range(index, index + horizon + 1))
        raw_states = np.stack([np.asarray(value) for value in raw_rows["observation.state"]])
        raw_actions = np.stack([np.asarray(value) for value in raw_rows["action"][:horizon]])
        max_next_state_error = max(
            max_next_state_error,
            _action_next_state_error(raw_states, raw_actions, horizon=horizon),
        )
        item = transformed[index]
        model_state = np.asarray(item["state"])
        model_actions = np.asarray(item["actions"])
        if model_state.shape != (umi_policy.MODEL_STATE_DIM,):
            raise ValueError(f"Transformed state must be 20-D, got {model_state.shape}")
        if model_actions.shape != (horizon, umi_policy.MODEL_STATE_DIM):
            raise ValueError(f"Transformed actions must have shape {(horizon, 20)}, got {model_actions.shape}")
        if item["image_mask"] != {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }:
            raise ValueError(f"Unexpected image mask: {item['image_mask']}")

        decoded = decode_actions({"state": model_state, "actions": model_actions})["actions"]
        expected = raw_actions
        max_position_roundtrip_error = max(
            max_position_roundtrip_error,
            float(np.max(np.abs(decoded[..., [0, 1, 2, 8, 9, 10]] - expected[..., [0, 1, 2, 8, 9, 10]]))),
        )
        max_rotation_roundtrip_error = max(
            max_rotation_roundtrip_error,
            float(
                np.max(
                    np.concatenate(
                        (
                            _rotation_error_degrees(decoded[..., 3:7], expected[..., 3:7]),
                            _rotation_error_degrees(decoded[..., 11:15], expected[..., 11:15]),
                        )
                    )
                )
            ),
        )
        max_gripper_roundtrip_error = max(
            max_gripper_roundtrip_error,
            float(np.max(np.abs(decoded[..., [7, 15]] - expected[..., [7, 15]]))),
        )
        transformed_states.append(model_state)
        transformed_actions.append(model_actions)

    states = np.stack(transformed_states)
    actions = np.stack(transformed_actions)
    print(f"repo_id: {args.repo_id}")
    print(f"config: {args.config_name}")
    print(f"fps: {metadata.fps}")
    print(f"recorded episodes: {metadata.total_episodes}")
    print(f"chunk duration: {horizon / metadata.fps:.3f} s ({horizon} targets)")
    print(f"sampled full chunks: {len(indices)}")
    print(f"state shape/range: {states.shape}, [{states.min():.6g}, {states.max():.6g}]")
    print(f"action shape/range: {actions.shape}, [{actions.min():.6g}, {actions.max():.6g}]")
    print(f"max action[t]-state[t+1] error: {max_next_state_error:.6g}")
    print(f"max action round-trip position error: {max_position_roundtrip_error:.6g}")
    print(f"max action round-trip rotation error (deg): {max_rotation_roundtrip_error:.6g}")
    print(f"max action round-trip gripper error: {max_gripper_roundtrip_error:.6g}")

    if max_next_state_error > 1e-5:
        raise SystemExit("FAILED: raw action[t] does not match state[t+1]")
    if max_position_roundtrip_error > 1e-5 or max_rotation_roundtrip_error > 1e-3 or max_gripper_roundtrip_error > 1e-5:
        raise SystemExit("FAILED: model action transform does not round-trip accurately")
    if args.config_name.endswith("_long_episode"):
        logger.warning(
            "Long-episode mode intentionally does not reject chunks that cross an internal physical-box boundary"
        )
    logger.info("Dual-Franka UMI dataset preflight passed")


if __name__ == "__main__":
    main()
