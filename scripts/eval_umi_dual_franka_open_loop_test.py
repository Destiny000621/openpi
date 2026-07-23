from pathlib import Path

import numpy as np
import pytest

from openpi.shared.normalize import NormStats
from scripts import eval_umi_dual_franka_open_loop as evaluator


def _stats(offset: float = 0.0) -> dict[str, NormStats]:
    return {
        "state": NormStats(
            mean=np.array([offset, 1.0]),
            std=np.array([2.0, 3.0]),
            q01=np.array([-1.0, -2.0]),
            q99=np.array([4.0, 5.0]),
        ),
        "actions": NormStats(
            mean=np.array([6.0, 7.0]),
            std=np.array([8.0, 9.0]),
            q01=None,
            q99=None,
        ),
    }


def test_assert_matching_norm_stats_accepts_identical_values():
    evaluator.assert_matching_norm_stats(
        _stats(),
        _stats(),
        reference_checkpoint=Path("checkpoint-a"),
        candidate_checkpoint=Path("checkpoint-b"),
    )


def test_assert_matching_norm_stats_rejects_different_values():
    with pytest.raises(ValueError, match="different normalization stats"):
        evaluator.assert_matching_norm_stats(
            _stats(),
            _stats(offset=0.25),
            reference_checkpoint=Path("checkpoint-a"),
            candidate_checkpoint=Path("checkpoint-b"),
        )


def test_build_eval_frames_includes_only_valid_final_chunk_start(monkeypatch):
    class Metadata:
        fps = 30.0

    class Dataset:
        def __init__(self):
            self.episode_data_index = {"from": np.array([10]), "to": np.array([60])}

    captured = {}

    def fake_dataset(repo_id, *, delta_timestamps):
        captured["repo_id"] = repo_id
        captured["delta_timestamps"] = delta_timestamps
        return Dataset()

    monkeypatch.setattr(evaluator, "LeRobotDatasetMetadata", lambda repo_id: Metadata())
    monkeypatch.setattr(evaluator, "LeRobotDataset", fake_dataset)
    data_config = type("DataConfig", (), {"action_sequence_keys": ("action",)})()

    dataset, frame_indices = evaluator.build_eval_frames(
        data_config,
        action_horizon=50,
        repo_id="local/eval",
        episodes=[0],
        stride=10,
        max_per_episode=8,
    )

    assert isinstance(dataset, Dataset)
    assert frame_indices == [10]
    assert captured["repo_id"] == "local/eval"
    np.testing.assert_allclose(captured["delta_timestamps"]["action"], np.arange(50) / 30.0)
