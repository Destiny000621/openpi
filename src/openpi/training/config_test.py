import numpy as np
import pytest

import openpi.policies.umi_dual_franka_policy as umi_dual_franka_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as transforms

_LOGICAL_REPO_ID = "local/cardboard_box_tcp_curated_logical_train"
_LONG_REPO_ID = "byang11259/cardboard_box_tcp_curated"
_PROMPT = "Assemble the cardboard box and put it into the bin"


@pytest.mark.parametrize(
    ("config_name", "repo_id", "representation", "input_type", "output_type"),
    [
        (
            "pi05_umi_dual_franka_cardboard_box_relative",
            _LOGICAL_REPO_ID,
            "relative",
            umi_dual_franka_policy.UmiDualFrankaRelativeInputs,
            umi_dual_franka_policy.UmiDualFrankaRelativeOutputs,
        ),
        (
            "pi05_umi_dual_franka_cardboard_box_absolute",
            _LOGICAL_REPO_ID,
            "absolute",
            umi_dual_franka_policy.UmiDualFrankaAbsoluteInputs,
            umi_dual_franka_policy.UmiDualFrankaAbsoluteOutputs,
        ),
        (
            "pi05_umi_dual_franka_cardboard_box_relative_long_episode",
            _LONG_REPO_ID,
            "relative",
            umi_dual_franka_policy.UmiDualFrankaRelativeInputs,
            umi_dual_franka_policy.UmiDualFrankaRelativeOutputs,
        ),
        (
            "pi05_umi_dual_franka_cardboard_box_absolute_long_episode",
            _LONG_REPO_ID,
            "absolute",
            umi_dual_franka_policy.UmiDualFrankaAbsoluteInputs,
            umi_dual_franka_policy.UmiDualFrankaAbsoluteOutputs,
        ),
    ],
)
def test_umi_dual_franka_config_contract(
    monkeypatch,
    tmp_path,
    config_name,
    repo_id,
    representation,
    input_type,
    output_type,
):
    # Keep this a local config test: tokenizer and norm-stat assets are covered by
    # their own tests and would otherwise require network or precomputed assets.
    monkeypatch.setattr(
        _config.DataConfigFactory,
        "_load_norm_stats",
        lambda self, assets_dir, asset_id: None,
    )
    monkeypatch.setattr(
        _config.ModelTransformFactory,
        "__call__",
        lambda self, model_config: transforms.Group(),
    )

    train_config = _config.get_config(config_name)
    assert train_config.model.pi05
    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 50
    assert isinstance(train_config.data, _config.LeRobotUmiDualFrankaDataConfig)
    assert train_config.data.repo_id == repo_id
    assert train_config.data.default_prompt == _PROMPT
    assert train_config.data.action_representation == representation
    assert train_config.data.assets.assets_dir is None
    assert train_config.data.assets.asset_id is None
    assert train_config.batch_size == 32
    assert train_config.fsdp_devices == 1
    assert train_config.num_workers == 8
    assert train_config.num_train_steps == 5_000
    assert train_config.assets_base_dir == "./assets"
    assert train_config.checkpoint_base_dir == "./checkpoints"

    data_config = train_config.data.create(tmp_path, train_config.model)
    assert data_config.repo_id == repo_id
    assert data_config.asset_id == repo_id
    assert data_config.norm_stats is None
    assert data_config.action_sequence_keys == ("action",)
    assert isinstance(data_config.data_transforms.inputs[0], input_type)
    assert isinstance(data_config.data_transforms.outputs[0], output_type)

    source = {
        "observation.state": np.zeros(16, dtype=np.float32),
        "observation.images.left_head": np.zeros((3, 8, 8), dtype=np.uint8),
        "observation.images.right_head": np.zeros((3, 8, 8), dtype=np.uint8),
        "action": np.zeros((50, 16), dtype=np.float32),
        "task": _PROMPT,
    }
    repacked = data_config.repack_transforms.inputs[0](source)
    assert repacked["observation/state"].shape == (16,)
    assert repacked["observation/left_head"].shape == (3, 8, 8)
    assert repacked["observation/right_head"].shape == (3, 8, 8)
    assert repacked["actions"].shape == (50, 16)
    assert repacked["prompt"] == _PROMPT


def test_umi_dual_franka_config_rejects_unknown_action_representation(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _config.DataConfigFactory,
        "_load_norm_stats",
        lambda self, assets_dir, asset_id: None,
    )

    data_factory = _config.LeRobotUmiDualFrankaDataConfig(
        repo_id=_LOGICAL_REPO_ID,
        action_representation="not-a-representation",  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="action_representation"):
        data_factory.create(tmp_path, _config.pi0_config.Pi0Config(pi05=True))


def test_long_episode_config_delegates_50_step_chunking_to_lerobot(monkeypatch, tmp_path):
    monkeypatch.setattr(
        _config.DataConfigFactory,
        "_load_norm_stats",
        lambda self, assets_dir, asset_id: None,
    )
    monkeypatch.setattr(
        _config.ModelTransformFactory,
        "__call__",
        lambda self, model_config: transforms.Group(),
    )
    train_config = _config.get_config("pi05_umi_dual_franka_cardboard_box_relative_long_episode")
    data_config = train_config.data.create(tmp_path, train_config.model)
    captured = {}
    sentinel = object()

    class Metadata:
        fps = 30.0

    def fake_dataset(repo_id, *, delta_timestamps):
        captured["repo_id"] = repo_id
        captured["delta_timestamps"] = delta_timestamps
        return sentinel

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda repo_id: Metadata())
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", fake_dataset)

    dataset = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=train_config.model.action_horizon,
        model_config=train_config.model,
    )

    assert dataset is sentinel
    assert captured["repo_id"] == _LONG_REPO_ID
    assert list(captured["delta_timestamps"]) == ["action"]
    np.testing.assert_allclose(captured["delta_timestamps"]["action"], np.arange(50) / Metadata.fps)
