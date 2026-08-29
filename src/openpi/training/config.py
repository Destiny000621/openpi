"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.franka_policy as franka_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.umi_dual_franka_policy as umi_dual_franka_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def _identity_rot6d_norm_stats(
    norm_stats: dict[str, _transforms.NormStats],
) -> dict[str, _transforms.NormStats]:
    """Replace the rot6d dims (3:9) of rot6d-layout state/actions stats with identity values.

    q01=-1, q99=+1 makes quantile normalization (x - q01) / (q99 - q01) * 2 - 1 the
    identity on those dims (up to the 1e-6 eps); mean=0, std=1 does the same for
    z-score. Other dims keep their computed stats. Applies to 10-D actions and
    10-D or 17-D (joints-appended) states; anything else passes through untouched.
    """
    widths = {"actions": (10,), "state": (10, 17)}
    out: dict[str, _transforms.NormStats] = {}
    for key, stats in norm_stats.items():
        if key not in widths or stats.mean.shape[-1] not in widths[key]:
            out[key] = stats
            continue
        mean, std = stats.mean.copy(), stats.std.copy()
        mean[..., 3:9] = 0.0
        std[..., 3:9] = 1.0
        q01 = None if stats.q01 is None else stats.q01.copy()
        q99 = None if stats.q99 is None else stats.q99.copy()
        if q01 is not None:
            q01[..., 3:9] = -1.0
        if q99 is not None:
            q99[..., 3:9] = 1.0
        out[key] = _normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)
    return out


@dataclasses.dataclass(frozen=True)
class LeRobotFrankaDataConfig(DataConfigFactory):
    """Data config for a single-arm Franka FR3 (29-D state, 8-D EE actions, 2 cameras).

    EE-space actions because the raw teleop never stamped arm joint_targets
    (dead columns) — the arm command lives in target_pose. Quaternions are
    canonicalized by the converter (per-stream continuity fix + dataset-level
    reference hemisphere; see scripts/convert_franka_raw_to_lerobot.py). The
    state front (dims 0-7) mirrors the action layout so DeltaActions lines up;
    the rest is extra proprio. The dataset has:
        observation.state          float32 (29,)  [qw,qx,qy,qz, x,y,z, gripper,
                                                   j0..j6, j0_vel..j6_vel,
                                                   gripper_vel, fx..tz], absolute
        action                     float32 (8,)   [qw,qx,qy,qz, x,y,z, gripper], absolute
        observation.images.camera0 video          wrist camera
        observation.images.camera1 video          side / third-person camera
    """

    # If true, convert the xyz translation to deltas w.r.t. the current state; the
    # quaternion (dims 0-3) and gripper (dim 7) stay absolute — elementwise
    # quaternion deltas aren't valid rotations, and absolute orientation regresses
    # cleanly. Poses are stored absolute in the dataset, so keep this True.
    use_delta_joint_actions: bool = True
    # Injected into the input data when the "prompt" key is not present.
    default_prompt: str | None = None
    # State width fed to the model. rot6d10: 10 (standard) or 17 when the dataset
    # was converted with --include-joints ([xyz, rot6d, gripper, j0..j6] — the
    # joints ride BEHIND the pose prefix so DeltaActions still lines up).
    # Legacy quat8 datasets: 29 (full proprio) or 8 (prefix slice) — a
    # git-restored quat8 config must set this explicitly (29 for the old
    # lan_insertion config, which relied on the former default).
    state_dim: int = 10
    # If true, train on the wrist camera only: the side camera (base_0_rgb) is
    # zeroed + masked off; the wrist stays in left_wrist_0_rgb. Same dataset,
    # no reconversion — controlled ablation of the third-person view.
    wrist_camera_only: bool = False
    # Action/state representation the dataset was converted with:
    #   "rot6d10": action 10-D [xyz, rot6d, gripper], delta mask (9, -1)
    #              (xyz + rot6d relative-to-state, gripper absolute; RLinf-style).
    #              Pair with state_dim=10 and a *_r6_v21 dataset. THE standard.
    #   "quat8":   LEGACY (all quat8 TrainConfigs were removed): action 8-D
    #              [quat, xyz, gripper], delta mask (-4, 3, -1). Kept only so a
    #              git-restored config can still serve an old quat8 checkpoint;
    #              do not use for new Franka training — quaternion hemisphere
    #              canonicalization proved fragile at qw~0.
    action_representation: Literal["quat8", "rot6d10"] = "rot6d10"
    # rot6d10 only. If False (the standard going forward), the rot6d dims (3:9 of
    # both state and actions) BYPASS normalization: their stats are replaced at
    # load time with identity-mapping values (q01=-1, q99=+1; mean=0, std=1), so
    # the quantile map (x-q01)/(q99-q01)*2-1 reduces to x -> x. rot6d components
    # are rotation-matrix entries already in [-1, 1] (and their relative deltas
    # are small), so the network regresses raw geometry that feeds
    # rotation_6d_to_matrix() directly. xyz and gripper keep openpi's normal
    # quantile normalization. Because train-time Normalize, the checkpoint's
    # baked assets (written from these in-memory stats), and serve-time
    # Unnormalize all consume data_config.norm_stats, the identity override is
    # consistent across the whole lifecycle. True stats stay on disk untouched.
    normalize_rot6d: bool = True
    # Optional fractional (x0, y0, x1, y1) crop of the WRIST image before the
    # 224x224 model resize (FrankaInputs.wrist_crop; the UMI image_crop pattern).
    # Raises effective px-per-mm at the fingertips/port without touching the
    # dataset or the vision tower; the side camera keeps its full FOV. Serving
    # contract: the client must send the RAW wrist frame (send_full_wrist: true
    # in the avantbot session YAML) — FrankaInputs rejects a pre-resized 224x224
    # wrist image so a mismatched client fails loudly instead of silently
    # cropping letterbox bars.
    wrist_crop: tuple[float, float, float, float] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Map the dataset's LeRobot keys onto the keys FrankaInputs expects.
        # camera1 (side) -> base_0_rgb; camera0 (wrist) -> left_wrist_0_rgb.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.camera1",
                        "observation/wrist_image": "observation.images.camera0",
                        "observation/state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )

        rot6d = self.action_representation == "rot6d10"
        data_transforms = _transforms.Group(
            inputs=[
                franka_policy.FrankaInputs(
                    model_type=model_config.model_type,
                    state_dim=self.state_dim,
                    wrist_camera_only=self.wrist_camera_only,
                    wrist_crop=self.wrist_crop,
                )
            ],
            outputs=[franka_policy.FrankaOutputs(action_dim=10 if rot6d else 8)],
        )

        if self.use_delta_joint_actions:
            if rot6d:
                # xyz + rot6d (0-8) delta vs. state, gripper (9) absolute.
                delta_action_mask = _transforms.make_bool_mask(9, -1)
            else:
                # quaternion (0-3) absolute, xyz (4-6) delta vs. state, gripper (7)
                # absolute. The 8-dim mask only touches actions[:8]/state[:8]; the
                # extra state proprio (dims 8-28) is never delta'd.
                delta_action_mask = _transforms.make_bool_mask(-4, 3, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        base_config = self.create_base_config(assets_dirs, model_config)
        if not self.normalize_rot6d:
            if not rot6d:
                raise ValueError("normalize_rot6d=False is only meaningful with action_representation='rot6d10'.")
            if base_config.norm_stats is not None:
                base_config = dataclasses.replace(
                    base_config, norm_stats=_identity_rot6d_norm_stats(base_config.norm_stats)
                )

        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # Our LeRobot dataset's action column is "action" (singular); the loader
            # reads the action-horizon sequence from this key before the repack runs.
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUmiDualFrankaDataConfig(DataConfigFactory):
    """LeRobot config for dual-arm UMI demonstrations retargeted to EEF space.

    The source dataset stores absolute 16-D state/action vectors:
        [left xyz, left quat_xyzw, left gripper,
         right xyz, right quat_xyzw, right gripper].

    The policy transforms convert those vectors to the compact 20-D rot6d
    representation used by the model. The relative variant additionally converts
    every action waypoint to a true SE(3) transform relative to the chunk-start
    state; the absolute baseline keeps action poses absolute.

    ``state_mode="gripper_only"`` (relative actions only) reduces the policy
    state to the 2-D absolute ``[left_gripper, right_gripper]`` vector — no
    pose dimensions — and serves the action chunk in the relative frame for
    client-side anchor composition. This keeps the policy free of
    scene/marker-absolute pose for cross-embodiment use.
    """

    action_representation: Literal["relative", "absolute"] = "relative"
    state_mode: Literal["full", "gripper_only"] = "full"
    # Optional centered square crop (pixel side length) applied to both camera
    # views in the shared input transform, before the model resize.
    image_crop: int | None = None
    default_prompt: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/state": "observation.state",
                        "observation/left_head": "observation.images.left_head",
                        "observation/right_head": "observation.images.right_head",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )

        if self.state_mode not in ("full", "gripper_only"):
            raise ValueError(f"state_mode must be either 'full' or 'gripper_only', got {self.state_mode!r}")

        match self.action_representation:
            case "relative":
                data_transforms = _transforms.Group(
                    inputs=[
                        umi_dual_franka_policy.UmiDualFrankaRelativeInputs(
                            model_type=model_config.model_type,
                            state_mode=self.state_mode,
                            image_crop=self.image_crop,
                        )
                    ],
                    outputs=[
                        umi_dual_franka_policy.UmiDualFrankaRelativeGripperOnlyOutputs()
                        if self.state_mode == "gripper_only"
                        else umi_dual_franka_policy.UmiDualFrankaRelativeOutputs()
                    ],
                )
            case "absolute":
                if self.state_mode != "full":
                    raise ValueError(
                        "state_mode='gripper_only' requires action_representation='relative'; "
                        "the absolute baseline decodes world-frame targets and needs absolute state."
                    )
                data_transforms = _transforms.Group(
                    inputs=[
                        umi_dual_franka_policy.UmiDualFrankaAbsoluteInputs(
                            model_type=model_config.model_type, image_crop=self.image_crop
                        )
                    ],
                    outputs=[umi_dual_franka_policy.UmiDualFrankaAbsoluteOutputs()],
                )
            case _:
                raise ValueError(
                    f"action_representation must be either 'relative' or 'absolute', got {self.action_representation!r}"
                )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
    #
    # YAM bimanual fine-tuning configs (custom, added by LOBE).
    #
    # Maps the YAM bimanual robot's 3 cameras (head_camera, left_wrist_camera, right_wrist_camera)
    # onto the AlohaInputs camera schema (cam_high, cam_left_wrist, cam_right_wrist). Uses
    # adapt_to_pi=False because YAM's joint convention is NOT Trossen Aloha — passing the
    # raw 14-dim joint+gripper state through is correct.
    TrainConfig(
        name="pi05_yam_place_vial",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/place_the_vial_v21",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            adapt_to_pi=False,
            default_prompt="place the vial into the stand",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head_camera",
                                "cam_left_wrist": "observation.images.left_wrist_camera",
                                "cam_right_wrist": "observation.images.right_wrist_camera",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,  # 8 per device × 8 GPUs
        checkpoint_base_dir="/mnt/localssd/sunlingfeng/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/sunlingfeng/openpi-assets",
    ),
    # v1: same as v0 but trained on the resampled, honest-fps dataset
    # (ttotmoon/8ml_vial_place_30fps, real-30Hz from 50-58Hz source recordings).
    TrainConfig(
        name="pi05_yam_vial_30fps",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/8ml_vial_place_v21",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            adapt_to_pi=False,
            default_prompt="place the vial into the stand",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head_camera",
                                "cam_left_wrist": "observation.images.left_wrist_camera",
                                "cam_right_wrist": "observation.images.right_wrist_camera",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
        batch_size=56,
        # Default num_workers=2 starves 7-GPU training (each GPU shares ~0.3 of a
        # video-decoding worker). 8 = one per GPU + spare. Watch RAM during run —
        # each worker has the policy preprocessor in mem (~few hundred MB).
        num_workers=8,
        checkpoint_base_dir="/mnt/localssd/sunlingfeng/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/sunlingfeng/openpi-assets",
    ),
    # train90 variant: first 177 episodes only. Held-out 20 eps (177-196) are
    # used by lobe's scripts/diagnose_policy_drift.py for true OOD comparison
    # vs FM v2 (also trained on the same train90 split).
    TrainConfig(
        name="pi05_yam_vial_30fps_train90",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/8ml_vial_place_v21_train90",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            adapt_to_pi=False,
            default_prompt="place the vial into the stand",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head_camera",
                                "cam_left_wrist": "observation.images.left_wrist_camera",
                                "cam_right_wrist": "observation.images.right_wrist_camera",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
        batch_size=64,
        # fsdp_devices=1 (default) replicates the full pi05 model on every GPU →
        # OOMs even at small batch sizes (pi05 is ~3B params; replicated state +
        # activations exceed 80 GB on H100). fsdp_devices=8 shards across all
        # 8 GPUs (~1.5 GB params/GPU). Required when training on 8×H100.
        fsdp_devices=8,
        num_workers=8,
        checkpoint_base_dir="/mnt/localssd/sunlingfeng/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/sunlingfeng/openpi-assets",
    ),
    # Sichang SFT: 4-vial pick-and-place, 180 episodes @ 30fps
    # (HF: Sichang0621/vials_4_30fps_180, converted to v2.1 → local/vials_4_30fps_180_v21).
    TrainConfig(
        name="pi05_yam_vial_4_30fps",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/vials_4_30fps_180_v21",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            adapt_to_pi=False,
            default_prompt="pick up all vials and place them in the stand",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head_camera",
                                "cam_left_wrist": "observation.images.left_wrist_camera",
                                "cam_right_wrist": "observation.images.right_wrist_camera",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=12000,
        batch_size=64,
        # fsdp_devices=8 shards the ~3B pi05 model across all 8 H200s; without it the
        # full model replicates per-GPU and OOMs. Required for 8-GPU training.
        fsdp_devices=8,
        num_workers=8,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # Sichang 2nd SFT: vials_4 (all 180 eps) augmented with 46 selected episodes from
    # ttotmoon/8ml_vial_place_30fps (eps 0-40, 165, 167, 168, 172, 177) → 226 eps / 236,684 frames
    # (local/vials_4_aug_8ml46_v21; this is what exp v1 trained on).
    # v2 additionally folds in all 83 correction episodes from local/vial_correction_v21
    # → 309 eps / 304,618 frames in local/vials_4_aug_8ml46_corr_v21.
    # Both built with scripts/merge_lerobot_v21.py; the merge drops `phase`/`correction_index`,
    # which only the correction dataset carries, since openpi does not consume them.
    # Single prompt for all data (prompt_from_task=False, so default_prompt covers every frame;
    # the 8ml single-vial demos are extra data for the 4-vial task).
    TrainConfig(
        name="pi05_yam_vial_4_30fps_aug",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/vials_4_aug_8ml46_corr_v21",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            adapt_to_pi=False,
            default_prompt="pick up all vials and place them in the stand",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head_camera",
                                "cam_left_wrist": "observation.images.left_wrist_camera",
                                "cam_right_wrist": "observation.images.right_wrist_camera",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # exp recovery_v2 first ran 0->9999 at num_train_steps=10_000, then was extended to
        # 15_000 and resumed from the 9999 checkpoint (--resume). Safe to extend because the
        # default CosineDecaySchedule has a fixed decay_steps=30_000 that does NOT depend on
        # num_train_steps, so the LR continues down the same curve (~2.0e-5 -> ~1.4e-5) with
        # no discontinuity at the resume point.
        num_train_steps=15_000,
        batch_size=80,
        fsdp_devices=4,
        num_workers=8,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),

    # add vial correction data
    TrainConfig(
        name="pi05_yam_vial_30fps_aug_correction",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/vial_correction_v21",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            adapt_to_pi=False,
            default_prompt="pick up all vials and place them in the stand",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head_camera",
                                "cam_left_wrist": "observation.images.left_wrist_camera",
                                "cam_right_wrist": "observation.images.right_wrist_camera",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        # Warm-start from the vial-aug checkpoint (step 10000) instead of pi05_base.
        # CheckpointWeightLoader takes the `params/` dir of a trained checkpoint;
        # it loads model weights only (optimizer state + step counter reset), so
        # this is a fresh 3k-step fine-tune on top of those weights.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/mnt/localssd/Sichang/openpi-checkpoints/pi05_yam_vial_4_30fps_aug/v1/10000/params"
        ),
        num_train_steps=3_000,
        batch_size=64,
        fsdp_devices=4,
        num_workers=4,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    #
    # ABC-130k (XDOF) single-task YAM configs. MCAP episodes converted to v2.1 by
    # scripts/convert_abc_mcap_to_lerobot_v21.py (yam_finetune.md §"Fine-tuning
    # from ABC-130k"). Same YAM platform as the vial configs — 14-D joints+grippers,
    # adapt_to_pi=False, trossen assets — 500 train episodes per task, 30 Hz.
    # Checkpoints at 5k/10k/14999 (save_interval=5k); val-split selection via
    # scripts/eval_open_loop.py picked 14999 for both tasks.
    #
    TrainConfig(
        name="pi05_yam_abc_earbuds",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/abc_earbuds_v21",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            adapt_to_pi=False,
            default_prompt="insert the wireless bluetooth earbuds into the charging case",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head_camera",
                                "cam_left_wrist": "observation.images.left_wrist_camera",
                                "cam_right_wrist": "observation.images.right_wrist_camera",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        # Extended 15k -> 18k steps: TRUE resume from the 14999 checkpoint.
        # fsdp_devices MUST stay 8: the checkpoint was saved on an 8-device mesh and
        # stock openpi's resume path relies on orbax's saved-sharding fallback, which
        # requires the same device topology (resuming with 4 visible GPUs fails with
        # "sharding passed to deserialization ... Got None"). Defaults give
        # save_interval=1000, keep_period=5000 — note 14999 is NOT a keep_period
        # multiple, so it is GC'd after the first new save (backup kept at
        # v1_14999_backup/).
        num_train_steps=18_000,
        batch_size=64,
        fsdp_devices=8,
        num_workers=8,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    TrainConfig(
        name="pi05_yam_abc_fold_box",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/abc_fold_box_v21",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            adapt_to_pi=False,
            default_prompt="fold the paper box",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head_camera",
                                "cam_left_wrist": "observation.images.left_wrist_camera",
                                "cam_right_wrist": "observation.images.right_wrist_camera",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=23_000,
        batch_size=64,
        fsdp_devices=8,
        num_workers=8,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # ABC-130k sort_the_legos_into_containers_by_color: first 500 of 4458 MCAP
    # episodes (sorted by uuid) converted the same way as pi05_yam_abc_earbuds.
    TrainConfig(
        name="pi05_yam_abc_sort_legos",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="local/abc_sort_legos_v21",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            adapt_to_pi=False,
            default_prompt="sort the legos into containers by color",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.head_camera",
                                "cam_left_wrist": "observation.images.left_wrist_camera",
                                "cam_right_wrist": "observation.images.right_wrist_camera",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=18_000,
        batch_size=128,
        fsdp_devices=4,
        num_workers=32,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    #
    # Dual-arm Franka UMI cardboard-box configs.
    #
    # Source: byang11259/cardboard_box_tcp_curated. The default pair uses a
    # derived one-logical-box-per-episode dataset. The explicit long-episode pair
    # below points at the original multi-box recordings and intentionally lets
    # the stock LeRobot loader form 50-step chunks across internal box
    # boundaries. All variants consume absolute 16-D dual-EEF state/actions and
    # convert them to compact 20-D rot6d vectors. Relative and absolute action
    # variants compute independent quantile norm stats.
    TrainConfig(
        name="pi05_umi_dual_franka_cardboard_box_relative",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotUmiDualFrankaDataConfig(
            repo_id="local/cardboard_box_tcp_curated_logical_train",
            default_prompt="Assemble the cardboard box and put it into the bin",
            action_representation="relative",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
        batch_size=32,
        fsdp_devices=1,
        num_workers=8,
        save_interval=5_000,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    TrainConfig(
        name="pi05_umi_dual_franka_cardboard_box_absolute",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotUmiDualFrankaDataConfig(
            repo_id="local/cardboard_box_tcp_curated_logical_train",
            default_prompt="Assemble the cardboard box and put it into the bin",
            action_representation="absolute",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
        batch_size=32,
        fsdp_devices=1,
        num_workers=8,
        save_interval=5_000,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    TrainConfig(
        name="pi05_umi_dual_franka_cardboard_box_relative_long_episode",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotUmiDualFrankaDataConfig(
            repo_id="local/cardboard_box_tcp_curated_x264",
            default_prompt="Assemble the cardboard box and put it into the bin",
            action_representation="relative",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
        batch_size=32,
        fsdp_devices=1,
        num_workers=8,
        save_interval=5_000,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    TrainConfig(
        name="pi05_umi_dual_franka_cardboard_box_absolute_long_episode",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotUmiDualFrankaDataConfig(
            repo_id="local/cardboard_box_tcp_curated_x264",
            default_prompt="Assemble the cardboard box and put it into the bin",
            action_representation="absolute",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
        batch_size=32,
        fsdp_devices=1,
        num_workers=8,
        save_interval=5_000,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # Full-source-set variant: identical recipe to the 10s gripper-only config
    # (2-D gripper state, relative chunks, 224 px crop) on the complete
    # vid7to54 export (46 session-length episodes, 205k frames, x264 re-encode;
    # episodes 37-39 relabeled from a placeholder to the standard task string).
    TrainConfig(
        name="pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotUmiDualFrankaDataConfig(
            repo_id="local/cardboard_box_tcp_vid7to54_x264",
            default_prompt="Assemble the cardboard box and put it into the bin",
            action_representation="relative",
            state_mode="gripper_only",
            image_crop=224,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10000,
        batch_size=128,
        fsdp_devices=8,
        num_workers=32,
        save_interval=5_000,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # Cross-embodiment variant with cropped views: same as the gripper-only
    # config below, plus a centered 272 px crop of both 384 px fisheye views
    # (largest square inscribed in the fisheye circle) applied in the shared
    # input transform. Removes the dead black corners and most scene/operator
    # periphery and magnifies the workspace ~1.4x after the model resize.
    TrainConfig(
        name="pi05_umi_dual_franka_cardboard_box_relative_gripper_only_crop272_long_episode",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotUmiDualFrankaDataConfig(
            repo_id="local/cardboard_box_tcp_curated_x264",
            default_prompt="Assemble the cardboard box and put it into the bin",
            action_representation="relative",
            state_mode="gripper_only",
            image_crop=272,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10000,
        batch_size=128,
        fsdp_devices=8,
        num_workers=8,
        save_interval=5_000,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # Cross-embodiment variant: relative actions with a 2-D gripper-only policy
    # state (no pose dimensions). The server returns relative chunks; the
    # robot client composes them with its own query-time TCP anchors.
    TrainConfig(
        name="pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
        data=LeRobotUmiDualFrankaDataConfig(
            repo_id="local/cardboard_box_tcp_curated_10s_x264",
            default_prompt="Assemble the cardboard box and put it into the bin",
            action_representation="relative",
            state_mode="gripper_only",
            image_crop=224,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10000,
        batch_size=128,
        fsdp_devices=8,
        num_workers=8,
        save_interval=5_000,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    #
    # Fine-tuning single-arm Franka FR3 configs (see docs/franka_finetune.md).
    #
    # Standard representation: rot6d10 — 10-D state AND action
    # [x,y,z, rot6d(6), gripper] from ee_pose/target_pose (EE-space because the
    # raw teleop never stamped arm joint_targets). DeltaActions(9,-1): xyz+rot6d
    # relative-to-state, gripper absolute. rot6d is quat-sign-invariant, so no
    # quaternion canonicalization exists anywhere in this path.
    # The legacy quat8 configs (pi05_franka_lan_insertion{,_s8,_s8_wrist},
    # pi05_franka_double_cable_left_s8{,_wrist}) were removed — quaternion
    # canonicalization proved fragile (qw~0 hemisphere splits). Their trained
    # checkpoints remain on disk; to serve one, restore its config from git
    # history (commit a7c3a4e or earlier).
    # Relative-EEF (rot6d) variant: 10-D action AND state [xyz, rot6d, gripper]
    # (RLinf-style), DeltaActions(9,-1) -> xyz+rot6d relative-to-state, gripper
    # absolute. rot6d is quat-sign-invariant, so this rep needs no quaternion
    # canonicalization. Curated 36-episode subset (scratchpad double_cable_36eps.txt)
    # converted with --rep rot6d10 --episode-list.
    TrainConfig(
        name="pi05_franka_double_cable_left_r6",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotFrankaDataConfig(
            repo_id="local/double_cable_insert_left_r6_v21",
            # Wrist-view scene: two black routers; both blue cables move from the
            # right router into the left router (hence ..._insert_left).
            default_prompt="Unplug the two cables from the right router, then insert them into the left router",
            use_delta_joint_actions=True,
            state_dim=10,
            action_representation="rot6d10",
            # This config's existing checkpoints (v1, loss 0.0068) trained with
            # rot6d normalized — keep True here for provenance/resume; the
            # raw-rot6d standard lives in the _rawrot config below.
            normalize_rot6d=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=4_000,
        save_interval=1_000,
        keep_period=1_000,
        batch_size=64,
        fsdp_devices=4,
        num_workers=32,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # Raw-rot6d variant (the standard going forward): identical to _r6 except the
    # rot6d dims (3:9 of state AND actions) bypass normalization — the model
    # regresses raw rotation-matrix columns (already in [-1,1]) that feed
    # rotation_6d_to_matrix() directly; xyz + gripper keep quantile norm.
    # Still 10-D actions including rot6d; only the normalization map changes.
    TrainConfig(
        name="pi05_franka_double_cable_left_r6_rawrot",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotFrankaDataConfig(
            repo_id="local/double_cable_insert_left_r6_v21",
            default_prompt="Unplug the two cables from the right router, then insert them into the left router",
            use_delta_joint_actions=True,
            state_dim=10,
            action_representation="rot6d10",
            normalize_rot6d=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=6_000,
        save_interval=2_000,
        keep_period=2_000,
        batch_size=128,
        fsdp_devices=8,
        num_workers=32,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # Same recipe as _r6_rawrot on the larger double_cable_100 recording set
    # (99 SUCCESS of 100 episodes, ~114k frames — vs 36 curated episodes/53k).
    # Same task and prompt; the routers sit further apart (grasp->release moves
    # ~21 cm laterally vs ~3 cm), so it is a distinct dataset, not a superset.
    TrainConfig(
        name="pi05_franka_double_cable_100_r6_rawrot",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotFrankaDataConfig(
            repo_id="local/double_cable_100_r6_v21",
            default_prompt="Unplug the two cables from the right router, then insert them into the left router",
            use_delta_joint_actions=True,
            state_dim=10,
            action_representation="rot6d10",
            normalize_rot6d=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=6_000,
        save_interval=2_000,
        keep_period=2_000,
        batch_size=128,
        fsdp_devices=8,
        num_workers=32,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # Wrist-crop variant of the dc100 rawrot recipe: identical data/model except
    # FrankaInputs crops the wrist image to the fractional box (0.339, 0.17,
    # 0.761, 0.92) before the 224x224 resize. The box is DATA-DRIVEN (2026-08-27
    # sweep of all 100 raw demos + 6 Aug-21 evals, orange-port detection):
    # target-port containment 100% insert / 99.8% grasp / 95.5% eval frames /
    # 89% approach (the rest are early-descent frames where the port is outside
    # even the full frame). Square-ish crop kills the 16:9 letterbox waste too:
    # px-per-mm at the port rises 2.37x (an RJ45 socket ~14x10 -> ~33x24 px),
    # more than a full-frame 448 input would give (2.0x) at zero extra compute —
    # motivation: Aug-21 evals failed on cm-level insertion aim (attempts
    # scattered 12-44 mm; measured z passed 30 mm below the demo contact floor).
    # Serving: client must send the RAW wrist frame — session YAML
    # franka_pi05_ee_fr3_wcrop.yaml sets send_full_wrist: true; a pre-resized
    # 224x224 wrist input is rejected loudly by FrankaInputs.
    # Run compute_norm_stats for this config name once (assets are per-name;
    # values are identical to the dc100 ones — images don't enter norm stats).
    TrainConfig(
        name="pi05_franka_double_cable_100_r6_rawrot_wcrop",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotFrankaDataConfig(
            repo_id="local/double_cable_100_r6_v21",
            default_prompt="Unplug the two cables from the right router, then insert them into the left router",
            use_delta_joint_actions=True,
            state_dim=10,
            action_representation="rot6d10",
            normalize_rot6d=False,
            wrist_crop=(0.339, 0.17, 0.761, 0.92),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=6_000,
        save_interval=2_000,
        keep_period=2_000,
        batch_size=128,
        fsdp_devices=8,
        num_workers=32,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # Same recipe as the 100-episode config, on the 99-episode re-converted
    # set (local/double_cable_99_r6_v21, 115,666 frames — a different episode
    # selection, not dc100 minus one). Separate config name so the dc100
    # checkpoints keep their provenance and norm-stats stay per-dataset.
    TrainConfig(
        name="pi05_franka_double_cable_99_r6_rawrot",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotFrankaDataConfig(
            repo_id="local/double_cable_99_r6_v21",
            default_prompt="Unplug the two cables from the right router, then insert them into the left router",
            use_delta_joint_actions=True,
            state_dim=10,
            action_representation="rot6d10",
            normalize_rot6d=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=10_000,
        save_interval=2_000,
        keep_period=2_000,
        batch_size=64,
        fsdp_devices=4,
        num_workers=16,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
    # Joints-in-state variant: identical to _r6_rawrot except the state carries
    # the 7 arm joint positions appended behind the pose prefix — 17-D
    # [xyz, rot6d, gripper, j0..j6] (dataset converted with --include-joints).
    # Actions unchanged (10-D); joints are proprio conditioning only.
    TrainConfig(
        name="pi05_franka_double_cable_left_r6_rawrot_joint",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotFrankaDataConfig(
            repo_id="local/double_cable_insert_left_r6j_v21",
            default_prompt="Unplug the two cables from the right router, then insert them into the left router",
            use_delta_joint_actions=True,
            state_dim=17,
            action_representation="rot6d10",
            normalize_rot6d=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=4_000,
        save_interval=1_000,
        keep_period=1_000,
        batch_size=64,
        fsdp_devices=4,
        num_workers=32,
        checkpoint_base_dir="/mnt/localssd/Sichang/openpi-checkpoints",
        assets_base_dir="/mnt/localssd/Sichang/openpi-assets",
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
