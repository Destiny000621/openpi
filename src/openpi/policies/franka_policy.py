import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# [quat, xyz, gripper | joint_pos arm | joint_vel | wrench] — the first 8 dims
# mirror the 8-D action layout (see scripts/convert_franka_raw_to_lerobot.py).
FRANKA_STATE_DIM = 29
FRANKA_ACTION_DIM = 8


def make_franka_example() -> dict:
    """Creates a random input example for the single-arm Franka policy.

    State is 29-D: [qw, qx, qy, qz, x, y, z, gripper, j0..j6, j0_vel..j6_vel,
    gripper_vel, fx, fy, fz, tx, ty, tz] (ee_pose + gripper + arm joint_pos +
    joint_vel + wrench). Actions are 8-D EE-space: [qw, qx, qy, qz, x, y, z,
    gripper] (target_pose + gripper; see scripts/convert_franka_raw_to_lerobot.py).
    Two cameras: a side (third-person) view and a wrist view.
    """
    return {
        "observation/state": np.random.rand(FRANKA_STATE_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Inputs transform for a single-arm Franka FR3 (8-D EE actions, 2 cameras).

    The full dataset state is 29-D: [qw, qx, qy, qz, x, y, z, gripper, j0..j6,
    j0_vel..j6_vel, gripper_vel, fx..tz]; actions are 8-D [qw, qx, qy, qz, x, y,
    z, gripper] (target_pose + gripper). ``state_dim`` selects the state prefix
    fed to the model: 29 (full proprio, default) or 8 (ee_pose + gripper only —
    dims 0-7 mirror the action layout, which DeltaActions relies on). At
    inference, a client may send either the full 29-D state or exactly
    ``state_dim`` dims. The two physical cameras map onto pi0.5's image slots as:
        base_0_rgb        <- side / third-person camera  (avantbot camera1)
        left_wrist_0_rgb  <- wrist camera                (avantbot camera0)
        right_wrist_0_rgb <- zeros (masked; Franka has no second wrist cam)

    The (possibly sliced) state and actions pass through unchanged; the
    model-side ``PadStatesAndActions`` pads them to the model action_dim (32).
    """

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType
    # State prefix width fed to the model: FRANKA_STATE_DIM (full) or 8 (lean).
    state_dim: int = FRANKA_STATE_DIM
    # If true, feed only the wrist camera: base_0_rgb is zeroed + masked off (the
    # standard missing-camera pattern), the wrist stays in its left_wrist_0_rgb
    # slot. Inference clients may then omit "observation/image" entirely.
    wrist_camera_only: bool = False

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        if state.shape[-1] == FRANKA_STATE_DIM:
            state = state[..., : self.state_dim]
        elif state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected {FRANKA_STATE_DIM}-D (full) or {self.state_dim}-D (sliced) Franka "
                f"state, got {state.shape}. 8-D-only datasets predate the 29-D schema — re-run "
                "scripts/convert_franka_raw_to_lerobot.py or update the client."
            )
        wrist_image = _parse_image(data["observation/wrist_image"])
        if self.wrist_camera_only:
            base_image = np.zeros_like(wrist_image)
        else:
            base_image = _parse_image(data["observation/image"])

        # Only pi0-FAST attends to padding images; mask them off for pi0/pi0.5.
        pad_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad the non-existent second wrist image with zeros.
                "right_wrist_0_rgb": np.zeros_like(wrist_image),
            },
            "image_mask": {
                "base_0_rgb": pad_mask if self.wrist_camera_only else np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": pad_mask,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Outputs transform: slice the model's padded action back to the Franka dim.

    quat8 (action_dim=8): [qw, qx, qy, qz, x, y, z, gripper] — absolute EE pose;
    renormalize the quaternion to unit length on the robot side before use.
    rot6d10 (action_dim=10): [x, y, z, r6_0..r6_5, gripper] — recover the rotation
    by Gram-Schmidt on the two 3-vectors on the robot side.
    """

    # 8 for the quat8 representation, 10 for rot6d10.
    action_dim: int = FRANKA_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        # Return the first action_dim dims; the rest is padding.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
