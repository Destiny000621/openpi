import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the single-arm Franka policy.

    State/action are 8-D EE-space: [qw, qx, qy, qz, x, y, z, gripper]
    (target_pose + gripper; see scripts/convert_franka_raw_to_lerobot.py). Two
    cameras: a side (third-person) view and a wrist view.
    """
    return {
        "observation/state": np.random.rand(8),
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
    """Inputs transform for a single-arm Franka FR3 (8-D EE-space, 2 cameras).

    State/action are [qw, qx, qy, qz, x, y, z, gripper] (EE pose + gripper). The
    two physical cameras map onto pi0.5's image slots as:
        base_0_rgb        <- side / third-person camera  (avantbot camera1)
        left_wrist_0_rgb  <- wrist camera                (avantbot camera0)
        right_wrist_0_rgb <- zeros (masked; Franka has no second wrist cam)

    State/actions are passed through unchanged (8-D); the model-side
    ``PadStatesAndActions`` pads them to the model action_dim (32).
    """

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad the non-existent second wrist image with zeros.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Only pi0-FAST attends to padding images; mask it off for pi0/pi0.5.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
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
    """Outputs transform: slice the model's padded action back to the Franka 8-D.

    Returns [qw, qx, qy, qz, x, y, z, gripper] — an absolute EE target pose the robot
    can send straight to the impedance controller / IK. Renormalize the quaternion to
    unit length on the robot side before use.
    """

    def __call__(self, data: dict) -> dict:
        # Return the first 8 action dims ([quat, xyz, gripper]); the rest is padding.
        return {"actions": np.asarray(data["actions"][:, :8])}
