"""OpenPI transforms for the dual-Franka cardboard-box UMI dataset.

The source dataset stores a 16-D absolute TCP state and action, ordered left
arm then right arm.  Each arm block is::

    [x, y, z, qx, qy, qz, qw, gripper]

This module exposes three representations for pi0.5:

* ``UmiDualFrankaRelativeInputs/Outputs`` (primary): absolute 20-D state and
  true SE(3), chunk-start-relative 20-D actions.
* ``UmiDualFrankaAbsoluteInputs/Outputs`` (baseline): absolute 20-D state and
  absolute 20-D actions.
* ``UmiDualFrankaRelativeInputs(state_mode="gripper_only")`` paired with
  ``UmiDualFrankaRelativeGripperOnlyOutputs`` (cross-embodiment variant): the
  policy state is the 2-D ``[left_gripper, right_gripper]`` vector — no pose
  dimensions at all — while actions stay fixed-anchor relative. The served
  chunk remains in the relative frame; the robot client composes it with its
  own query-time TCP anchors.

Each 20-D arm block is ``[xyz, rotation_6d, gripper]``.  The rotation-6D
convention is the first two *rows* of a rotation matrix, flattened row-major.
The paired decoder orthonormalizes those rows and projects the result to
SO(3).  Do not mix this encoder with a column-based rotation-6D decoder.
"""

import dataclasses
from typing import Literal

import numpy as np

from openpi import transforms
from openpi.models import model as _model

RAW_ARM_DIM = 8
RAW_STATE_DIM = 2 * RAW_ARM_DIM
MODEL_ARM_DIM = 10
MODEL_STATE_DIM = 2 * MODEL_ARM_DIM

# Policy state representations. "full" is the absolute 20-D pose; the other two
# carry no absolute pose (cross-embodiment): "gripper_only" is the 2-D absolute
# gripper vector, "relative_history" is the 20-D observation-window pose
# expressed in the current TCP frame plus the current grippers.
STATE_MODES = ("full", "gripper_only", "relative_history")
GRIPPER_STATE_DIM = 2

_EPS = 1e-8
_MISSING = object()


def _as_float_array(value, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{name} must have a floating dtype, got {array.dtype}.")
    if array.dtype.itemsize < np.dtype(np.float32).itemsize:
        array = array.astype(np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _require_last_dim(array: np.ndarray, dim: int, *, name: str) -> np.ndarray:
    if array.ndim < 1 or array.shape[-1] != dim:
        raise ValueError(f"{name} must have shape (..., {dim}), got {array.shape}.")
    return array


def _expand_anchor_for_actions(anchor: np.ndarray, actions: np.ndarray, *, name: str) -> np.ndarray:
    if anchor.ndim > actions.ndim:
        raise ValueError(f"{name} has too many dimensions for actions: anchor {anchor.shape}, actions {actions.shape}.")
    while anchor.ndim < actions.ndim:
        anchor = np.expand_dims(anchor, axis=-2)
    try:
        np.broadcast_shapes(anchor.shape[:-1], actions.shape[:-1])
    except ValueError as error:
        raise ValueError(
            f"{name} leading dimensions {anchor.shape[:-1]} do not broadcast with "
            f"action dimensions {actions.shape[:-1]}."
        ) from error
    return anchor


def project_matrix_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Return the closest proper rotation matrix under the Frobenius norm."""
    matrix = _as_float_array(matrix, name="matrix")
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape (..., 3, 3), got {matrix.shape}.")

    u, _, vh = np.linalg.svd(matrix)
    projected = u @ vh
    correction = np.where(np.linalg.det(projected) < 0.0, -1.0, 1.0)
    u = u.copy()
    u[..., :, -1] *= correction[..., np.newaxis]
    return u @ vh


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert unit or non-unit ``xyzw`` quaternions to rotation matrices."""
    quaternion = _as_float_array(quaternion, name="quaternion")
    _require_last_dim(quaternion, 4, name="quaternion")
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm <= _EPS):
        raise ValueError("quaternion norm must be non-zero.")
    qx, qy, qz, qw = np.moveaxis(quaternion / norm, -1, 0)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    row0 = np.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)), axis=-1)
    row1 = np.stack((2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)), axis=-1)
    row2 = np.stack((2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)), axis=-1)
    return np.stack((row0, row1, row2), axis=-2)


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert matrices to normalized ``xyzw`` quaternions with non-negative ``w``."""
    matrix = project_matrix_to_so3(matrix)
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    # q_abs is twice the absolute value of [w, x, y, z].  Construct one
    # candidate with each component as the denominator, then choose the most
    # numerically stable candidate.  This also handles exact 180-degree
    # rotations, where trace/sign-only formulae are ambiguous.
    q_abs = np.sqrt(
        np.maximum(
            np.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                axis=-1,
            ),
            0.0,
        )
    )
    candidates = np.stack(
        (
            np.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), axis=-1),
            np.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), axis=-1),
            np.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), axis=-1),
            np.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3] ** 2), axis=-1),
        ),
        axis=-2,
    )
    candidates /= 2.0 * np.maximum(q_abs[..., np.newaxis], _EPS)
    best = np.argmax(q_abs, axis=-1)
    quaternion_wxyz = np.take_along_axis(candidates, best[..., np.newaxis, np.newaxis], axis=-2)[..., 0, :]
    quaternion = quaternion_wxyz[..., (1, 2, 3, 0)]
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    quaternion = quaternion / np.maximum(norm, _EPS)
    return np.where(quaternion[..., 3:4] < 0.0, -quaternion, quaternion)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Encode the first two matrix rows in row-major order."""
    matrix = _as_float_array(matrix, name="matrix")
    if matrix.ndim < 2 or matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix must have shape (..., 3, 3), got {matrix.shape}.")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Decode row-based rotation-6D values with robust Gram-Schmidt projection."""
    rotation_6d = _as_float_array(rotation_6d, name="rotation_6d")
    _require_last_dim(rotation_6d, 6, name="rotation_6d")
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:6]

    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    default_first = np.zeros_like(first)
    default_first[..., 0] = 1.0
    first_unit = first / np.maximum(first_norm, _EPS)
    first_unit = np.where(first_norm > _EPS, first_unit, default_first)

    second_orthogonal = second - np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)

    # When a model predicts collinear/zero rows, choose the Cartesian basis
    # least aligned with row one and orthogonalize it.
    fallback_indices = np.argmin(np.abs(first_unit), axis=-1)
    fallback_basis = np.eye(3, dtype=rotation_6d.dtype)[fallback_indices]
    fallback_second = fallback_basis - np.sum(first_unit * fallback_basis, axis=-1, keepdims=True) * first_unit
    fallback_second /= np.maximum(np.linalg.norm(fallback_second, axis=-1, keepdims=True), _EPS)

    second_unit = second_orthogonal / np.maximum(second_norm, _EPS)
    second_unit = np.where(second_norm > _EPS, second_unit, fallback_second)
    third_unit = np.cross(first_unit, second_unit)

    return np.stack((first_unit, second_unit, third_unit), axis=-2)


def raw16_to_absolute20(raw: np.ndarray) -> np.ndarray:
    """Convert absolute dual-arm ``xyz + quat_xyzw + grip`` to absolute 20-D."""
    raw = _as_float_array(raw, name="raw state/action")
    _require_last_dim(raw, RAW_STATE_DIM, name="raw state/action")
    blocks = []
    for start in (0, RAW_ARM_DIM):
        rotation = quaternion_xyzw_to_matrix(raw[..., start + 3 : start + 7])
        blocks.append(
            np.concatenate(
                (
                    raw[..., start : start + 3],
                    matrix_to_rotation_6d(rotation),
                    raw[..., start + 7 : start + 8],
                ),
                axis=-1,
            )
        )
    return np.concatenate(blocks, axis=-1)


def absolute20_to_raw16(absolute: np.ndarray) -> np.ndarray:
    """Convert absolute dual-arm 20-D values back to absolute raw 16-D."""
    absolute = _as_float_array(absolute, name="absolute state/action")
    _require_last_dim(absolute, MODEL_STATE_DIM, name="absolute state/action")
    blocks = []
    for start in (0, MODEL_ARM_DIM):
        rotation = rotation_6d_to_matrix(absolute[..., start + 3 : start + 9])
        blocks.append(
            np.concatenate(
                (
                    absolute[..., start : start + 3],
                    matrix_to_quaternion_xyzw(rotation),
                    absolute[..., start + 9 : start + 10],
                ),
                axis=-1,
            )
        )
    return np.concatenate(blocks, axis=-1)


def raw16_to_gripper_state2(raw: np.ndarray) -> np.ndarray:
    """Extract the 2-D ``[left_gripper, right_gripper]`` policy state.

    Used by the gripper-only state mode: with a single observation timestep,
    pose-relative proprioception is identically zero, so the absolute pose is
    the only pose signal — and it is scene/marker specific. Dropping it keeps
    the policy embodiment-agnostic. Grippers stay absolute in ``[0, 1]`` with
    ``1 = open``.
    """
    raw = _as_float_array(raw, name="raw state")
    _require_last_dim(raw, RAW_STATE_DIM, name="raw state")
    return raw[..., (RAW_ARM_DIM - 1, RAW_STATE_DIM - 1)]


def raw16_actions_to_relative20(current_raw_state: np.ndarray, future_raw_actions: np.ndarray) -> np.ndarray:
    """Encode every future waypoint relative to the same current-state anchor.

    For each arm and every chunk waypoint, this computes
    ``T_relative = inv(T_current) @ T_future``.  Grippers remain future
    absolute targets and are never differenced.
    """
    current_raw_state = _as_float_array(current_raw_state, name="current raw state")
    future_raw_actions = _as_float_array(future_raw_actions, name="future raw actions")
    _require_last_dim(current_raw_state, RAW_STATE_DIM, name="current raw state")
    _require_last_dim(future_raw_actions, RAW_STATE_DIM, name="future raw actions")
    current_raw_state = _expand_anchor_for_actions(current_raw_state, future_raw_actions, name="current raw state")

    blocks = []
    for start in (0, RAW_ARM_DIM):
        current_position = current_raw_state[..., start : start + 3]
        current_rotation = quaternion_xyzw_to_matrix(current_raw_state[..., start + 3 : start + 7])
        future_position = future_raw_actions[..., start : start + 3]
        future_rotation = quaternion_xyzw_to_matrix(future_raw_actions[..., start + 3 : start + 7])

        relative_position = np.einsum("...ji,...j->...i", current_rotation, future_position - current_position)
        relative_rotation = np.einsum("...ji,...jk->...ik", current_rotation, future_rotation)
        blocks.append(
            np.concatenate(
                (
                    relative_position,
                    matrix_to_rotation_6d(relative_rotation),
                    future_raw_actions[..., start + 7 : start + 8],
                ),
                axis=-1,
            )
        )
    return np.concatenate(blocks, axis=-1)


def relative20_actions_to_raw16(current_absolute20_state: np.ndarray, relative20_actions: np.ndarray) -> np.ndarray:
    """Compose relative 20-D waypoints with their fixed absolute state anchor."""
    current_absolute20_state = _as_float_array(current_absolute20_state, name="current absolute 20-D state")
    relative20_actions = _as_float_array(relative20_actions, name="relative 20-D actions")
    _require_last_dim(current_absolute20_state, MODEL_STATE_DIM, name="current absolute 20-D state")
    _require_last_dim(relative20_actions, MODEL_STATE_DIM, name="relative 20-D actions")
    current_absolute20_state = _expand_anchor_for_actions(
        current_absolute20_state, relative20_actions, name="current absolute 20-D state"
    )

    blocks = []
    for model_start in (0, MODEL_ARM_DIM):
        current_position = current_absolute20_state[..., model_start : model_start + 3]
        current_rotation = rotation_6d_to_matrix(current_absolute20_state[..., model_start + 3 : model_start + 9])
        relative_position = relative20_actions[..., model_start : model_start + 3]
        relative_rotation = rotation_6d_to_matrix(relative20_actions[..., model_start + 3 : model_start + 9])

        absolute_position = current_position + np.einsum("...ij,...j->...i", current_rotation, relative_position)
        absolute_rotation = np.einsum("...ij,...jk->...ik", current_rotation, relative_rotation)
        blocks.append(
            np.concatenate(
                (
                    absolute_position,
                    matrix_to_quaternion_xyzw(absolute_rotation),
                    relative20_actions[..., model_start + 9 : model_start + 10],
                ),
                axis=-1,
            )
        )
    return np.concatenate(blocks, axis=-1)


ROT6D_ACTION_DIMS = (*range(3, 9), *range(MODEL_ARM_DIM + 3, MODEL_ARM_DIM + 9))


def neutralize_rot6d_norm_stats(norm_stats: dict | None, *, keys: tuple[str, ...] = ("actions",)) -> dict | None:
    """Neutralize the rot6d dimensions of the given norm-stat entries.

    Neutral parameters (mean 0 / std 1 / q01 -1 / q99 +1) make both z-score and
    quantile normalization pass those dimensions through unchanged (up to the
    normalizer's epsilon). rot6d components are rotation-matrix entries, already
    bounded in [-1, 1] and geometrically meaningful, so they are fed to the model
    raw and reach ``rotation_6d_to_matrix`` untouched. Translation and gripper
    dimensions keep their data-driven statistics.

    ``keys`` selects which entries to patch: ``"actions"`` always applies;
    ``"state"`` applies to the 20-D relative-history state, whose arm blocks use
    the identical ``[xyz(3), rot6d(6), gripper(1)]`` layout. Entries that are
    missing, or whose dimension is too small to contain rot6d (the 2-D
    gripper-only state), are left alone.
    """
    if norm_stats is None:
        return norm_stats

    def _patched(array, value):
        if array is None:
            return None
        flat = np.asarray(array).reshape(-1).copy()
        flat[list(ROT6D_ACTION_DIMS)] = value
        return flat

    patched = dict(norm_stats)
    for key in keys:
        entry = patched.get(key)
        if entry is None or np.asarray(entry.mean).reshape(-1).shape[-1] < MODEL_STATE_DIM:
            continue
        patched[key] = dataclasses.replace(
            entry,
            mean=_patched(entry.mean, 0.0),
            std=_patched(entry.std, 1.0),
            q01=_patched(entry.q01, -1.0),
            q99=_patched(entry.q99, 1.0),
        )
    return patched


def neutralize_rot6d_action_norm_stats(norm_stats: dict | None) -> dict | None:
    """Backwards-compatible wrapper: neutralize only the action rot6d dims."""
    return neutralize_rot6d_norm_stats(norm_stats, keys=("actions",))


def raw16_history_to_relative_state20(raw_states: np.ndarray) -> np.ndarray:
    """Encode an observation history as a relative 20-D policy state.

    ``raw_states`` carries the observation window oldest-first with shape
    ``(..., H, 16)``; the last row is the current frame. Per arm this returns::

        [history_rel_xyz(3), history_rel_rot6d(6), current_gripper_abs(1)]

    where the history pose is expressed in the CURRENT TCP frame,
    ``T_rel = inverse(T_current) @ T_history`` — the same true-SE(3) convention
    and row-major rot6d encoding the relative action targets use. Grippers are
    the current absolute values, never differenced.

    Like the actions, this carries no absolute pose, so it keeps the policy
    embodiment- and scene-agnostic while exposing recent motion (the signal a
    single-frame observation cannot provide). A 1-D ``(16,)`` input, or a window
    whose frames are identical (episode start, where LeRobot clamps the padded
    history to the current frame), yields an identity relative pose, i.e. "no
    motion".
    """
    raw_states = _as_float_array(raw_states, name="raw dual-Franka state history")
    _require_last_dim(raw_states, RAW_STATE_DIM, name="raw dual-Franka state history")
    current = raw_states if raw_states.ndim == 1 else raw_states[..., -1, :]
    history = raw_states if raw_states.ndim == 1 else raw_states[..., 0, :]

    # Reuse the action encoder for the pose math so the SE(3)/rot6d conventions
    # cannot drift apart; it takes the gripper from its second argument, so
    # overwrite both gripper slots with the current absolute values.
    state20 = np.array(raw16_actions_to_relative20(current, history), copy=True)
    for model_start, raw_start in ((0, 0), (MODEL_ARM_DIM, RAW_ARM_DIM)):
        state20[..., model_start + 9] = current[..., raw_start + 7]
    return state20


def relative20_actions_to_relative_raw16(relative20_actions: np.ndarray) -> np.ndarray:
    """Convert relative 20-D waypoints to raw-layout **relative** 16-D chunks.

    This is a pure representation change (rotation-6D to quaternion, identical
    math to ``absolute20_to_raw16``); the values stay fixed-anchor relative.
    Each arm block is ``[rel_xyz, rel_quat_xyzw, future_gripper_abs]``, with the
    translation and rotation still expressed in that arm's query-time TCP
    frame. The robot client composes every waypoint with its own saved
    query-time anchor: ``T_world_tcp_pred[k] = T_world_tcp_query @ T_rel[k]``.
    """
    return absolute20_to_raw16(relative20_actions)


def _get_first(data: dict, keys: tuple[str, ...], *, name: str, optional: bool = False):
    for key in keys:
        if key in data:
            return data[key]
    if optional:
        return _MISSING
    raise KeyError(f"Missing {name}; expected one of {keys}.")


def _get_image(data: dict, name: str):
    flat_keys = (
        f"observation/{name}",
        f"observation.images.{name}",
        f"observation/image/{name}",
        name,
    )
    value = _get_first(data, flat_keys, name=f"{name} image", optional=True)
    if value is not _MISSING:
        return value

    for container_key in ("images", "observation/images", "observation.images"):
        container = data.get(container_key)
        if isinstance(container, dict) and name in container:
            return container[name]
    raise KeyError(f"Missing {name} image; expected one of {flat_keys} or an images mapping.")


def _parse_image(value, *, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must be a 3-D HWC or CHW image, got {image.shape}.")
    if image.shape[-1] == 3:
        pass
    elif image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(f"{name} must have exactly three RGB channels, got {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        if not np.all(np.isfinite(image)):
            raise ValueError(f"{name} must contain only finite values.")
        minimum = float(np.min(image))
        maximum = float(np.max(image))
        if minimum >= 0.0 and maximum <= 1.0:
            image = image * 255.0
        elif minimum >= -1.0 and maximum <= 1.0:
            image = (image + 1.0) * 127.5
        elif minimum < 0.0 or maximum > 255.0:
            raise ValueError(f"{name} float values must lie in [0, 1], [-1, 1], or [0, 255].")
        image = np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)
    elif np.issubdtype(image.dtype, np.integer):
        if np.min(image) < 0 or np.max(image) > 255:
            raise ValueError(f"{name} integer values must lie in [0, 255].")
        image = image.astype(np.uint8, copy=False)
    else:
        raise TypeError(f"{name} must have an integer or floating dtype, got {image.dtype}.")
    return np.ascontiguousarray(image)


def _validate_model_type(model_type: _model.ModelType) -> None:
    if model_type != _model.ModelType.PI05:
        raise ValueError(f"UMI dual-Franka transforms are defined for pi0.5, got model type {model_type}.")


def center_crop_image(image: np.ndarray, side: int, *, name: str) -> np.ndarray:
    """Center-crop an HWC image to a ``side x side`` square.

    Used to trim the fisheye frame before the model resize: a 272 px crop of
    the 384 px ``front_equi`` view is the largest square inscribed in the
    fisheye circle, removing the dead black corners and most of the
    scene/operator periphery while keeping the gripper fingers and workspace.
    Applied in the shared input transform so training and serving stay
    identical.
    """
    if image.ndim != 3:
        raise ValueError(f"{name} must be a 3-D HWC image, got {image.shape}.")
    height, width = image.shape[0], image.shape[1]
    if side <= 0 or side > min(height, width):
        raise ValueError(f"image_crop must lie in (0, {min(height, width)}] for {name} of shape {image.shape}, got {side}.")
    top = (height - side) // 2
    left = (width - side) // 2
    return np.ascontiguousarray(image[top : top + side, left : left + side])


def _build_inputs(
    data: dict, *, relative_actions: bool, state_mode: str = "full", image_crop: int | None = None
) -> dict:
    if state_mode not in STATE_MODES:
        raise ValueError(f"state_mode must be one of {STATE_MODES}, got {state_mode!r}.")
    raw_state = _get_first(
        data,
        ("observation/state", "observation.state", "state"),
        name="raw dual-Franka state",
    )
    raw_state = _as_float_array(raw_state, name="raw dual-Franka state")
    _require_last_dim(raw_state, RAW_STATE_DIM, name="raw dual-Franka state")

    # An observation window arrives oldest-first as (..., H, 16). Everything
    # anchored on "now" — the relative-action anchor below included — must use
    # the last row; only the relative-history state reads the earlier frames.
    current_raw_state = raw_state if raw_state.ndim == 1 else raw_state[..., -1, :]

    match state_mode:
        case "gripper_only":
            state = raw16_to_gripper_state2(current_raw_state)
        case "relative_history":
            state = raw16_history_to_relative_state20(raw_state)
        case _:
            state = raw16_to_absolute20(current_raw_state)

    left_image = _parse_image(_get_image(data, "left_head"), name="left_head")
    right_image = _parse_image(_get_image(data, "right_head"), name="right_head")
    if image_crop is not None:
        left_image = center_crop_image(left_image, image_crop, name="left_head")
        right_image = center_crop_image(right_image, image_crop, name="right_head")
    inputs = {
        "state": state,
        "image": {
            "base_0_rgb": np.zeros_like(left_image),
            "left_wrist_0_rgb": left_image,
            "right_wrist_0_rgb": right_image,
        },
        "image_mask": {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
    }

    raw_actions = _get_first(data, ("actions", "action"), name="raw actions", optional=True)
    if raw_actions is not _MISSING:
        raw_actions = _as_float_array(raw_actions, name="raw dual-Franka actions")
        _require_last_dim(raw_actions, RAW_STATE_DIM, name="raw dual-Franka actions")
        inputs["actions"] = (
            raw16_actions_to_relative20(current_raw_state, raw_actions)
            if relative_actions
            else raw16_to_absolute20(raw_actions)
        )

    prompt = _get_first(data, ("prompt", "task"), name="prompt", optional=True)
    if prompt is not _MISSING:
        inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
    return inputs


@dataclasses.dataclass(frozen=True)
class UmiDualFrankaRelativeInputs(transforms.DataTransformFn):
    """Map raw UMI observations/actions to the primary pi0.5 representation.

    ``state_mode`` selects the policy state: ``"full"`` absolute 20-D pose,
    ``"gripper_only"`` the 2-D ``[left_gripper, right_gripper]`` vector, or
    ``"relative_history"`` the 20-D observation-window pose relative to the
    current TCP frame plus the current grippers. The relative action encoding is
    unchanged in every mode — its anchor is always the current raw absolute
    state, i.e. the last frame of the observation window.
    """

    model_type: _model.ModelType
    state_mode: Literal["full", "gripper_only", "relative_history"] = "full"
    # Optional centered square crop (pixel side length) applied to both camera
    # views before the model resize, e.g. 272 for the 384 px fisheye views.
    image_crop: int | None = None

    def __post_init__(self) -> None:
        _validate_model_type(self.model_type)
        if self.state_mode not in STATE_MODES:
            raise ValueError(f"state_mode must be one of {STATE_MODES}, got {self.state_mode!r}.")

    def __call__(self, data: dict) -> dict:
        return _build_inputs(
            data, relative_actions=True, state_mode=self.state_mode, image_crop=self.image_crop
        )


@dataclasses.dataclass(frozen=True)
class UmiDualFrankaRelativeOutputs(transforms.DataTransformFn):
    """Decode unnormalized relative model actions to absolute raw 16-D targets."""

    def __call__(self, data: dict) -> dict:
        actions = _as_float_array(data["actions"], name="model actions")
        state = _as_float_array(data["state"], name="model state")
        if actions.ndim < 1 or actions.shape[-1] < MODEL_STATE_DIM:
            raise ValueError(f"model actions must have at least {MODEL_STATE_DIM} values, got {actions.shape}.")
        if state.ndim < 1 or state.shape[-1] < MODEL_STATE_DIM:
            raise ValueError(f"model state must have at least {MODEL_STATE_DIM} values, got {state.shape}.")
        return {"actions": relative20_actions_to_raw16(state[..., :MODEL_STATE_DIM], actions[..., :MODEL_STATE_DIM])}


@dataclasses.dataclass(frozen=True)
class UmiDualFrankaRelativeGripperOnlyOutputs(transforms.DataTransformFn):
    """Decode unnormalized relative model actions to raw-layout relative 16-D chunks.

    The gripper-only configs deliberately keep absolute pose out of the policy,
    so the server cannot compose an absolute anchor. The response stays in the
    relative frame: per arm ``[rel_xyz, rel_quat_xyzw, gripper_abs]``. The
    robot client composes ``T_world_tcp_pred[k] = T_world_tcp_query @
    T_rel_pred[k]`` per arm with its own saved query-time anchor, following the
    same fixed-anchor rules as the primary relative config.
    """

    def __call__(self, data: dict) -> dict:
        actions = _as_float_array(data["actions"], name="model actions")
        if actions.ndim < 1 or actions.shape[-1] < MODEL_STATE_DIM:
            raise ValueError(f"model actions must have at least {MODEL_STATE_DIM} values, got {actions.shape}.")
        return {"actions": relative20_actions_to_relative_raw16(actions[..., :MODEL_STATE_DIM])}


@dataclasses.dataclass(frozen=True)
class UmiDualFrankaAbsoluteInputs(transforms.DataTransformFn):
    """Map raw UMI observations/actions to the absolute-action pi0.5 baseline."""

    model_type: _model.ModelType
    image_crop: int | None = None

    def __post_init__(self) -> None:
        _validate_model_type(self.model_type)

    def __call__(self, data: dict) -> dict:
        return _build_inputs(data, relative_actions=False, image_crop=self.image_crop)


@dataclasses.dataclass(frozen=True)
class UmiDualFrankaAbsoluteOutputs(transforms.DataTransformFn):
    """Decode unnormalized absolute 20-D model actions to raw absolute 16-D."""

    def __call__(self, data: dict) -> dict:
        actions = _as_float_array(data["actions"], name="model actions")
        if actions.ndim < 1 or actions.shape[-1] < MODEL_STATE_DIM:
            raise ValueError(f"model actions must have at least {MODEL_STATE_DIM} values, got {actions.shape}.")
        return {"actions": absolute20_to_raw16(actions[..., :MODEL_STATE_DIM])}
