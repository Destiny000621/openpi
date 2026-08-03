import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import umi_dual_franka_policy as policy


def _quat_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    return np.array([0.0, 0.0, np.sin(radians / 2.0), np.cos(radians / 2.0)], dtype=np.float64)


def _raw_pose(
    left_position=(0.0, 0.0, 0.0),
    left_quaternion=(0.0, 0.0, 0.0, 1.0),
    left_gripper=0.0,
    right_position=(0.0, 0.0, 0.0),
    right_quaternion=(0.0, 0.0, 0.0, 1.0),
    right_gripper=0.0,
) -> np.ndarray:
    return np.asarray(
        [
            *left_position,
            *left_quaternion,
            left_gripper,
            *right_position,
            *right_quaternion,
            right_gripper,
        ],
        dtype=np.float64,
    )


def _assert_same_raw_poses(actual: np.ndarray, expected: np.ndarray, *, atol=1e-6) -> None:
    np.testing.assert_allclose(actual[..., :3], expected[..., :3], atol=atol)
    np.testing.assert_allclose(actual[..., 7], expected[..., 7], atol=atol)
    np.testing.assert_allclose(actual[..., 8:11], expected[..., 8:11], atol=atol)
    np.testing.assert_allclose(actual[..., 15], expected[..., 15], atol=atol)
    for start in (3, 11):
        actual_rotation = policy.quaternion_xyzw_to_matrix(actual[..., start : start + 4])
        expected_rotation = policy.quaternion_xyzw_to_matrix(expected[..., start : start + 4])
        np.testing.assert_allclose(actual_rotation, expected_rotation, atol=atol)


def _sample_state_and_actions() -> tuple[np.ndarray, np.ndarray]:
    state = _raw_pose(
        left_position=(0.4, -0.2, 0.3),
        left_quaternion=_quat_z(90.0),
        left_gripper=0.2,
        right_position=(-0.3, 0.5, 0.1),
        right_quaternion=_quat_z(-35.0),
        right_gripper=0.8,
    )
    actions = np.stack(
        (
            _raw_pose(
                left_position=(0.5, -0.2, 0.35),
                left_quaternion=_quat_z(100.0),
                left_gripper=0.1,
                right_position=(-0.25, 0.48, 0.2),
                right_quaternion=_quat_z(-30.0),
                right_gripper=0.9,
            ),
            _raw_pose(
                left_position=(0.6, -0.1, 0.4),
                left_quaternion=_quat_z(120.0),
                left_gripper=0.7,
                right_position=(-0.1, 0.4, 0.3),
                right_quaternion=_quat_z(5.0),
                right_gripper=0.3,
            ),
        )
    )
    return state, actions


def _input_record(state: np.ndarray, actions: np.ndarray | None = None) -> dict:
    record = {
        "observation/state": state,
        "observation/left_head": np.full((3, 5, 7), 0.5, dtype=np.float32),
        "observation/right_head": np.full((5, 7, 3), 17, dtype=np.uint8),
        "prompt": b"Assemble the cardboard box and put it into the bin",
    }
    if actions is not None:
        record["actions"] = actions
    return record


def test_rotation_6d_is_paired_and_projects_to_so3() -> None:
    rotations = policy.quaternion_xyzw_to_matrix(np.stack((_quat_z(31.0), np.array([0.2, -0.3, 0.1, 0.9]))))
    rotation_6d = policy.matrix_to_rotation_6d(rotations)

    np.testing.assert_allclose(rotation_6d, rotations[..., :2, :].reshape(2, 6), atol=1e-6)
    decoded = policy.rotation_6d_to_matrix(rotation_6d)
    np.testing.assert_allclose(decoded, rotations, atol=1e-6)
    np.testing.assert_allclose(
        decoded @ np.swapaxes(decoded, -1, -2),
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        atol=1e-6,
    )
    np.testing.assert_allclose(np.linalg.det(decoded), np.ones(2), atol=1e-6)

    degenerate = policy.rotation_6d_to_matrix(np.zeros((4, 6), dtype=np.float32))
    np.testing.assert_allclose(
        degenerate @ np.swapaxes(degenerate, -1, -2), np.broadcast_to(np.eye(3), (4, 3, 3)), atol=1e-6
    )
    np.testing.assert_allclose(np.linalg.det(degenerate), np.ones(4), atol=1e-6)


def test_absolute_round_trip_and_quaternion_sign_invariance() -> None:
    state, _ = _sample_state_and_actions()
    sign_flipped = state.copy()
    sign_flipped[3:7] *= -1.0
    sign_flipped[11:15] *= -1.0

    model_state = policy.raw16_to_absolute20(state)
    flipped_model_state = policy.raw16_to_absolute20(sign_flipped)
    np.testing.assert_allclose(flipped_model_state, model_state, atol=1e-7)

    decoded = policy.absolute20_to_raw16(model_state)
    _assert_same_raw_poses(decoded, state)
    np.testing.assert_allclose(decoded[[7, 15]], state[[7, 15]])


def test_matrix_to_quaternion_handles_exact_half_turn_with_mixed_axis_signs() -> None:
    axis = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    quaternion = np.array([*axis, 0.0])
    rotation = policy.quaternion_xyzw_to_matrix(quaternion)

    decoded = policy.matrix_to_quaternion_xyzw(rotation)

    np.testing.assert_allclose(policy.quaternion_xyzw_to_matrix(decoded), rotation, atol=1e-6)


def test_relative_round_trip_uses_tool_frame_translation() -> None:
    state = _raw_pose(left_quaternion=_quat_z(90.0), right_position=(1.0, 2.0, 3.0))
    future = _raw_pose(
        left_position=(1.0, 0.0, 0.0),
        left_quaternion=_quat_z(135.0),
        left_gripper=0.75,
        right_position=(1.0, 3.0, 3.0),
        right_quaternion=_quat_z(-20.0),
        right_gripper=0.25,
    )[np.newaxis]

    relative = policy.raw16_actions_to_relative20(state, future)

    # A +world-X target is -tool-Y when the current tool is at +90 deg yaw.
    np.testing.assert_allclose(relative[0, :3], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(relative[0, 10:13], [0.0, 1.0, 0.0], atol=1e-6)
    decoded = policy.relative20_actions_to_raw16(policy.raw16_to_absolute20(state), relative)
    _assert_same_raw_poses(decoded, future)


def test_relative_chunk_has_one_fixed_anchor_and_preserves_grippers_and_arm_order() -> None:
    state = _raw_pose(left_position=(10.0, 0.0, 0.0), right_position=(-10.0, 0.0, 0.0))
    actions = np.stack(
        [
            _raw_pose(
                left_position=(11.0, 0.0, 0.0),
                left_gripper=0.1,
                right_position=(-9.0, 0.0, 0.0),
                right_gripper=0.9,
            ),
            _raw_pose(
                left_position=(12.0, 0.0, 0.0),
                left_gripper=0.2,
                right_position=(-8.0, 0.0, 0.0),
                right_gripper=0.8,
            ),
            _raw_pose(
                left_position=(13.0, 0.0, 0.0),
                left_gripper=0.3,
                right_position=(-7.0, 0.0, 0.0),
                right_gripper=0.7,
            ),
        ]
    )

    relative = policy.raw16_actions_to_relative20(state, actions)

    # These are 1, 2, 3 from the query state, not three incremental 1s.
    np.testing.assert_allclose(relative[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(relative[:, 10], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(relative[:, 9], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(relative[:, 19], [0.9, 0.8, 0.7])
    decoded = policy.relative20_actions_to_raw16(policy.raw16_to_absolute20(state), relative)
    _assert_same_raw_poses(decoded, actions)


def test_relative_inputs_map_fisheye_views_and_masks() -> None:
    state, actions = _sample_state_and_actions()
    transform = policy.UmiDualFrankaRelativeInputs(model_type=_model.ModelType.PI05)

    result = transform(_input_record(state, actions))

    assert result["state"].shape == (policy.MODEL_STATE_DIM,)
    assert result["actions"].shape == (2, policy.MODEL_STATE_DIM)
    assert result["image"]["left_wrist_0_rgb"].shape == (5, 7, 3)
    assert result["image"]["left_wrist_0_rgb"].dtype == np.uint8
    assert np.all(result["image"]["left_wrist_0_rgb"] == 128)
    assert np.all(result["image"]["right_wrist_0_rgb"] == 17)
    assert np.all(result["image"]["base_0_rgb"] == 0)
    assert result["image_mask"] == {
        "base_0_rgb": np.False_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.True_,
    }
    assert result["prompt"] == "Assemble the cardboard box and put it into the bin"
    _assert_same_raw_poses(policy.relative20_actions_to_raw16(result["state"], result["actions"]), actions)


def test_relative_outputs_slice_padding_and_support_batches() -> None:
    state, actions = _sample_state_and_actions()
    batch_states = np.stack((state, state))
    batch_actions = np.stack((actions, actions))
    model_states = policy.raw16_to_absolute20(batch_states)
    relative_actions = policy.raw16_actions_to_relative20(batch_states, batch_actions)
    padded_states = np.pad(model_states, ((0, 0), (0, 12)), constant_values=123.0)
    padded_actions = np.pad(relative_actions, ((0, 0), (0, 0), (0, 12)), constant_values=456.0)

    result = policy.UmiDualFrankaRelativeOutputs()({"state": padded_states, "actions": padded_actions})["actions"]

    assert result.shape == (2, 2, policy.RAW_STATE_DIM)
    _assert_same_raw_poses(result, batch_actions)


def test_absolute_baseline_inputs_and_outputs_are_paired_and_batch_safe() -> None:
    state, actions = _sample_state_and_actions()
    inputs = policy.UmiDualFrankaAbsoluteInputs(model_type=_model.ModelType.PI05)(_input_record(state, actions))

    np.testing.assert_allclose(inputs["state"], policy.raw16_to_absolute20(state))
    np.testing.assert_allclose(inputs["actions"], policy.raw16_to_absolute20(actions))

    batched_padded = np.pad(
        np.stack((inputs["actions"], inputs["actions"])),
        ((0, 0), (0, 0), (0, 12)),
        constant_values=-999.0,
    )
    decoded = policy.UmiDualFrankaAbsoluteOutputs()({"actions": batched_padded})["actions"]
    assert decoded.shape == (2, 2, policy.RAW_STATE_DIM)
    _assert_same_raw_poses(decoded, np.stack((actions, actions)))


def test_input_accepts_raw_lerobot_aliases_and_inference_without_actions() -> None:
    state, _ = _sample_state_and_actions()
    transform = policy.UmiDualFrankaRelativeInputs(model_type=_model.ModelType.PI05)
    result = transform(
        {
            "observation.state": state,
            "observation.images.left_head": np.zeros((4, 6, 3), dtype=np.uint8),
            "observation.images.right_head": np.zeros((4, 6, 3), dtype=np.uint8),
            "task": "fold",
        }
    )

    assert "actions" not in result
    assert result["state"].shape == (policy.MODEL_STATE_DIM,)
    assert result["prompt"] == "fold"


@pytest.mark.parametrize(
    ("record_update", "match"),
    [
        ({"observation/state": np.zeros(15, dtype=np.float32)}, "16"),
        ({"actions": np.zeros((3, 15), dtype=np.float32)}, "16"),
        ({"observation/state": np.zeros(16, dtype=np.int32)}, "floating dtype"),
        ({"observation/left_head": np.zeros((5, 7), dtype=np.uint8)}, "3-D"),
    ],
)
def test_input_rejects_invalid_shapes_and_dtypes(record_update: dict, match: str) -> None:
    state, actions = _sample_state_and_actions()
    record = _input_record(state, actions)
    record.update(record_update)
    transform = policy.UmiDualFrankaRelativeInputs(model_type=_model.ModelType.PI05)

    with pytest.raises((TypeError, ValueError), match=match):
        transform(record)


def test_zero_quaternion_and_non_pi05_are_rejected() -> None:
    state, _ = _sample_state_and_actions()
    state[3:7] = 0.0
    with pytest.raises(ValueError, match="non-zero"):
        policy.raw16_to_absolute20(state)

    with pytest.raises(ValueError, match=r"pi0\.5"):
        policy.UmiDualFrankaRelativeInputs(model_type=_model.ModelType.PI0)


def test_gripper_only_inputs_emit_2d_state_and_keep_relative_actions() -> None:
    state, actions = _sample_state_and_actions()
    full = policy.UmiDualFrankaRelativeInputs(model_type=_model.ModelType.PI05)(_input_record(state, actions))
    gripper_only = policy.UmiDualFrankaRelativeInputs(model_type=_model.ModelType.PI05, state_mode="gripper_only")(
        _input_record(state, actions)
    )

    assert gripper_only["state"].shape == (policy.GRIPPER_STATE_DIM,)
    np.testing.assert_allclose(gripper_only["state"], state[[7, 15]])
    # The relative-action anchor must come from the full raw state, not the
    # reduced policy state.
    np.testing.assert_allclose(gripper_only["actions"], full["actions"])
    assert gripper_only["image_mask"] == full["image_mask"]
    np.testing.assert_array_equal(gripper_only["image"]["left_wrist_0_rgb"], full["image"]["left_wrist_0_rgb"])
    assert gripper_only["prompt"] == full["prompt"]


def test_gripper_only_outputs_return_relative_chunks_for_client_composition() -> None:
    state, actions = _sample_state_and_actions()
    relative = policy.raw16_actions_to_relative20(state, actions)
    padded = np.pad(relative, ((0, 0), (0, 12)), constant_values=77.0)

    served = policy.UmiDualFrankaRelativeGripperOnlyOutputs()({"actions": padded})["actions"]

    assert served.shape == (2, policy.RAW_STATE_DIM)
    for start in (3, 11):
        np.testing.assert_allclose(np.linalg.norm(served[..., start : start + 4], axis=-1), 1.0, atol=1e-6)
        assert np.all(served[..., start + 3] >= 0.0)
    # The served chunk is exactly the relative decode against an identity anchor.
    identity20 = policy.raw16_to_absolute20(_raw_pose())
    _assert_same_raw_poses(served, policy.relative20_actions_to_raw16(identity20, relative))
    # Client-side composition with the query anchor recovers the absolute targets.
    composed = policy.relative20_actions_to_raw16(
        policy.raw16_to_absolute20(state), policy.raw16_to_absolute20(served)
    )
    _assert_same_raw_poses(composed, actions)
    np.testing.assert_allclose(served[..., [7, 15]], actions[..., [7, 15]], atol=1e-6)


def test_gripper_only_rejects_unknown_state_mode() -> None:
    with pytest.raises(ValueError, match="state_mode"):
        policy.UmiDualFrankaRelativeInputs(model_type=_model.ModelType.PI05, state_mode="nope")  # type: ignore[arg-type]


def test_image_crop_trims_views_and_leaves_state_actions_unchanged() -> None:
    state, actions = _sample_state_and_actions()
    record = _input_record(state, actions)
    full = policy.UmiDualFrankaRelativeInputs(model_type=_model.ModelType.PI05, state_mode="gripper_only")(
        dict(record)
    )
    cropped = policy.UmiDualFrankaRelativeInputs(
        model_type=_model.ModelType.PI05, state_mode="gripper_only", image_crop=3
    )(dict(record))

    # The 5x7 test images crop to the centered 3x3 square: rows 1..3, cols 2..4.
    assert cropped["image"]["left_wrist_0_rgb"].shape == (3, 3, 3)
    np.testing.assert_array_equal(
        cropped["image"]["left_wrist_0_rgb"], full["image"]["left_wrist_0_rgb"][1:4, 2:5]
    )
    np.testing.assert_array_equal(
        cropped["image"]["right_wrist_0_rgb"], full["image"]["right_wrist_0_rgb"][1:4, 2:5]
    )
    assert cropped["image"]["base_0_rgb"].shape == (3, 3, 3)
    np.testing.assert_allclose(cropped["state"], full["state"])
    np.testing.assert_allclose(cropped["actions"], full["actions"])


def test_image_crop_rejects_out_of_range_sides() -> None:
    state, actions = _sample_state_and_actions()
    for bad in (0, -1, 6):
        transform = policy.UmiDualFrankaRelativeInputs(model_type=_model.ModelType.PI05, image_crop=bad)
        with pytest.raises(ValueError, match="image_crop"):
            transform(_input_record(state, actions))
