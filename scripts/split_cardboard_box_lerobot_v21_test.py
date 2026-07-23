import numpy as np
import pyarrow as pa
import pytest

from scripts import split_cardboard_box_lerobot_v21 as splitter


def _segment(**overrides) -> splitter.Segment:
    fields = {
        "source_episode": 0,
        "box_index": 0,
        "start_frame": 0,
        "end_frame": 60,
        "split": "train",
        "task": splitter.DEFAULT_TASK,
    }
    fields.update(overrides)
    return splitter.Segment(**fields)


def test_validate_segments_rejects_source_episode_split_leakage():
    segments = [
        _segment(box_index=0, start_frame=0, end_frame=60, split="train"),
        _segment(box_index=1, start_frame=60, end_frame=120, split="val"),
    ]

    with pytest.raises(ValueError, match="collection-session leakage"):
        splitter.validate_segments(segments, {0: 120}, action_horizon=50)


def test_validate_segments_rejects_overlap():
    segments = [
        _segment(box_index=0, start_frame=0, end_frame=70),
        _segment(box_index=1, start_frame=60, end_frame=120),
    ]

    with pytest.raises(ValueError, match="Overlapping segments"):
        splitter.validate_segments(segments, {0: 120}, action_horizon=50)


def test_slice_rewrites_indices_and_terminal_action():
    states = np.arange(5 * 16, dtype=np.float32).reshape(5, 16)
    actions = np.concatenate([states[1:], states[-1:]], axis=0)
    vector_type = pa.list_(pa.float32(), 16)
    table = pa.table(
        {
            "observation.state": pa.array(states.tolist(), type=vector_type),
            "action": pa.array(actions.tolist(), type=vector_type),
            "episode_index": pa.array([3] * 5, type=pa.int64()),
            "frame_index": pa.array(range(5), type=pa.int64()),
            "index": pa.array(range(100, 105), type=pa.int64()),
            "task_index": pa.array([8] * 5, type=pa.int64()),
            "timestamp": pa.array(np.arange(5) / 30.0, type=pa.float32()),
        }
    )
    segment = _segment(start_frame=1, end_frame=5)

    output = splitter.slice_and_rewrite_table(
        table,
        segment,
        output_episode=7,
        global_index_offset=200,
        output_task_index=2,
        fps=30.0,
    )

    output_states = np.asarray(output.column("observation.state").to_pylist())
    output_actions = np.asarray(output.column("action").to_pylist())
    assert output.num_rows == 4
    np.testing.assert_array_equal(output.column("episode_index").to_pylist(), [7] * 4)
    np.testing.assert_array_equal(output.column("frame_index").to_pylist(), range(4))
    np.testing.assert_array_equal(output.column("index").to_pylist(), range(200, 204))
    np.testing.assert_array_equal(output.column("task_index").to_pylist(), [2] * 4)
    np.testing.assert_allclose(output_actions[:-1], output_states[1:])
    np.testing.assert_allclose(output_actions[-1], output_states[-1])


def test_slice_rejects_action_next_state_mismatch():
    states = np.zeros((3, 16), dtype=np.float32)
    actions = np.ones((3, 16), dtype=np.float32)
    vector_type = pa.list_(pa.float32(), 16)
    table = pa.table(
        {
            "observation.state": pa.array(states.tolist(), type=vector_type),
            "action": pa.array(actions.tolist(), type=vector_type),
            "episode_index": pa.array([0] * 3, type=pa.int64()),
            "frame_index": pa.array(range(3), type=pa.int64()),
            "index": pa.array(range(3), type=pa.int64()),
            "task_index": pa.array([0] * 3, type=pa.int64()),
            "timestamp": pa.array(np.arange(3) / 30.0, type=pa.float32()),
        }
    )

    with pytest.raises(ValueError, match=r"action\[t\] == observation.state\[t\+1\]"):
        splitter.slice_and_rewrite_table(
            table,
            _segment(start_frame=0, end_frame=3),
            output_episode=0,
            global_index_offset=0,
            output_task_index=0,
            fps=30.0,
        )
