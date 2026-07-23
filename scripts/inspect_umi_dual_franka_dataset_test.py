import numpy as np
import pytest

from openpi.policies import umi_dual_franka_policy as umi_policy
from scripts import inspect_umi_dual_franka_dataset as inspector


def test_action_next_state_error_checks_entire_chunk():
    horizon = 50
    states = np.zeros((horizon + 1, umi_policy.RAW_STATE_DIM), dtype=np.float32)
    actions = states[1:].copy()
    actions[-1, -1] = 0.25

    assert inspector._action_next_state_error(states, actions, horizon=horizon) == pytest.approx(0.25)  # noqa: SLF001


def test_action_next_state_error_rejects_short_state_chunk():
    horizon = 50
    states = np.zeros((horizon, umi_policy.RAW_STATE_DIM), dtype=np.float32)
    actions = np.zeros((horizon, umi_policy.RAW_STATE_DIM), dtype=np.float32)

    with pytest.raises(ValueError, match="Raw state/action chunks must have shapes"):
        inspector._action_next_state_error(states, actions, horizon=horizon)  # noqa: SLF001
