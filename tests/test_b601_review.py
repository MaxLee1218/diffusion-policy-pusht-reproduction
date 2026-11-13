"""Detect time-shifted labels and changes during a held command."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("matplotlib")
spec = importlib.util.spec_from_file_location(
    "review_b601", Path(__file__).resolve().parents[1] / "scripts/review_b601.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def example():
    timing = pd.DataFrame({
        "camera_received_monotonic_ns": [1000000000, 1100000000, 1200000000, 1300000000, 1400000000],
        "feedback_read_complete_monotonic_ns": [1001000000, 1101000000, 1201000000, 1301000000, 1401000000],
        "command_sent_complete_monotonic_ns": [np.nan, 1107000000, np.nan, 1307000000, np.nan],
    })
    action = np.zeros((5, 7), dtype=np.float32)
    action[1:3, -1] = 8
    return np.zeros_like(action), action, timing


def test_held_target_is_valid_even_when_measured_state_differs():
    state, action, timing = example()
    result = review.label_metrics(state, action, timing)
    assert result["action_change_rows"] == result["command_rows"] == 2
    assert result["held_transition_max_abs_change"] == 0
    assert result["action_changes_without_command"] == 0
    assert result["command_precedes_feedback_rows"] == 0
    assert result["command_after_feedback_ms"]["median"] == 6


def test_one_frame_label_shift_is_detected():
    state, action, timing = example()
    shifted = np.concatenate([action[:1], action[:-1]])
    result = review.label_metrics(state, shifted, timing)
    assert result["action_changes_without_command"] == 2
    assert result["commands_without_action_change_after_first_frame"] == 2
    assert result["held_transition_max_abs_change"] == 8


def test_reversed_command_feedback_order_is_detected():
    state, action, timing = example()
    timing.loc[1, "command_sent_complete_monotonic_ns"] = 1100000000
    result = review.label_metrics(state, action, timing)
    assert result["command_precedes_feedback_rows"] == 1
