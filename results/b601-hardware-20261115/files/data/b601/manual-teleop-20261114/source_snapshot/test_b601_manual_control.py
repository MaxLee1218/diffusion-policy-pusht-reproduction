import importlib.util
from pathlib import Path

import numpy as np
import pytest

from real_robot.manual_control import GRIPPER_KEYS, MOVES, JogModel, KeyLatch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manual_reference", ROOT/"collection/b601/b601_pick_place.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


@pytest.fixture
def model():
    original = reference.Kinematics(reference.URDF)
    return JogModel({"origins": [value.tolist() for value in original.origins],
                     "axes": [value.tolist() for value in original.axes], "tool": original.tool.tolist(),
                     "limits": original.limits_deg.tolist()+[[-270., 0.]],
                     "bounds": [[.19, .38], [-.09, .113], [.004, .152]]})


def test_portable_model_matches_original_for_nonzero_joint_rotations(model):
    original = reference.Kinematics(reference.URDF)
    for q in ([0.]*6, [3., -98., -59., 52., 3., -4.], [-15., -110., -70., 48., -4., -17.]):
        np.testing.assert_allclose(model.pose(q), original.pose(q), rtol=0, atol=1e-12)


@pytest.mark.parametrize("key", list(MOVES))
def test_each_translation_key_requests_six_mm_and_preserves_orientation(model, key):
    current = np.array([3., -98., -59., 52., 3., -4., -20.])
    start = model.pose(current)
    target, destination = model.goal(current, key)
    np.testing.assert_allclose(destination-start[:3, 3], .006*np.array(MOVES[key]), atol=1e-12)
    np.testing.assert_allclose(model.pose(target)[:3, 3], destination, atol=.0001)
    np.testing.assert_allclose(model.pose(target)[:3, :3], start[:3, :3], atol=.001)
    assert target[6] == current[6]


def test_workspace_violation_rejects_entire_step_without_shortening(model):
    current = np.array([3., -98., -59., 52., 3., -4., -20.])
    x = model.pose(current)[0, 3]
    model.bounds[0, 1] = x+.003
    with pytest.raises(ValueError, match="6 mm 目标越界"):
        model.goal(current, "w")


def test_left_opens_and_right_closes_in_motor_units(model):
    current = np.array([3., -98., -59., 52., 3., -4., -20.])
    for key, sign in GRIPPER_KEYS.items():
        target, _ = model.goal(current, key)
        np.testing.assert_array_equal(target[:6], current[:6])
        assert target[6] == current[6]+sign*8
    current[6] = 0.
    with pytest.raises(ValueError, match="夹爪已到"):
        model.goal(current, "Right")


def test_repeated_keydown_requires_release_before_next_step():
    latch = KeyLatch()
    assert latch.press("w")
    assert not any(latch.press("w") for _ in range(20))
    latch.release("w")
    assert latch.press("w")
