import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "evaluate_b601", Path(__file__).resolve().parents[1] / "scripts/evaluate_b601_checkpoints.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_metrics_average_stochastic_samples_instead_of_best_of_seeds():
    target = np.zeros((2, 8, 7))
    predictions = np.stack([target, np.ones_like(target)*2])
    result = module.metrics(predictions, target, np.ones(7), np.array([True, False]))
    assert result["windows"] == 1
    assert result["normalized_mse"] == pytest.approx(2.)
    assert result["normalized_mse_by_seed"] == pytest.approx([0., 4.])


def test_joint_scaling_and_subset_do_not_mix_gripper_into_arm_error():
    target = np.zeros((2, 8, 7))
    predictions = np.zeros((1, 2, 8, 7))
    predictions[0, 0, :, 6] = 6.
    predictions[0, 1] = 100.
    result = module.metrics(predictions, target, np.array([1.]*6+[1/6]), np.array([True, False]))
    assert result["arm_mae_motor_degrees"] == 0.
    assert result["gripper_mae_motor_degrees"] == 6.
    assert result["normalized_mse"] == pytest.approx(1/7)
    assert module.metrics(predictions, target, np.ones(7), np.zeros(2, dtype=bool)) == {"windows": 0}
