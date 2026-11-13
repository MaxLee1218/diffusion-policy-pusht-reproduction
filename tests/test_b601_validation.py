import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "validate_b601", Path(__file__).resolve().parents[1] / "scripts/validate_b601_training.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_epoch_without_scheduled_action_sampling_is_valid():
    module.validate_epoch_metrics([
        {"train_loss": .01, "val_loss": .02},
        {"train_loss": .01, "val_loss": .02, "train_action_mse_error": .1},
    ])


@pytest.mark.parametrize("row", [
    {"train_loss": .01},
    {"train_loss": float("nan"), "val_loss": .02},
    {"train_loss": .01, "val_loss": .02, "train_action_mse_error": float("inf")},
])
def test_required_losses_and_present_action_metrics_must_be_finite(row):
    with pytest.raises(ValueError):
        module.validate_epoch_metrics([row])
