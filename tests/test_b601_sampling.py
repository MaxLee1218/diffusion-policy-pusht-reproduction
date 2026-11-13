import json

import numpy as np
import pytest
import torch
from diffusers import DDPMScheduler

from scripts.benchmark_b601_sampling import (check_reference_predictions, make_scheduler,
                                            round_order, summarize_latency)


@pytest.mark.parametrize("steps", [100, 50, 20, 10])
def test_ddim_recovers_known_clean_sample_with_correct_strided_transition(steps):
    reference = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    scheduler = make_scheduler(reference, "ddim", steps)
    assert len(scheduler.timesteps) == steps
    torch.testing.assert_close(reference.alphas_cumprod, scheduler.alphas_cumprod, rtol=0, atol=0)
    clean = torch.tensor([[[.2, -.4, .6]]])
    noise = torch.tensor([[[-.1, .3, .5]]])
    for timestep in [scheduler.timesteps[0], scheduler.timesteps[-1]]:
        alpha = scheduler.alphas_cumprod[timestep]
        noisy = alpha.sqrt()*clean+(1-alpha).sqrt()*noise
        result = scheduler.step(noise, timestep, noisy, eta=0.)
        previous = int(timestep)-100//steps
        prev_alpha = scheduler.alphas_cumprod[previous] if previous >= 0 else torch.tensor(1.)
        expected = prev_alpha.sqrt()*clean+(1-prev_alpha).sqrt()*noise
        torch.testing.assert_close(result.prev_sample, expected, rtol=1e-5, atol=1e-6)
    # Include the exact betas in serializable audit metadata.
    serialized = json.loads(scheduler.to_json_string())
    np.testing.assert_array_equal(serialized["trained_betas"], reference.betas.numpy())


def test_full_ddpm_retains_original_forward_noise_schedule():
    reference = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    restored = make_scheduler(reference, "ddpm", 100)
    torch.testing.assert_close(reference.betas, restored.betas, rtol=0, atol=0)
    assert restored.timesteps.tolist() == list(range(99, -1, -1))


@pytest.mark.parametrize("family,steps", [("ddpm", 20), ("ddim", 0), ("ddim", 101),
                                         ("ddim", 33), ("unknown", 100)])
def test_rejects_unsupported_and_incorrect_reverse_schedules(family, steps):
    with pytest.raises(ValueError):
        make_scheduler(DDPMScheduler(num_train_timesteps=100), family, steps)


def test_latency_order_balances_each_variant_over_order_positions():
    order = np.array([round_order(round_index, 5) for round_index in range(100)])
    for column in order.T:
        np.testing.assert_array_equal(np.bincount(column), np.full(5, 20))


def test_latency_statistics_use_p95_not_maximum_and_reject_invalid_measurements():
    result = summarize_latency(list(range(1, 101)))
    assert result["median_seconds"] == 50.5
    assert result["p95_seconds"] == pytest.approx(95.05)
    assert result["max_seconds"] == 100
    for invalid in ([], [0.], [-1.], [float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            summarize_latency(invalid)


def test_reference_check_accepts_small_numerical_differences_but_rejects_prediction_changes():
    target = np.zeros((2, 8, 7), dtype=np.float32)
    previous = np.ones((2, 2, 8, 7), dtype=np.float32)
    result = check_reference_predictions(previous+.0001, previous, target, np.ones(7))
    assert result["relative_mse_difference"] < .001
    with pytest.raises(ValueError):
        check_reference_predictions(previous+.1, previous, target, np.ones(7))


def test_reference_check_rejects_small_motor_error_on_a_narrow_range_joint():
    target = np.zeros((2, 8, 7), dtype=np.float32)
    previous = np.ones((2, 2, 8, 7), dtype=np.float32)
    current = previous.copy()
    current[0, 0, 0, 4] += .002
    # A tiny motor-degree error may still represent a large normalized error.
    with pytest.raises(ValueError):
        check_reference_predictions(current, previous, target, np.array([1.]*4+[10.]+[1.]*2))
