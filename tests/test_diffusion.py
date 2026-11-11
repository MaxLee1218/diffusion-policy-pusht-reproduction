import torch


def test_scheduler_noise_matches_closed_form(policy):
    scheduler = policy.noise_scheduler
    actions = torch.ones((2, 16, 2))
    noise = torch.full_like(actions, -0.5)
    timesteps = torch.tensor([0, scheduler.config.num_train_timesteps - 1])
    alpha = scheduler.alphas_cumprod[timesteps, None, None]
    expected = alpha.sqrt() * actions + (1 - alpha).sqrt() * noise
    torch.testing.assert_close(scheduler.add_noise(actions, noise, timesteps), expected)
    assert scheduler.config.prediction_type == "epsilon"
