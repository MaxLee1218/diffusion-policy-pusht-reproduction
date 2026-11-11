import torch


def test_unet_predicts_noise_with_the_action_shape(policy, batch):
    with torch.no_grad():
        normalized = policy.normalizer["action"].normalize(batch["action"])
        observations = policy.normalizer.normalize(batch["obs"])
        frames = {
            key: value[:, :2].flatten(0, 1) for key, value in observations.items()
        }
        conditioning = policy.obs_encoder(frames).reshape(normalized.shape[0], -1)
        noise = torch.randn_like(normalized)
        timesteps = torch.tensor([0, 50])
        noisy_actions = policy.noise_scheduler.add_noise(normalized, noise, timesteps)
        predicted_noise = policy.model(noisy_actions, timesteps, global_cond=conditioning)

    assert predicted_noise.shape == noise.shape == batch["action"].shape == (2, 16, 2)
    assert torch.isfinite(predicted_noise).all()


def test_policy_samples_finite_actions_within_the_workspace(policy, batch):
    policy.eval()
    observations = {key: value[:, :2] for key, value in batch["obs"].items()}
    with torch.no_grad():
        torch.manual_seed(42)
        prediction = policy.predict_action(observations)
        torch.manual_seed(43)
        another_prediction = policy.predict_action(observations)

    trajectory = prediction["action_pred"]
    assert trajectory.shape == (2, 16, 2)
    assert prediction["action"].shape == (2, 8, 2)
    torch.testing.assert_close(prediction["action"], trajectory[:, 1:9])
    assert torch.isfinite(trajectory).all()
    assert not torch.allclose(trajectory, another_prediction["action_pred"])

    for sampled_actions in (trajectory, another_prediction["action_pred"]):
        assert (sampled_actions >= 0).all()
        assert (sampled_actions <= 512).all()
