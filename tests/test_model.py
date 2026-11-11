import torch


def test_policy_samples_an_action_chunk(policy, batch, smoke_config):
    policy.eval()
    with torch.no_grad():
        prediction = policy.predict_action(batch["obs"])
    assert prediction["action"].shape == (2, smoke_config.n_action_steps, 2)
    assert prediction["action_pred"].shape == (2, smoke_config.horizon, 2)
    assert torch.isfinite(prediction["action"]).all()
