import hydra
import torch


def test_optimizer_updates_policy_on_finite_loss(policy, batch, smoke_config):
    policy.train()
    optimizer = hydra.utils.instantiate(smoke_config.optimizer, params=policy.parameters())
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in parameters]
    loss = policy.compute_loss(batch)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    optimizer.step()
    assert any(
        not torch.equal(parameter.detach(), initial)
        for parameter, initial in zip(parameters, before)
    )
