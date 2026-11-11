import dill
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import (
    TrainDiffusionUnetHybridWorkspace,
)


def test_checkpoint_restores_policy_optimizer_and_progress(smoke_config, dataset, batch, tmp_path):
    config = OmegaConf.create(OmegaConf.to_container(smoke_config))
    workspace = TrainDiffusionUnetHybridWorkspace(config, output_dir=str(tmp_path))
    workspace.model.set_normalizer(dataset.get_normalizer())
    workspace.model.compute_loss(batch).backward()
    workspace.optimizer.step()
    workspace.global_step = 1
    workspace.epoch = 2
    path = workspace.save_checkpoint(use_thread=False)

    payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
    restored = TrainDiffusionUnetHybridWorkspace(payload["cfg"], output_dir=str(tmp_path))
    restored.load_payload(payload)

    assert restored.global_step == 1
    assert restored.epoch == 2
    torch.testing.assert_close(restored.model.state_dict(), workspace.model.state_dict())
    assert restored.optimizer.state_dict()["state"]
    torch.testing.assert_close(restored.optimizer.state_dict(), workspace.optimizer.state_dict())
