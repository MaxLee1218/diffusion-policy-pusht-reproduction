import copy

from omegaconf import OmegaConf
import torch

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from real_robot.workspace import ResumeAwareEMAModel, TrainB601Workspace


def small_workspace(path):
    # Exercise serialization and resume with tiny networks and no GPU/hardware.
    workspace = object.__new__(TrainB601Workspace)
    BaseWorkspace.__init__(workspace, OmegaConf.create({"test": True}), output_dir=str(path))
    workspace.model = torch.nn.Linear(2, 2)
    workspace.ema_model = copy.deepcopy(workspace.model)
    workspace.optimizer = torch.optim.AdamW(workspace.model.parameters(), lr=.001)
    workspace.epoch, workspace.global_step = 7, 10
    return workspace


def test_periodic_checkpoint_advances_epoch_and_restores_ema_update_count(tmp_path):
    workspace = small_workspace(tmp_path)
    workspace.model(torch.ones(2, 2)).sum().backward()
    workspace.optimizer.step()
    checkpoint = workspace.save_checkpoint()
    restored = small_workspace(tmp_path)
    restored.load_checkpoint(checkpoint, map_location="cpu", weights_only=False)
    assert (restored.epoch, restored.global_step) == (8, 11)
    torch.testing.assert_close(restored.optimizer.state_dict(), workspace.optimizer.state_dict())
    ema = ResumeAwareEMAModel(restored.ema_model)
    assert ema.optimization_step == 1
    ema.step(restored.model)
    assert ema.optimization_step == 2


def test_final_checkpoint_preserves_next_epoch_counters(tmp_path):
    workspace = small_workspace(tmp_path)
    workspace._at_training_end = True
    checkpoint = workspace.save_checkpoint()
    restored = small_workspace(tmp_path)
    restored.load_checkpoint(checkpoint, map_location="cpu", weights_only=False)
    assert (restored.epoch, restored.global_step) == (7, 10)
    assert restored.checkpoint_counters_are_next
