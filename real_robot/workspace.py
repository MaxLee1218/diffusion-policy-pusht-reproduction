"""Measure the existing Diffusion Policy workspace on B601 demonstrations."""

import json
from pathlib import Path
import time

import hydra
from omegaconf import OmegaConf
import torch

from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from diffusion_policy.model.diffusion.ema_model import EMAModel
from real_robot.dataset import sha256


def optimizer_update_count(optimizer):
    return max((int(state["step"]) for state in optimizer.state.values() if "step" in state), default=0)


class ResumeAwareEMAModel(EMAModel):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.optimization_step = int(getattr(model, "_b601_optimizer_updates", 0))


class TrainB601Workspace(TrainDiffusionUnetHybridWorkspace):
    include_keys = tuple(TrainDiffusionUnetHybridWorkspace.include_keys) + ("checkpoint_counters_are_next",)

    def save_checkpoint(self, *args, **kwargs):
        self.checkpoint_counters_are_next = bool(getattr(self, "_at_training_end", False))
        path = super().save_checkpoint(*args, **kwargs)
        if self._saving_thread is not None:
            self._saving_thread.join()
        return path

    def load_checkpoint(self, *args, **kwargs):
        payload = super().load_checkpoint(*args, **kwargs)
        # Upstream periodic checkpoints store the completed epoch before increment.
        # Our final checkpoints store the counters already advanced to the next epoch.
        if not getattr(self, "checkpoint_counters_are_next", False):
            self.epoch += 1
            self.global_step += 1
        updates = optimizer_update_count(self.optimizer)
        self.ema_model._b601_optimizer_updates = updates
        self._resume_info = {"source_epoch": int(dill_value(payload, "epoch")),
                             "next_epoch": int(self.epoch), "global_step": int(self.global_step),
                             "optimizer_updates": updates, "ema_initial_optimization_step": updates}
        (Path(self.output_dir) / "resume_state.json").write_text(json.dumps(self._resume_info, indent=2)+"\n")
        return payload

    def run(self):
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        dataset = hydra.utils.instantiate(self.cfg.task.dataset)
        provenance = {
            "split": dataset.split, "selection_sha256": sha256(dataset.selection_path),
            "split_sha256": sha256(dataset.split_path),
            "train_windows": len(dataset), "validation_windows": len(dataset.get_validation_dataset()),
            "normalizer_fit": "training episodes only", "observation_history": self.cfg.n_obs_steps,
            "prediction_horizon": self.cfg.horizon, "action_dimensions": 7,
            "image_shape": list(dataset.replay_buffer["img"].shape[1:]),
            "image_color": "RGB", "crop": None, "hardware_connected": False,
        }
        (output / "dataset_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
        OmegaConf.save(self.cfg, output / "effective_config.yaml", resolve=True)
        device = torch.device(self.cfg.training.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        self._at_training_end = False
        super().run()
        self._at_training_end = True
        self.save_checkpoint()
        if self._saving_thread is not None:
            self._saving_thread.join()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metrics = {
            "elapsed_seconds": time.perf_counter()-started, "device": str(device),
            "epochs_completed_this_invocation": self.cfg.training.num_epochs,
            "global_step": int(self.global_step), "epoch_counter": int(self.epoch),
            "model_parameters": sum(p.numel() for p in self.model.parameters()),
            "training_precision": "float32", "ema": bool(self.cfg.training.use_ema),
            "optimizer_updates": optimizer_update_count(self.optimizer),
            "resume": getattr(self, "_resume_info", None),
        }
        if device.type == "cuda":
            metrics.update(gpu=torch.cuda.get_device_name(device),
                           peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
        (output / "runtime.json").write_text(json.dumps(metrics, indent=2)+"\n")


def dill_value(payload, key):
    import dill
    return dill.loads(payload["pickles"][key])
