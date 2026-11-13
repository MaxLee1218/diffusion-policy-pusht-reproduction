"""Check a local B601 checkpoint, held-out losses, sampling and optimizer resume."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/diffusion_policy")]

import dill
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


def to_device(batch, device):
    return dict_apply(batch, lambda tensor: tensor.to(device, non_blocking=True))


def optimizer_steps(optimizer):
    return max(int(state["step"]) for state in optimizer.state.values() if "step" in state)


def validate_epoch_metrics(epoch_metrics):
    for row in epoch_metrics:
        for key in ("train_loss", "val_loss"):
            if key not in row or not np.isfinite(row[key]):
                raise ValueError(f"Missing or nonfinite required epoch metric: {key}")
        # Action sampling runs only every sample_every epochs in full training.
        if "train_action_mse_error" in row and not np.isfinite(row["train_action_mse_error"]):
            raise ValueError("Nonfinite sampled training action error")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint = args.run_dir / "checkpoints/latest.ckpt"
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(args.run_dir))
    workspace.load_payload(payload, include_keys=["epoch", "global_step"])
    torch.testing.assert_close(workspace.model.state_dict(), payload["state_dicts"]["model"], rtol=0, atol=0)
    torch.testing.assert_close(workspace.ema_model.state_dict(), payload["state_dicts"]["ema_model"], rtol=0, atol=0)
    torch.testing.assert_close(workspace.optimizer.state_dict(), payload["state_dicts"]["optimizer"], rtol=0, atol=0)
    for model in (workspace.model, workspace.ema_model):
        if not all(torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
            raise ValueError("Nonfinite checkpoint tensor")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation = dataset.get_validation_dataset()
    device = torch.device(args.device)
    policy = workspace.ema_model.to(device).eval()
    torch.manual_seed(20261113)
    np.random.seed(20261113)
    loss_sum, sample_count = 0., 0
    with torch.inference_mode():
        for batch in DataLoader(validation, batch_size=cfg.val_dataloader.batch_size,
                                num_workers=2, shuffle=False, pin_memory=device.type == "cuda"):
            batch = to_device(batch, device)
            loss = policy.compute_loss(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite held-out EMA loss")
            count = len(batch["action"])
            loss_sum += loss.item()*count
            sample_count += count

    # One middle window from each held-out episode, with the complete 100-step sampler.
    episode_ids = np.searchsorted(dataset.episode_ends, validation.sampler.indices[:, 0], side="right")
    chosen = []
    for episode in dataset.split["validation_episode_indices"]:
        matches = np.flatnonzero(episode_ids == episode)
        chosen.append(int(matches[len(matches)//2]))
    batch = to_device(default_collate([validation[index] for index in chosen]), device)
    with torch.inference_mode():
        torch.manual_seed(101)
        predicted = policy.predict_action(batch["obs"])
        torch.manual_seed(101)
        repeated = policy.predict_action(batch["obs"])
    torch.testing.assert_close(predicted["action_pred"], repeated["action_pred"], rtol=1e-5, atol=1e-5)
    expected = (len(chosen), cfg.horizon, 7)
    if tuple(predicted["action_pred"].shape) != expected or not torch.isfinite(predicted["action_pred"]).all():
        raise ValueError("Invalid sampled action sequence")
    stats = policy.normalizer.get_input_stats()["action"]
    action = predicted["action_pred"]
    within_range = bool(((action >= stats["min"]) & (action <= stats["max"])).all())
    normalized_abs_max = policy.normalizer["action"].normalize(action).abs().max().item()
    # diffusers 0.11.1 multiplies clipped x0 by beta0/(1-alpha_bar0)
    # at the last step. Float32 subtraction makes this slightly different from 1.
    scheduler_bound = abs(float(policy.noise_scheduler.betas[0] /
                                (1-policy.noise_scheduler.alphas_cumprod[0])))
    within_scheduler_bound = normalized_abs_max <= scheduler_bound+1e-5
    if not within_scheduler_bound:
        raise ValueError(f"Sampled normalized actions exceed the scheduler bound: {normalized_abs_max}")
    overshoot = torch.maximum(torch.clamp(stats["min"]-action, min=0),
                              torch.clamp(action-stats["max"], min=0)).max().item()
    prediction_mse = torch.mean((action-batch["action"])**2).item()
    prediction_mae = torch.mean(torch.abs(action-batch["action"]), dim=(0, 1)).cpu().tolist()

    # Continue one optimizer update in memory; retain the archived checkpoint unchanged.
    workspace.model.to(device).train()
    optimizer_to(workspace.optimizer, device)
    train_batch = to_device(next(iter(DataLoader(dataset, batch_size=2, num_workers=0))), device)
    steps_before = optimizer_steps(workspace.optimizer)
    workspace.optimizer.zero_grad(set_to_none=True)
    loss = workspace.model.compute_loss(train_batch)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite restored-model loss")
    loss.backward()
    parameters = [p for p in workspace.model.parameters() if p.grad is not None]
    if not all(torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError("Nonfinite gradients after checkpoint restore")
    parameter = next(p for p in parameters if torch.count_nonzero(p.grad))
    before = parameter.detach().clone()
    workspace.optimizer.step()
    steps_after = optimizer_steps(workspace.optimizer)
    if steps_after != steps_before+1 or torch.equal(parameter.detach(), before):
        raise ValueError("Restored optimizer did not advance model weights")

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_array = action.cpu().numpy()
    with (args.output / "heldout_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        names = list(dataset.replay_buffer.root.attrs["joint_names"])
        writer.writerow(["episode_index", "horizon_step", *names])
        for i, episode in enumerate(dataset.split["validation_episode_indices"]):
            for step, row in enumerate(prediction_array[i]):
                writer.writerow([episode, step, *row.tolist()])
    logs = [json.loads(line) for line in (args.run_dir / "logs.json.txt").read_text().splitlines()]
    epoch_metrics = [line for line in logs if "val_loss" in line]
    validate_epoch_metrics(epoch_metrics)
    report = {
        "status": "pass", "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "runtime": json.loads((args.run_dir / "runtime.json").read_text()),
        "epoch_metrics": epoch_metrics,
        "dataset": json.loads((args.run_dir / "dataset_provenance.json").read_text()),
        "full_heldout_ema_noise_mse": loss_sum/sample_count,
        "full_heldout_windows": sample_count,
        "prediction": {"weights": "ema_model", "heldout_episode_indices": dataset.split["validation_episode_indices"],
                       "inference_steps": cfg.policy.num_inference_steps,
                       "trajectory_shape": list(action.shape), "finite": True,
                       "within_training_normalizer_range": within_range,
                       "within_scheduler_roundoff_bound": within_scheduler_bound,
                       "normalized_abs_max": normalized_abs_max,
                       "scheduler_final_clip_bound": scheduler_bound,
                       "max_training_range_overshoot_motor_degrees": overshoot,
                       "same_seed_repeat_matches": True,
                       "mse_motor_degrees_squared": prediction_mse,
                       "mae_motor_degrees_by_joint": prediction_mae},
        "checkpoint_restore": {"model_ema_optimizer_exactly_loaded": True,
                               "stored_epoch": int(workspace.epoch), "stored_global_step": int(workspace.global_step),
                               "optimizer_steps_before": steps_before, "optimizer_steps_after": steps_after,
                               "finite_gradients": True, "parameter_updated": True,
                               "original_checkpoint_unchanged": True},
        "notes": [f"This invocation configured {cfg.training.num_epochs} epochs with a per-epoch training batch cap of {cfg.training.max_train_steps}; null means complete passes.",
                  "Training-workspace val_loss uses main-model weights and the configured validation batch cap; the independent EMA loss covers all held-out windows.",
                  "Prediction errors are offline comparisons, not real-robot success metrics.",
                  "diffusers 0.11.1's last-step float32 coefficient can slightly exceed 1; action bounds are checked against its derived limit and the observed motor-degree overshoot is reported.",
                  "The upstream checkpoint stores counters before the end-of-epoch increment and does not store RNG/dataloader/scheduler state.",
                  "With capped epochs, upstream global_step includes extra end-of-epoch increments; optimizer state gives the actual update count.",
                  "The optimizer continuation check was in memory and did not overwrite the checkpoint.",
                  "No robot connection or motion was performed."],
    }
    (args.output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
