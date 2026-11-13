"""Compare B601 saved policies against baselines on a frozen offline protocol."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/diffusion_policy")]

import dill
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from diffusion_policy.common.pytorch_util import dict_apply


def metrics(predictions, target, scale, mask):
    """Average sampled errors, never select the best random sample after seeing GT."""
    if not np.any(mask):
        return {"windows": 0}
    error = predictions[:, mask] - target[None, mask]
    return {
        "windows": int(mask.sum()),
        "normalized_mse": float(np.mean((error * scale) ** 2)),
        "rmse_motor_degrees": float(np.sqrt(np.mean(error ** 2))),
        "arm_mae_motor_degrees": float(np.mean(np.abs(error[..., :6]))),
        "gripper_mae_motor_degrees": float(np.mean(np.abs(error[..., 6]))),
        "mae_motor_degrees_by_joint": np.mean(np.abs(error), axis=(0, 1, 2)).tolist(),
        "first_action_normalized_mse": float(np.mean((error[:, :, 0] * scale) ** 2)),
        "last_action_normalized_mse": float(np.mean((error[:, :, -1] * scale) ** 2)),
        "normalized_mse_by_seed": np.mean((error * scale) ** 2, axis=(1, 2, 3)).tolist(),
    }


def plot_breakdown(report, output):
    import matplotlib.pyplot as plt
    baseline = report["baselines"]["hold_measured_position"]
    model = report["models"]["final_ema"]["metrics"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    groups = ["all", "target_changes", "target_held", "gripper_changes"]
    positions = np.arange(len(groups))
    axes[0].bar(positions-.18, [baseline[g].get("normalized_mse", np.nan) for g in groups], .36,
                label="Hold measured position", color="#888888")
    axes[0].bar(positions+.18, [model[g].get("normalized_mse", np.nan) for g in groups], .36,
                label="Final EMA", color="#369c82")
    labels = [f"{label} ({model[group]['windows']})" for label, group in
              zip(["All", "Changing", "Held", "Gripper change"], groups)]
    axes[0].set_xticks(positions, labels, rotation=15)
    axes[0].set(ylabel="Executable-action normalized MSE", title="Errors by target-change subset")
    axes[0].legend(); axes[0].grid(axis="y", alpha=.2)
    episodes = report["protocol"]["validation_episodes"]
    gains = [(1-model[f"episode_{ep}"]["normalized_mse"] /
              baseline[f"episode_{ep}"]["normalized_mse"])*100 for ep in episodes]
    axes[1].bar([str(ep) for ep in episodes], gains,
                color=["#369c82" if gain >= 0 else "#c78132" for gain in gains])
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set(xlabel="Original held-out episode index (equally sampled)",
                ylabel="MSE reduction versus hold (%)", title="Improvement varies across episodes")
    axes[1].grid(axis="y", alpha=.2)
    fig.savefig(output / "prediction_breakdown.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short-checkpoint", type=Path, required=True)
    parser.add_argument("--windows-per-episode", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.windows_per_episode < 1:
        parser.error("windows-per-episode must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    epoch_rows = []
    for line in (args.run_dir / "logs.json.txt").open():
        row = json.loads(line)
        if "val_loss" in row:
            epoch_rows.append(row)
    best = min(epoch_rows, key=lambda r: r["val_loss"])
    best_path = args.run_dir / "checkpoints" / f"epoch={best['epoch']:04d}-val_loss={best['val_loss']:.4f}.ckpt"
    latest = args.run_dir / "checkpoints/latest.ckpt"
    payload = torch.load(latest, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    del payload
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation = dataset.get_validation_dataset()
    episode_ids = np.searchsorted(dataset.episode_ends, validation.sampler.indices[:, 0], side="right")
    chosen, selected_episodes = [], []
    for episode in dataset.split["validation_episode_indices"]:
        available = np.flatnonzero(episode_ids == episode)
        # Equal temporal bins, chosen before inspecting actions or predictions.
        positions = ((np.arange(args.windows_per_episode) + .5) * len(available)
                     / args.windows_per_episode).astype(int)
        indices = available[np.unique(positions)]
        chosen.extend(indices.tolist())
        selected_episodes.extend([episode] * len(indices))
    selected_episodes = np.array(selected_episodes)
    batch = default_collate([validation[index] for index in chosen])
    start, end = cfg.n_obs_steps - 1, cfg.n_obs_steps - 1 + cfg.n_action_steps
    target = batch["action"][:, start:end].numpy()
    state = batch["obs"]["joint_pos"][:, -1].numpy()
    normalizer = dataset.get_normalizer()
    scale = normalizer["action"].params_dict["scale"].detach().numpy()
    held = np.repeat(state[:, None], cfg.n_action_steps, axis=1)
    mean = dataset.training_values("action").mean(axis=0)
    constant = np.broadcast_to(mean, target.shape).copy()
    masks = {
        "all": np.ones(len(chosen), dtype=bool),
        "target_changes": np.max(np.abs(target - target[:, :1]), axis=(1, 2)) >= 1.,
        "gripper_changes": np.max(np.abs(target[..., 6] - target[:, :1, 6]), axis=1) >= 1.,
    }
    masks["target_held"] = ~masks["target_changes"]
    masks.update({f"episode_{ep}": selected_episodes == ep for ep in np.unique(selected_episodes)})
    report = {
        "protocol": {
            "validation_episodes": dataset.split["validation_episode_indices"],
            "validation_windows_total": len(validation), "sampled_windows": len(chosen),
            "sampling": "one window at the midpoint of each equal temporal bin, per episode",
            "validation_dataset_indices": chosen, "episode_for_each_window": selected_episodes.tolist(),
            "seeds": args.seeds, "seed_per_batch": "seed + first selected-window ordinal in batch",
            "batch_size": 8, "inference_steps": cfg.policy.num_inference_steps,
            "evaluated_action_indices": [start, end], "action_index_end_exclusive": True,
            "nominal_action_rate_hz": 10, "prediction_units": "absolute motor degrees",
            "normalizer": "limits, fitted only to training episodes; normalized MSE weights axes by training range",
            "target_change_threshold_motor_degrees": 1.,
            "full_validation_noise_seed": 20261113,
            "best_checkpoint_selection": "minimum logged full-validation main-model noise MSE",
            "best_checkpoint_formal_epoch": best["epoch"] - epoch_rows[0]["epoch"] + 1,
            "best_checkpoint_logged_epoch": best["epoch"],
            "split_sha256": hashlib.sha256(Path(dataset.split_path).read_bytes()).hexdigest(),
        },
        "training": {"epochs": len(epoch_rows), "first": epoch_rows[0], "best": best,
                     "last": epoch_rows[-1], "epoch_metrics": epoch_rows},
        "baselines": {}, "models": {},
        "limitations": [
            "These four validation episodes also selected checkpoints; this is not an independent test set.",
            "Overlapping windows within four episodes do not constitute independent robot trials.",
            "Targets are recorded commands, not achieved states or task success labels.",
            "Future demonstrations may be multimodal; sampled action error is not a task success probability.",
            "No robot was connected or moved. No closed-loop success rate is available.",
        ],
    }
    for name, prediction in [("hold_measured_position", held), ("training_mean_target", constant)]:
        report["baselines"][name] = {key: metrics(prediction[None], target, scale, mask)
                                      for key, mask in masks.items()}
    arrays = {"target": target, "measured_position": state, "validation_indices": np.array(chosen),
              "episode_indices": selected_episodes, "normalizer_scale": scale}
    configurations = [("short_ema", args.short_checkpoint, "ema_model"),
                      ("best_main", best_path, "model"), ("best_ema", best_path, "ema_model"),
                      ("final_main", latest, "model"), ("final_ema", latest, "ema_model")]
    device = torch.device(args.device)
    for name, path, weights in configurations:
        print(f"Evaluating {name}: {path}", flush=True)
        payload = torch.load(path, map_location="cpu", pickle_module=dill, weights_only=False)
        policy = hydra.utils.instantiate(payload["cfg"].policy)
        policy.load_state_dict(payload["state_dicts"][weights])
        del payload
        policy.to(device).eval()
        # Ensure comparisons use identical data normalization, not just matching shapes.
        torch.testing.assert_close(policy.normalizer["action"].params_dict["scale"].cpu(),
                                   normalizer["action"].params_dict["scale"], rtol=0, atol=0)
        torch.manual_seed(20261113)
        loss_sum, count = 0., 0
        with torch.inference_mode():
            for part in DataLoader(validation, batch_size=8, num_workers=2, shuffle=False):
                part = dict_apply(part, lambda t: t.to(device))
                loss = policy.compute_loss(part)
                loss_sum += loss.item() * len(part["action"])
                count += len(part["action"])
            predictions, batch_times = [], []
            for seed in args.seeds:
                pieces = []
                for offset in range(0, len(chosen), 8):
                    obs = {key: value[offset:offset+8].to(device) for key, value in batch["obs"].items()}
                    torch.manual_seed(seed + offset)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    began = time.perf_counter()
                    prediction = policy.predict_action(obs)["action"]
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    batch_times.append(time.perf_counter()-began)
                    pieces.append(prediction.cpu().numpy())
                predictions.append(np.concatenate(pieces))
        predictions = np.stack(predictions)
        if not np.isfinite(predictions).all() or not np.isfinite(loss_sum):
            raise ValueError(f"Nonfinite predictions or noise loss for {name}")
        result = {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "weights": weights, "full_validation_noise_mse": loss_sum/count,
                  "full_validation_windows": count, "finite": True,
                  "median_batch_inference_seconds": float(np.median(batch_times)),
                  "metrics": {key: metrics(predictions, target, scale, mask) for key, mask in masks.items()}}
        report["models"][name] = result
        arrays[name] = predictions
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        np.savez_compressed(args.output / "predictions.npz", **arrays)
        print(name, json.dumps(result["metrics"]["all"]), flush=True)
        del policy
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    epochs = np.arange(1, len(epoch_rows)+1)
    axes[0].plot(epochs, [r["train_loss"] for r in epoch_rows], label="Train noise MSE")
    axes[0].plot(epochs, [r["val_loss"] for r in epoch_rows], label="Validation noise MSE (main)")
    axes[0].axvline(report["protocol"]["best_checkpoint_formal_epoch"], color="gray", linestyle="--")
    axes[0].set(xlabel="Full training epoch", ylabel="Noise MSE", yscale="log")
    axes[0].legend(); axes[0].grid(alpha=.2)
    names = ["hold", "short EMA", "best main", "best EMA", "final main", "final EMA"]
    values = [report["baselines"]["hold_measured_position"]["all"]["normalized_mse"]]
    values.extend(report["models"][name]["metrics"]["all"]["normalized_mse"]
                  for name, _, _ in configurations)
    axes[1].bar(names, values, color=["#888888", "#9b9bd4", "#369c82", "#16785e", "#e6a45a", "#c78132"])
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set(ylabel=f"{cfg.n_action_steps}-action normalized MSE", yscale="log",
                title=f"{len(chosen)} held-out windows, mean over {len(args.seeds)} seeds")
    axes[1].grid(axis="y", alpha=.2)
    fig.savefig(args.output / "evaluation.png", dpi=160)
    plot_breakdown(report, args.output)
    print(f"Saved evaluation to {args.output}", flush=True)


if __name__ == "__main__":
    main()
