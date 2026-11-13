"""Measure frozen B601 EMA sampling accuracy and interleaved batch-one latency."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream/diffusion_policy")]

import diffusers
from diffusers import DDIMScheduler, DDPMScheduler
import dill
import hydra
import numpy as np
import torch
from torch.utils.data import default_collate

from scripts.evaluate_b601_checkpoints import metrics


VARIANTS = (("ddpm_100", 100), ("ddim_100", 100), ("ddim_50", 50),
            ("ddim_20", 20), ("ddim_10", 10))


def sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def make_scheduler(reference, family, steps):
    """Keep the learned forward process exact; never use old DDPM for strided steps."""
    training_steps = reference.config.num_train_timesteps
    if not 1 <= steps <= training_steps or training_steps % steps:
        raise ValueError("Use a positive inference step count dividing the training step count")
    if family == "ddpm":
        if steps != training_steps:
            raise ValueError("Pinned DDPM 0.11.1 does not implement strided reverse transitions")
        scheduler = DDPMScheduler.from_config(reference.config)
    elif family == "ddim":
        scheduler = DDIMScheduler(
            num_train_timesteps=training_steps,
            trained_betas=reference.betas.detach().cpu().numpy().copy(),
            clip_sample=reference.config.clip_sample,
            prediction_type=reference.config.prediction_type,
            set_alpha_to_one=True,
            steps_offset=0,
        )
    else:
        raise ValueError(f"Unknown sampler: {family}")
    torch.testing.assert_close(scheduler.betas, reference.betas, rtol=0, atol=0)
    torch.testing.assert_close(scheduler.alphas_cumprod, reference.alphas_cumprod, rtol=0, atol=0)
    scheduler.set_timesteps(steps)
    if len(scheduler.timesteps) != steps:
        raise ValueError("Scheduler produced an unexpected number of steps")
    return scheduler


def activate(policy, scheduler, steps):
    policy.noise_scheduler = scheduler
    policy.num_inference_steps = steps
    policy.kwargs = ({"eta": 0., "use_clipped_model_output": False}
                     if isinstance(scheduler, DDIMScheduler) else {})


def round_order(round_index, count):
    # Every variant occupies every order position once in each complete cycle.
    offset = round_index % count
    return [(offset + position) % count for position in range(count)]


def summarize_latency(seconds):
    values = np.asarray(seconds, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Latency measurements must be finite and positive")
    return {"measurements": len(values), "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.quantile(values, .95)),
            "min_seconds": float(values.min()), "max_seconds": float(values.max()),
            "mean_seconds": float(values.mean())}


def save_tables(report, output):
    fields = ["variant", "steps", "median_seconds", "p95_seconds", "normalized_mse",
              "arm_mae_motor_degrees", "gripper_mae_motor_degrees", "held_normalized_mse",
              "changing_normalized_mse", "episode_8_normalized_mse"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, result in report["variants"].items():
            all_metrics = result["metrics"]["all"]
            latency = result.get("latency", {})
            writer.writerow({"variant": name, "steps": result["steps"],
                             "median_seconds": latency.get("median_seconds"),
                             "p95_seconds": latency.get("p95_seconds"),
                             **{key: all_metrics[key] for key in fields[4:7]},
                             "held_normalized_mse": result["metrics"]["target_held"]["normalized_mse"],
                             "changing_normalized_mse": result["metrics"]["target_changes"]["normalized_mse"],
                             "episode_8_normalized_mse": result["metrics"]["episode_8"]["normalized_mse"]})


def plot_results(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(report["variants"])
    results = report["variants"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    median = [results[n]["latency"]["median_seconds"] for n in names]
    p95 = [results[n]["latency"]["p95_seconds"] for n in names]
    pos = np.arange(len(names))
    axes[0].bar(pos-.18, median, .36, label="Median")
    axes[0].bar(pos+.18, p95, .36, label="P95")
    axes[0].axhline(.8, color="gray", linestyle="--", label="0.8 s action chunk")
    axes[0].set_xticks(pos, names, rotation=25)
    axes[0].set(ylabel="Batch-one inference seconds", title="100 interleaved measurements / variant")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.2)
    baseline = results["ddpm_100"]["metrics"]
    groups = ["all", "target_held", "gripper_changes", "episode_8"]
    labels = ["All MSE", "Held MSE", "Gripper MAE", "Episode 8 MSE"]
    for i, (group, label) in enumerate(zip(groups, labels)):
        key = "gripper_mae_motor_degrees" if group == "gripper_changes" else "normalized_mse"
        values = [results[name]["metrics"][group][key] / baseline[group][key] for name in names]
        axes[1].plot(pos, values, marker="o", label=label)
    axes[1].axhline(1, color="gray", linestyle="--")
    axes[1].set_xticks(pos, names, rotation=25)
    axes[1].set(ylabel="Error / DDPM 100 error", title="Quality relative to current sampler")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    for name in names:
        result = results[name]
        x = result["latency"]["p95_seconds"]
        y = result["metrics"]["all"]["normalized_mse"]
        axes[2].scatter(x, y, s=45)
        axes[2].annotate(name, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
    axes[2].set(xlabel="P95 single-observation seconds", ylabel="Normalized action MSE",
                title="Lower and left is better")
    axes[2].grid(alpha=.2)
    fig.savefig(output / "speed_accuracy.png", dpi=170)
    plt.close(fig)


def run(args):
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status = {"status": "preparing", "started_at": now(), "hardware_connected": False}

    def progress(phase, **values):
        status.update(status=phase, updated_at=now(), **values)
        write_json(output / "status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    progress("preparing")
    try:
        prior = json.loads((args.reference / "evaluation.json").read_text())
        reference_model = prior["models"]["final_ema"]
        checkpoint = Path(reference_model["checkpoint"])
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != reference_model["checkpoint_sha256"]:
            raise ValueError("Final checkpoint changed since the previous evaluation")
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
        cfg = payload["cfg"]
        policy = hydra.utils.instantiate(cfg.policy)
        policy.load_state_dict(payload["state_dicts"]["ema_model"])
        del payload
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if sha256(dataset.split_path) != prior["protocol"]["split_sha256"]:
            raise ValueError("Frozen validation split changed")
        validation = dataset.get_validation_dataset()
        indices = prior["protocol"]["validation_dataset_indices"]
        episode_ids = np.searchsorted(dataset.episode_ends, validation.sampler.indices[indices, 0], side="right")
        np.testing.assert_array_equal(episode_ids, prior["protocol"]["episode_for_each_window"])
        batch = default_collate([validation[index] for index in indices])
        start, end = cfg.n_obs_steps-1, cfg.n_obs_steps-1+cfg.n_action_steps
        target = batch["action"][:, start:end].numpy()
        train_normalizer = dataset.get_normalizer()
        for key in ("action", "joint_pos"):
            for parameter in ("scale", "offset"):
                torch.testing.assert_close(policy.normalizer[key].params_dict[parameter],
                                           train_normalizer[key].params_dict[parameter], rtol=0, atol=0)
        scale = train_normalizer["action"].params_dict["scale"].detach().numpy()
        with np.load(args.reference / "predictions.npz") as prior_arrays:
            np.testing.assert_array_equal(target, prior_arrays["target"])
            np.testing.assert_array_equal(scale, prior_arrays["normalizer_scale"])
            prior_ddpm = prior_arrays["final_ema"].copy()
        seeds = prior["protocol"]["seeds"]
        reference_scheduler = policy.noise_scheduler
        schedulers = {name: make_scheduler(reference_scheduler, name.split("_")[0], steps)
                      for name, steps in VARIANTS}
        masks = {"all": np.ones(len(indices), dtype=bool),
                 "target_changes": np.max(np.abs(target-target[:, :1]), axis=(1, 2)) >= 1.,
                 "gripper_changes": np.max(np.abs(target[..., 6]-target[:, :1, 6]), axis=1) >= 1.}
        masks["target_held"] = ~masks["target_changes"]
        masks.update({f"episode_{ep}": episode_ids == ep for ep in np.unique(episode_ids)})
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("The deployment latency comparison requires an available CUDA device")
        torch.cuda.set_device(device)
        policy.to(device).eval()
        arrays = {"target": target, "validation_indices": np.array(indices),
                  "episode_indices": episode_ids, "normalizer_scale": scale}
        held = np.repeat(batch["obs"]["joint_pos"][:, -1:].numpy(), cfg.n_action_steps, axis=1)
        report = {
            "status": "running", "started_at": status["started_at"],
            "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash, "weights": "ema_model",
            "protocol": {"reference_evaluation": str(args.reference / "evaluation.json"),
                         "reference_sha256": sha256(args.reference / "evaluation.json"),
                         "validation_dataset_indices": indices, "episode_for_each_window": episode_ids.tolist(),
                         "split_sha256": prior["protocol"]["split_sha256"],
                         "accuracy_windows": len(indices), "accuracy_seeds": seeds,
                         "accuracy_batch_size": 8, "accuracy_seed_per_batch": "seed + first window ordinal",
                         "action_indices": [start, end], "precision": "float32",
                         "latency_batch_size": 1, "latency_warmup_per_variant": args.warmup,
                         "latency_measurements_per_variant": args.repetitions,
                         "latency_order": "round-robin variants, rotate starting variant each round",
                         "latency_observation_ordinal": "(round_index * 7) % 128",
                         "latency_seed": "9000 + round_index (identical across variants)",
                         "latency_scope": "CPU observation transfer + predict_action + CPU action return + CUDA synchronization",
                         "latency_excludes": "disk/image preprocessing, camera/robot I/O and RNG seed setup",
                         "noise_schedule": "exact copied training betas, 100 training timesteps",
                         "ddim": {"eta": 0., "set_alpha_to_one": True, "steps_offset": 0,
                                  "use_clipped_model_output": False,
                                  "timestep_spacing": "pinned scheduler default leading (last timestep zero)"},
                         "model_weights_modified": False, "hardware_connected": False},
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "diffusers": diffusers.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(device), "device": str(device),
                            "torch_threads": torch.get_num_threads(),
                            "cudnn_benchmark": torch.backends.cudnn.benchmark,
                            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32},
            "hold_baseline": {key: metrics(held[None], target, scale, mask) for key, mask in masks.items()},
            "variants": {},
            "limitations": ["Only four validation episodes, also used for checkpoint selection; no independent test success rate.",
                            "Single-observation latency excludes live camera, preprocessing and robot I/O.",
                            "An action chunk of 0.8 seconds is an overlap budget, not an end-to-end closed-loop guarantee.",
                            "Closeness in offline command error does not establish safe or successful robot behavior.",
                            "No training, hardware connection, robot motion or persistent inference configuration change."],
        }
        with torch.inference_mode():
            for name, steps in VARIANTS:
                progress("accuracy", variant=name, accuracy_variants_finished=len(report["variants"]))
                scheduler = schedulers[name]
                activate(policy, scheduler, steps)
                seed_predictions = []
                normalized_max = 0.
                for seed in seeds:
                    pieces = []
                    for offset in range(0, len(indices), 8):
                        obs = {key: value[offset:offset+8].to(device) for key, value in batch["obs"].items()}
                        torch.manual_seed(seed+offset)
                        prediction = policy.predict_action(obs)
                        full = prediction["action_pred"]
                        if not torch.isfinite(full).all():
                            raise ValueError(f"Nonfinite sampled actions: {name}")
                        normalized_max = max(normalized_max,
                                             policy.normalizer["action"].normalize(full).abs().max().item())
                        pieces.append(prediction["action"].cpu().numpy())
                    seed_predictions.append(np.concatenate(pieces))
                predictions = np.stack(seed_predictions)
                if normalized_max > 1.00005:
                    raise ValueError(f"Actions exceed known scheduler roundoff bound: {name}: {normalized_max}")
                result = {"steps": steps, "scheduler": type(scheduler).__name__,
                          "scheduler_config": json.loads(scheduler.to_json_string()),
                          "timesteps": scheduler.timesteps.tolist(),
                          "finite": True, "normalized_abs_max_full_horizon": normalized_max,
                          "metrics": {key: metrics(predictions, target, scale, mask) for key, mask in masks.items()}}
                # Do not select the best random sample based on the demonstrated future.
                if name == "ddpm_100":
                    difference = float(np.max(np.abs(predictions-prior_ddpm)))
                    result["previous_ddpm_predictions_max_abs_difference_motor_degrees"] = difference
                    np.testing.assert_allclose(predictions, prior_ddpm, rtol=1e-4, atol=1e-4)
                torch.manual_seed(777)
                one_obs = {key: value[:1].to(device) for key, value in batch["obs"].items()}
                first = policy.predict_action(one_obs)["action"]
                torch.manual_seed(777)
                second = policy.predict_action(one_obs)["action"]
                torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
                result["same_seed_repeat_matches"] = True
                report["variants"][name] = result
                arrays[name] = predictions
                write_json(output / "benchmark.json", report)
                np.savez_compressed(output / "predictions.npz", **arrays)
                save_tables(report, output)
                print(name, json.dumps(result["metrics"]["all"]), flush=True)

            def timed_prediction(variant_index, observation_index, seed):
                name, steps = VARIANTS[variant_index]
                activate(policy, schedulers[name], steps)
                obs = {key: value[observation_index:observation_index+1] for key, value in batch["obs"].items()}
                torch.manual_seed(seed)
                torch.cuda.synchronize(device)
                began = time.perf_counter()
                predicted = policy.predict_action({key: value.to(device) for key, value in obs.items()})["action"].cpu()
                torch.cuda.synchronize(device)
                seconds = time.perf_counter()-began
                if not torch.isfinite(predicted).all():
                    raise ValueError("Nonfinite action in batch-one latency measurement")
                return seconds

            progress("latency_warmup", variant=None)
            for repetition in range(args.warmup):
                for variant_index in round_order(repetition, len(VARIANTS)):
                    timed_prediction(variant_index, (repetition*7) % len(indices), 8000+repetition)
            measurements = {name: [] for name, _ in VARIANTS}
            with (output / "latency.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["round", "order_position", "variant", "validation_dataset_index", "seed", "seconds"])
                for repetition in range(args.repetitions):
                    observation_index = (repetition*7) % len(indices)
                    for position, variant_index in enumerate(round_order(repetition, len(VARIANTS))):
                        name, _ = VARIANTS[variant_index]
                        seconds = timed_prediction(variant_index, observation_index, 9000+repetition)
                        measurements[name].append(seconds)
                        writer.writerow([repetition, position, name, indices[observation_index], 9000+repetition, seconds])
                    handle.flush()
                    if repetition == 0 or (repetition+1) % 5 == 0 or repetition == args.repetitions-1:
                        for name in measurements:
                            report["variants"][name]["latency"] = summarize_latency(measurements[name])
                        write_json(output / "benchmark.json", report)
                        save_tables(report, output)
                        progress("latency", latency_rounds_finished=repetition+1,
                                 latency_rounds_total=args.repetitions)

        if sha256(checkpoint) != checkpoint_hash:
            raise ValueError("Checkpoint changed while benchmarking")
        report.update(status="complete", finished_at=now(), checkpoint_unchanged=True)
        write_json(output / "benchmark.json", report)
        save_tables(report, output)
        plot_results(report, output)
        source_paths = [Path(__file__), ROOT / "scripts/evaluate_b601_checkpoints.py"]
        for path in source_paths:
            destination = output / "source_snapshot" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        manifest_paths = [path for path in output.rglob("*") if path.is_file() and path.name not in ("status.json", "manifest.json")]
        write_json(output / "manifest.json", {"created_at": now(), "files": {str(path): sha256(path) for path in manifest_paths}})
        progress("complete", checkpoint_unchanged=True, hardware_connected=False)
    except BaseException as error:
        progress("failed", error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("results/b601-seed42/evaluation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 1:
        parser.error("warmup and repetitions must be positive")
    run(args)


if __name__ == "__main__":
    main()
