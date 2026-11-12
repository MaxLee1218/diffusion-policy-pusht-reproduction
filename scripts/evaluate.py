"""Evaluate a trusted Diffusion Policy workspace checkpoint on PushT."""

import argparse
import json
from pathlib import Path

import dill
import hydra
import torch
import wandb


def load_policy(checkpoint, device):
    payload = torch.load(
        checkpoint, map_location="cpu", pickle_module=dill, weights_only=False
    )
    config = payload["cfg"]
    policy = hydra.utils.instantiate(config.policy)
    weights = "ema_model" if config.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][weights])
    return config, policy.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")

    config, policy = load_policy(args.checkpoint, args.device)

    args.output_dir.mkdir(parents=True)
    runner = hydra.utils.instantiate(config.task.env_runner, output_dir=str(args.output_dir))
    try:
        torch.manual_seed(config.training.seed)
        metrics = runner.run(policy)
    finally:
        runner.env.close()

    report = {
        key: str(Path(value._path).relative_to(args.output_dir))
        if isinstance(value, wandb.Video)
        else float(value)
        for key, value in metrics.items()
    }
    (args.output_dir / "eval_log.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "eval_log.json")


if __name__ == "__main__":
    main()
