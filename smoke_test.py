"""Measure a few PushT training iterations on a local CUDA GPU."""

import argparse
import copy
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent


def check_cuda(device):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("Run the GPU smoke test on Linux x86_64.")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access CUDA. Check the NVIDIA driver and Docker --gpus flag.")
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    architectures = torch.cuda.get_arch_list()
    # Native cubins support later minor revisions within the same major revision.
    supported = any(
        f"sm_{major}{revision}" in architectures for revision in range(minor + 1)
    )
    if not supported:
        raise SystemExit(f"Device sm_{major}{minor} has no compatible native target in {architectures}.")
    return {
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"{major}.{minor}",
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "torch_architectures": architectures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", choices=("baseline", "rtx4060_laptop"), default="baseline")
    parser.add_argument("--steps", type=int, choices=(2, 3), default=3)
    parser.add_argument("--batch-size", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    device = torch.device("cuda", args.device)
    report = check_cuda(device)
    upstream = ROOT / "upstream" / "diffusion_policy"
    if not (upstream / "train.py").is_file():
        parser.error("Initialize upstream with git submodule update --init.")
    sys.path.insert(0, str(upstream))

    with hydra.initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        config = hydra.compose(config_name=args.config_name)
    replay = ROOT / config.task.dataset.zarr_path
    if not replay.is_dir():
        parser.error("Run ./scripts/prepare_data.sh first.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = hydra.utils.instantiate(config.task.dataset, zarr_path=str(replay))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    batches = iter(loader)
    normalizer = dataset.get_normalizer()
    torch.cuda.reset_peak_memory_stats(device)

    policy = hydra.utils.instantiate(config.policy)
    policy.set_normalizer(normalizer)
    policy.to(device).train()
    optimizer = hydra.utils.instantiate(config.optimizer, params=policy.parameters())
    ema = hydra.utils.instantiate(config.ema, model=copy.deepcopy(policy))

    durations = []
    losses = []
    for _ in range(args.steps):
        # Time host-to-device transfer and training. Fetching from the dataset happens first.
        batch = next(batches)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        batch = {
            "obs": {key: value.to(device, non_blocking=True) for key, value in batch["obs"].items()},
            "action": batch["action"].to(device, non_blocking=True),
        }
        optimizer.zero_grad(set_to_none=True)
        loss = policy.compute_loss(batch)
        loss.backward()
        optimizer.step()
        ema.step(policy)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
        losses.append(loss.item())
        if not np.isfinite(losses[-1]):
            raise RuntimeError(f"Non-finite loss at step {len(losses)}")

    gib = 1024 ** 3
    report.update(
        config_name=args.config_name,
        seed=args.seed,
        batch_size=args.batch_size,
        steps=args.steps,
        model_parameters=sum(parameter.numel() for parameter in policy.parameters()),
        loss=losses,
        iteration_seconds=durations,
        median_iteration_seconds=statistics.median(durations[1:]),
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / gib,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device) / gib,
        gpu_total_gib=torch.cuda.get_device_properties(device).total_memory / gib,
    )
    summary = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        output.write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
