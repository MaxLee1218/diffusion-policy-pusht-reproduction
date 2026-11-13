"""Supervise a full B601 training invocation and validate its final checkpoint."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(output, status):
    temporary = output / "training_status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2)+"\n")
    temporary.replace(output / "training_status.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.run_dir.exists():
        parser.error(f"Choose a fresh run directory: {args.run_dir}")
    if not args.source_checkpoint.is_file():
        parser.error(f"Missing source checkpoint: {args.source_checkpoint}")
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"status": "preparing", "started_at": now(), "supervisor_pid": os.getpid(),
              "process_group": os.getpgrp(), "run_dir": str(args.run_dir),
              "output_dir": str(args.output), "profile": "b601_train",
              "planned_full_epochs": 100, "hardware_connected": False}
    write_status(args.output, status)
    try:
        (args.run_dir / "checkpoints").mkdir(parents=True)
        destination = args.run_dir / "checkpoints/latest.ckpt"
        shutil.copy2(args.source_checkpoint, destination)
        source_hash = hashlib.sha256(args.source_checkpoint.read_bytes()).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("Copied checkpoint checksum differs")
        provenance = {"source_checkpoint": str(args.source_checkpoint), "sha256": source_hash,
                      "restore": ["model", "ema_model", "optimizer"],
                      "ema_step_source": "restored optimizer update count",
                      "historical_counters": "continue at the next epoch; source short run used capped epochs",
                      "lr_schedule": "cosine over the full training budget; no second warmup"}
        (args.run_dir / "initial_checkpoint.json").write_text(json.dumps(provenance, indent=2)+"\n")
        (args.output / "initial_checkpoint.json").write_text(json.dumps(provenance, indent=2)+"\n")
        environment = os.environ.copy()
        environment.update(PUSHT_CONFIG="b601_train", PYTHON=sys.executable,
                           OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           WANDB_MODE="disabled", WANDB_SILENT="true")
        command = ["bash", "scripts/train.sh", str(args.run_dir), "--resume"]
        status.update(status="training", training_command=command, source_checkpoint_sha256=source_hash)
        with (args.run_dir / "train.log").open("x") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            status["training_pid"] = process.pid
            write_status(args.output, status)
            code = process.wait()
        if code:
            raise RuntimeError(f"Training exited with code {code}; see {args.run_dir / 'train.log'}")
        status.update(status="validating_checkpoint", training_finished_at=now())
        write_status(args.output, status)
        with (args.output / "checkpoint-validation.log").open("w") as log:
            subprocess.run([sys.executable, "scripts/validate_b601_training.py", str(args.run_dir),
                            "--output", str(args.output), "--device", "cuda:0"],
                           env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        validation = json.loads((args.output / "validation.json").read_text())
        for name in ("runtime.json", "dataset_provenance.json", "effective_config.yaml", "logs.json.txt"):
            shutil.copy2(args.run_dir / name, args.output / name)
        status.update(status="complete", finished_at=now(),
                      runtime=validation["runtime"],
                      full_heldout_ema_noise_mse=validation["full_heldout_ema_noise_mse"])
        write_status(args.output, status)
    except Exception as error:
        status.update(status="failed", finished_at=now(), error=str(error))
        write_status(args.output, status)
        raise


if __name__ == "__main__":
    main()
