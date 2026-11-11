"""Run the local 4060 experiment and persist its process status."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
RUN = ROOT / "experiments/rtx4060-seed42"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_status(status):
    temporary = REPORT / "status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(REPORT / "status.json")


if __name__ == "__main__":
    os.chdir(ROOT)
    if RUN.exists():
        raise SystemExit(f"Run directory already exists: {RUN}")
    env = os.environ.copy()
    env.update(
        PYTHON=str(ROOT / ".venv/bin/python"),
        PATH=str(ROOT / ".venv/bin") + os.pathsep + env["PATH"],
        PYTHONUNBUFFERED="1",
        PUSHT_CONFIG="rtx4060_laptop",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    command = [
        "./scripts/train.sh",
        "experiments/rtx4060-seed42",
        "training.seed=42",
        "dataloader.num_workers=2",
        "val_dataloader.num_workers=2",
    ]
    with (REPORT / "console.log").open("x") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        status = {
            "state": "running",
            "started_at": timestamp(),
            "supervisor_pid": os.getpid(),
            "training_launcher_pid": process.pid,
            "command": command,
            "profile": "rtx4060_laptop",
            "epochs": 300,
            "seed": 42,
            "run_dir": str(RUN.relative_to(ROOT)),
            "console_log": str((REPORT / "console.log").relative_to(ROOT)),
        }
        save_status(status)
        returncode = process.wait()
    status.update(
        state="completed" if returncode == 0 else "failed",
        exit_code=returncode,
        finished_at=timestamp(),
    )
    save_status(status)
    raise SystemExit(returncode)
