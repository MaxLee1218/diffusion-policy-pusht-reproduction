"""Supervise the independent 50-seed horizon evaluation and save exit status."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/horizons-test"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_status(status):
    temporary = CONTROL / "status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(CONTROL / "status.json")


def main():
    os.chdir(ROOT)
    command = [
        "./scripts/evaluate.sh", "horizons",
        "experiments/rtx4060-seed42/checkpoints/epoch=0100-test_mean_score=0.791.ckpt",
        "results/horizons-test", "--device", "cuda:0",
        "--action-steps", "2", "4", "8", "12",
        "--test-start-seed", "300000", "--n-test", "50",
        "--sampling-seed", "42", "--max-steps", "300",
        "--disturbance-step", "50", "--disturbance-delta-v", "120",
        "--record-first", "2",
    ]
    status = {
        "state": "starting", "started_at": timestamp(),
        "supervisor_pid": os.getpid(), "command": command,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "output_dir": str(OUTPUT.relative_to(ROOT)),
        "console_log": str((CONTROL / "console.log").relative_to(ROOT)),
        "expected_episodes": 400, "expected_conditions": 8,
        "environment_seeds": [300000, 300049],
        "sampling_seeds": [42, 91],
        "tmux_socket": "pusht-experiments",
        "tmux_session": "pusht-horizons-test",
        "monitor_session": "pusht-horizons-monitor",
    }
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite existing results: {OUTPUT}")
    save_status(status)
    env = os.environ.copy()
    env.update(
        PYTHON=str(ROOT / ".venv/bin/python"),
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
    )
    try:
        with (CONTROL / "console.log").open("x") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
            )
            status.update(state="running", evaluation_pid=process.pid)
            save_status(status)
            returncode = process.wait()
        status.update(
            state="completed" if returncode == 0 else "failed",
            exit_code=returncode, finished_at=timestamp(),
        )
    except BaseException as exc:
        status.update(state="failed", error=repr(exc), finished_at=timestamp())
        save_status(status)
        raise
    save_status(status)
    print(json.dumps(status, indent=2), flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
