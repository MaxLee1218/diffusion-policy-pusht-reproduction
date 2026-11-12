"""Read-only terminal dashboard; closing it does not stop the evaluation."""

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/horizons-test"


def snapshot():
    try:
        status = json.loads((CONTROL / "status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        status = {"state": "waiting for launch"}
    rows = []
    try:
        for row in csv.DictReader(io.StringIO((OUTPUT / "episodes.csv").read_text())):
            if row.get("success") in ("True", "False") and row.get("mean_inference_ms"):
                float(row["mean_inference_ms"])
                rows.append(row)
    except (FileNotFoundError, ValueError, csv.Error):
        pass
    try:
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader",
        ], text=True, stderr=subprocess.STDOUT, timeout=3).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        gpu = f"GPU query unavailable: {exc}"
    elapsed = "--"
    if "started_at" in status:
        end = datetime.fromisoformat(status["finished_at"]) if "finished_at" in status else datetime.now(timezone.utc)
        seconds = int((end - datetime.fromisoformat(status["started_at"])).total_seconds())
        elapsed = f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    lines = [
        "PushT 50-seed evaluation monitor",
        f"State: {status['state']}   Completed: {len(rows)}/400 ({len(rows) / 4:.1f}%)   Elapsed: {elapsed}",
        f"Evaluation PID: {status.get('evaluation_pid', '--')}   Supervisor PID: {status.get('supervisor_pid', '--')}",
        "Environment seeds: 300000-300049 | Sampling seeds: 42-91",
        "Epoch 100 EMA | 100 diffusion steps | +X disturbance at step 50, delta-v=120",
        "", "GPU: " + gpu, "",
        "Condition    Actions  Episodes  Mean score  Successes  Disturbed",
    ]
    for scenario in ("clean", "disturbed"):
        for horizon in (2, 4, 8, 12):
            group = [r for r in rows if r['scenario'] == scenario and int(r['action_steps']) == horizon]
            score = f"{mean(float(r['max_normalized_coverage_score']) for r in group):.4f}" if group else "--"
            successes = sum(r['success'] == 'True' for r in group)
            exposed = sum(r['disturbance_applied'] == 'True' for r in group)
            lines.append(f"{scenario:11}  {horizon:>5}  {len(group):>5}/50  {score:>10}  {successes:>4}/{len(group):<3}  {exposed:>9}")
    if rows:
        row = rows[-1]
        lines[5] = f"Last completed: {row['scenario']}, {row['action_steps']} actions, seed {row['test_seed']}, score {float(row['max_normalized_coverage_score']):.4f}"
    if "exit_code" in status:
        lines.append(f"Exit code: {status['exit_code']}")
    if "error" in status:
        lines.append(f"Error: {status['error']}")
    lines += ["", "Recent log:"]
    try:
        with (CONTROL / "console.log").open("rb") as file:
            file.seek(0, 2)
            file.seek(max(0, file.tell() - 6000))
            tail = file.read().decode("utf-8", errors="replace")
        lines.extend(tail.replace("\r", "\n").splitlines()[-5:])
    except FileNotFoundError:
        lines.append("Waiting for evaluator output...")
    lines += [
        "", f"Results: {OUTPUT}", f"Log: {CONTROL / 'console.log'}",
        "Refresh: 5s | Ctrl+B then D: detach | Ctrl+C: stop monitor only",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            view = snapshot()
            if not args.once:
                size = shutil.get_terminal_size(fallback=(120, 30))
                lines = [line[:max(1, size.columns - 1)] for line in view.splitlines()]
                view = "\n".join(lines[:max(1, size.lines - 1)])
            print(("" if args.once else "\033[2J\033[H") + view, flush=True)
            if args.once:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass
