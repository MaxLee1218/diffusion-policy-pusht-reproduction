"""Native live training monitor. Closing it leaves background training running."""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


BACKGROUND = "#0e1625"
PANEL = "#172236"
TEXT = "#e6edf7"
MUTED = "#93a6c2"
BLUE = "#56b4ff"
GREEN = "#66dfb0"
ORANGE = "#ffbf69"


class MetricLog:
    """Incrementally consume complete JSON lines, including a partially written tail."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b""
        self.steps = 0
        self.losses = deque(maxlen=6000)
        self.epoch_metrics = []
        self.epoch_batches = {}
        self.last = {}

    def refresh(self):
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.offset:
            self.__init__(self.path)
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            payload = handle.read()
            self.offset = handle.tell()
        lines = (self.pending+payload).split(b"\n")
        self.pending = lines.pop()
        for line in lines:
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or "train_loss" not in row:
                    continue
                loss = float(row["train_loss"])
                if not math.isfinite(loss):
                    row["nonfinite_loss"] = True
                self.steps += 1
                epoch = int(row["epoch"])
                self.epoch_batches[epoch] = self.epoch_batches.get(epoch, 0)+1
                self.last = row
                if "val_loss" in row:
                    self.epoch_metrics.append(row)
                else:
                    self.losses.append((self.steps, loss))
            except (ValueError, TypeError, KeyError):
                continue


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Monitor:
    def __init__(self, root, run_dir, output):
        self.root, self.run_dir, self.output = root, run_dir, output
        self.log = MetricLog(run_dir / "logs.json.txt")
        self.launch = load_json(output / "launch.json")
        self.stop = threading.Event()
        self.gpu = {}
        self.gpu_lock = threading.Lock()
        self.root.title("B601 实时训练监控 | Live Training Monitor")
        self.root.geometry("1180x820+100+70")
        self.root.minsize(1000, 730)
        self.root.configure(bg=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("B601.Horizontal.TProgressbar", background=BLUE, troughcolor=PANEL,
                        bordercolor=PANEL, lightcolor=BLUE, darkcolor=BLUE)

        header = tk.Frame(root, bg=BACKGROUND)
        header.pack(fill="x", padx=24, pady=(20, 10))
        tk.Label(header, text="B601  /  TRAINING", fg=TEXT, bg=BACKGROUND,
                 font=("DejaVu Sans", 20, "bold")).pack(side="left")
        self.state_label = tk.Label(header, text="CONNECTING", fg=BLUE, bg=BACKGROUND,
                                    font=("DejaVu Sans", 11, "bold"))
        self.state_label.pack(side="right")
        tk.Label(root, text="30 training episodes  /  4 held-out episodes   ·   320 × 240 RGB + 7 joints   ·   FP32 + EMA",
                 bg=BACKGROUND, fg=MUTED, font=("DejaVu Sans", 10)).pack(anchor="w", padx=25)
        self.progress = ttk.Progressbar(root, style="B601.Horizontal.TProgressbar", maximum=100)
        self.progress.pack(fill="x", padx=25, pady=(15, 4))
        self.progress_text = tk.Label(root, text="Loading metrics…", fg=MUTED, bg=BACKGROUND,
                                      font=("DejaVu Sans", 10))
        self.progress_text.pack(anchor="w", padx=25)

        cards = tk.Frame(root, bg=BACKGROUND)
        cards.pack(fill="x", padx=20, pady=14)
        self.values, self.details = {}, {}
        for index, (key, title) in enumerate([
            ("epoch", "FULL EPOCH"), ("loss", "LATEST TRAIN LOSS"),
            ("val", "BEST VALIDATION LOSS"), ("gpu", "GPU MEMORY"),
            ("elapsed", "ELAPSED"), ("eta", "ESTIMATED REMAINING"),
        ]):
            cards.columnconfigure(index, weight=1, uniform="card")
            card = tk.Frame(cards, bg=PANEL, padx=12, pady=12)
            card.grid(row=0, column=index, sticky="nsew", padx=5)
            tk.Label(card, text=title, bg=PANEL, fg=MUTED,
                     font=("DejaVu Sans", 8, "bold")).pack(anchor="w")
            variable, detail = tk.StringVar(value="—"), tk.StringVar(value=" ")
            self.values[key], self.details[key] = variable, detail
            tk.Label(card, textvariable=variable, bg=PANEL, fg=TEXT,
                     font=("DejaVu Sans", 17, "bold")).pack(anchor="w", pady=(8, 4))
            tk.Label(card, textvariable=detail, bg=PANEL, fg=MUTED,
                     font=("DejaVu Sans", 8)).pack(anchor="w")

        figure = Figure(figsize=(11, 3.8), dpi=100, facecolor=BACKGROUND, layout="constrained")
        self.train_ax, self.val_ax = figure.subplots(1, 2)
        for axis in (self.train_ax, self.val_ax):
            axis.set_facecolor(PANEL)
            axis.tick_params(colors=MUTED, labelsize=8)
            axis.xaxis.label.set_color(MUTED)
            axis.yaxis.label.set_color(MUTED)
            axis.title.set_color(TEXT)
            axis.grid(alpha=.15, color=MUTED)
            for spine in axis.spines.values():
                spine.set_color("#31415a")
        self.train_ax.set(title="Training loss · latest 6,000 updates", xlabel="New optimizer updates", ylabel="Noise MSE")
        self.train_ax.set_yscale("log")
        self.raw_line, = self.train_ax.plot([], [], color=BLUE, alpha=.25, lw=.7, label="Batch")
        self.smooth_line, = self.train_ax.plot([], [], color=BLUE, lw=1.8, label="100-update mean")
        self.val_ax.set(title="Full held-out validation", xlabel="Full training epoch", ylabel="Noise MSE")
        self.val_line, = self.val_ax.plot([], [], color=ORANGE, marker="o", ms=3, lw=1.5, label="Main model")
        self.ema_point, = self.val_ax.plot([], [], color=GREEN, marker="*", ms=12, lw=0, label="Final EMA")
        for axis in (self.train_ax, self.val_ax):
            axis.legend(facecolor=PANEL, edgecolor=PANEL, labelcolor=TEXT, fontsize=8)
        self.canvas = FigureCanvasTkAgg(figure, master=root)
        self.canvas.get_tk_widget().configure(bg=BACKGROUND, highlightthickness=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=18)

        self.message = tk.StringVar(value="Waiting for the training log.")
        tk.Label(root, textvariable=self.message, bg=BACKGROUND, fg=GREEN,
                 font=("DejaVu Sans", 10), anchor="w").pack(fill="x", padx=25, pady=(10, 4))
        footer = tk.Frame(root, bg=BACKGROUND)
        footer.pack(fill="x", padx=25, pady=(3, 16))
        tk.Label(footer, text="Refresh: 2 s   ·   Closing this window leaves background training running.",
                 bg=BACKGROUND, fg=MUTED, font=("DejaVu Sans", 9)).pack(side="left")
        tk.Button(footer, text="Copy checkpoint folder", command=self.copy_checkpoint_path,
                  bg=PANEL, fg=TEXT, activebackground="#243650", activeforeground=TEXT,
                  relief="flat", padx=12, pady=5).pack(side="right")

        threading.Thread(target=self.gpu_worker, daemon=True).start()
        self.root.after(100, self.refresh)
        self.root.lift()

    def copy_checkpoint_path(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(str((self.run_dir / "checkpoints").resolve()))

    def gpu_worker(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                values = [float(value.strip()) for value in result.stdout.splitlines()[0].split(",")]
                with self.gpu_lock:
                    self.gpu = dict(zip(("utilization", "used_mib", "total_mib", "temperature", "watts"), values))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            self.stop.wait(6)

    def refresh(self):
        try:
            self.update()
        except Exception as error:
            self.message.set(f"Monitor refresh: {error}")
        if not self.stop.is_set():
            self.root.after(2000, self.refresh)

    def update(self):
        self.log.refresh()
        status = load_json(self.output / "training_status.json")
        resume = load_json(self.run_dir / "resume_state.json")
        planned = int(self.launch.get("planned_full_epochs", 100))
        total = int(self.launch.get("planned_additional_optimizer_updates", 208600))
        batches = int(self.launch.get("train_batches_per_epoch", 2086))
        first_epoch = int(resume.get("next_epoch", 3))
        raw_epoch = int(self.log.last.get("epoch", first_epoch))
        formal_epoch = max(1, raw_epoch-first_epoch+1)
        complete = min(100., self.log.steps/max(total, 1)*100)
        state = status.get("status", "starting")
        alive = False
        try:
            os.kill(int(status["supervisor_pid"]), 0)
            alive = True
        except (KeyError, OSError, ValueError):
            pass
        if state not in ("complete", "failed") and not alive and status:
            state = "process_exited"
        labels = {"training": "● TRAINING", "validating_checkpoint": "● FINAL CHECKS",
                  "complete": "● COMPLETE", "failed": "● FAILED", "process_exited": "● PROCESS EXITED"}
        color = GREEN if state in ("training", "complete") else ORANGE
        if self.log.last.get("nonfinite_loss"):
            state, color = "nonfinite_loss", "#ff7b8c"
        self.state_label.configure(text=labels.get(state, state.upper()), fg=color)
        self.progress["value"] = complete
        self.progress_text.configure(text=f"{complete:.2f}%   ·   {self.log.steps:,} / {total:,} new updates   ·   "
                                          f"current epoch batch {self.log.epoch_batches.get(raw_epoch, 0):,} / {batches:,}")
        self.values["epoch"].set(f"{min(formal_epoch, planned)} / {planned}")
        self.details["epoch"].set(f"{len(self.log.epoch_metrics)} fully validated")
        loss = self.log.last.get("train_loss")
        self.values["loss"].set(f"{loss:.5f}" if loss is not None else "—")
        self.details["loss"].set(f"LR {self.log.last.get('lr', 0):.2e}")
        best = min((row["val_loss"] for row in self.log.epoch_metrics), default=None)
        self.values["val"].set(f"{best:.5f}" if best is not None else "Pending")
        self.details["val"].set("4 episodes / 2,347 windows")
        with self.gpu_lock:
            gpu = self.gpu.copy()
        if gpu:
            self.values["gpu"].set(f"{gpu['used_mib']/1024:.2f} / {gpu['total_mib']/1024:.1f} GB")
            self.details["gpu"].set(f"{gpu['utilization']:.0f}% load  ·  {gpu['temperature']:.0f}°C  ·  {gpu['watts']:.0f} W")
        started = self.launch.get("launched_at", status.get("started_at"))
        elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(started)).total_seconds() if started else 0
        self.values["elapsed"].set(duration(elapsed))
        rate = self.log.steps/max(elapsed, 1)
        self.details["elapsed"].set(f"{rate:.2f} updates/s incl. validation")
        remaining = max(0, total-self.log.steps)/rate if self.log.steps > 10 else None
        self.values["eta"].set(duration(remaining) if remaining is not None else "Estimating")
        self.details["eta"].set("Based on observed throughput")

        if self.log.losses:
            points = np.asarray(self.log.losses, dtype=float)
            stride = max(1, len(points)//900)
            self.raw_line.set_data(points[::stride, 0], np.maximum(points[::stride, 1], 1e-8))
            window = min(100, len(points))
            smoothed = np.convolve(points[:, 1], np.ones(window)/window, mode="valid")
            self.smooth_line.set_data(points[window-1:, 0], np.maximum(smoothed, 1e-8))
            self.train_ax.relim()
            self.train_ax.autoscale_view()
        if self.log.epoch_metrics:
            self.val_line.set_data([row["epoch"]-first_epoch+1 for row in self.log.epoch_metrics],
                                   [row["val_loss"] for row in self.log.epoch_metrics])
            self.val_ax.relim()
            self.val_ax.autoscale_view()
        final = load_json(self.output / "validation.json") if state == "complete" else {}
        if final:
            self.ema_point.set_data([planned], [final["full_heldout_ema_noise_mse"]])
            self.val_ax.relim()
            self.val_ax.autoscale_view()
        self.canvas.draw_idle()
        stamp = datetime.now().strftime("%H:%M:%S")
        message = f"Live at {stamp}  ·  {self.run_dir}  ·  checkpoint saved after every full epoch"
        if state in ("failed", "process_exited", "nonfinite_loss"):
            message = f"{state.upper()}: {status.get('error', 'Inspect train.log and supervisor.log.')}"
        elif state == "complete":
            message = f"Training and checkpoint checks completed. Final EMA loss: {final.get('full_heldout_ema_noise_mse', float('nan')):.5f}"
        self.message.set(message)
        snapshot = {"pid": os.getpid(), "window_title": self.root.title(),
                    "visible": bool(self.root.winfo_viewable()), "updated_at": datetime.now(timezone.utc).isoformat(),
                    "training_state": state, "training_supervisor_alive": alive,
                    "formal_epoch": formal_epoch, "new_updates": self.log.steps,
                    "completion_percent": complete, "latest_train_loss": loss, "best_val_loss": best,
                    "estimated_remaining_seconds": remaining, "gpu": gpu,
                    "geometry": [self.root.winfo_rootx(), self.root.winfo_rooty(),
                                 self.root.winfo_width(), self.root.winfo_height()]}
        temporary = self.output / "monitor_status.json.tmp"
        temporary.write_text(json.dumps(snapshot, indent=2)+"\n")
        temporary.replace(self.output / "monitor_status.json")

    def close(self):
        self.stop.set()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("experiments/b601-seed42"))
    parser.add_argument("--output", type=Path, default=Path("results/b601-seed42"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    Monitor(root, args.run_dir, args.output)
    root.mainloop()


if __name__ == "__main__":
    main()
