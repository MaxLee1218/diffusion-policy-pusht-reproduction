"""Plot the archived RTX 4060 baseline and compare checkpoints by test seed."""

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_baseline(metrics, rollouts, best_epoch, path):
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "figure.facecolor": "white",
    })
    fig, (loss_ax, rollout_ax) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.10, right=0.97, hspace=0.30)
    fig.text(0.10, 0.955, "PushT image-policy baseline", fontsize=17, weight="bold")
    subtitle = f"RTX 4060 Laptop · training seed 42 · {len(metrics)} complete epochs"
    fig.text(0.10, 0.905, subtitle,
             color="#475569")

    epochs = [int(row["epoch"]) for row in metrics]
    loss_ax.plot(epochs, [float(row["train_loss"]) for row in metrics],
                 color="#146c80", label="Training")
    loss_ax.plot(epochs, [float(row["val_loss"]) for row in metrics],
                 color="#b54f26", label="Validation")
    loss_ax.set_yscale("log")
    loss_ax.set_ylabel("Diffusion loss (log scale)")
    loss_ax.legend(frameon=False, loc="upper right", ncol=2)

    eval_epochs = [row["epoch"] for row in rollouts]
    rollout_ax.scatter(eval_epochs, [row["mean_score"] for row in rollouts],
                       color="#146c80", marker="o", s=34, label="Mean rollout score", zorder=3)
    rollout_ax.scatter(eval_epochs, [row["score_equal_one_fraction"] for row in rollouts],
                       color="#b54f26", marker="s", s=27, label="Fraction at score 1.0", zorder=3)
    best = next(row for row in rollouts if row["epoch"] == best_epoch)
    rollout_ax.annotate(f'{best["mean_score"]:.3f}', (best_epoch, best["mean_score"]),
                        xytext=(6, -8), textcoords="offset points", color="#146c80",
                        va="top", weight="bold")
    rollout_ax.set(ylim=(-0.04, 1.04), ylabel="Score or fraction", xlabel="Epoch (zero-based)")
    rollout_ax.legend(frameon=False, loc="upper left", ncol=2)
    for ax in (loss_ax, rollout_ax):
        ax.axvline(best_epoch, ls="--", lw=1, color="#94a3b8")
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "results/rtx4060-seed42")
    args = parser.parse_args()
    metrics = read_csv(args.archive / "epoch_metrics.csv")
    scores = read_csv(args.archive / "test_episode_scores.csv")
    by_epoch = {}
    for row in scores:
        epoch, seed = int(row["epoch"]), int(row["test_seed"])
        episode_scores = by_epoch.setdefault(epoch, {})
        if seed in episode_scores:
            raise ValueError(f"Duplicate episode for epoch {epoch}, seed {seed}.")
        episode_scores[seed] = float(row["max_normalized_coverage_score"])

    logged_means = {int(row["epoch"]): float(row["test/mean_score"])
                    for row in metrics if row["test/mean_score"]}
    if by_epoch.keys() != logged_means.keys():
        raise ValueError("Rollout epochs differ between the two archive CSVs.")
    epochs = sorted(by_epoch)
    seeds = sorted(by_epoch[epochs[0]])
    if any(sorted(group) != seeds for group in by_epoch.values()):
        raise ValueError("Rollout epochs do not share the same test seeds.")

    rollouts = []
    for epoch in epochs:
        values = list(by_epoch[epoch].values())
        if not math.isclose(mean(values), logged_means[epoch], rel_tol=1e-10):
            raise ValueError(f"Episode scores disagree with the logged mean at epoch {epoch}.")
        full_scores = values.count(1.0)
        rollouts.append({
            "epoch": epoch,
            "episodes": len(values),
            "mean_score": mean(values),
            "median_score": median(values),
            "score_equal_one_count": full_scores,
            "score_equal_one_fraction": full_scores / len(values),
            "score_below_half_count": sum(score < 0.5 for score in values),
        })

    best_epoch = max(logged_means, key=logged_means.get)
    latest_epoch = epochs[-1]
    paired = [
        {"test_seed": seed}
        | {f"score_epoch_{epoch}": by_epoch[epoch][seed] for epoch in epochs}
        | {f"delta_epoch_{latest_epoch}_minus_{best_epoch}":
           by_epoch[latest_epoch][seed] - by_epoch[best_epoch][seed]}
        for seed in seeds
    ]
    output = args.archive / "analysis"
    output.mkdir(exist_ok=True)
    write_csv(output / "rollout_summary.csv", rollouts)
    write_csv(output / "paired_episode_scores.csv", paired)
    plot_baseline(metrics, rollouts, best_epoch, output / "baseline.png")
    print(output)


if __name__ == "__main__":
    main()
