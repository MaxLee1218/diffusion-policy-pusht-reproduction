"""Analyze a paired PushT horizon sweep from its archived protocol and episodes."""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


SCENARIOS = ("clean", "disturbed")
METRICS = ("peak_score", "success_rate", "inference_seconds")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2026


def read_run(run):
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    horizons = sorted(protocol["action_steps"])
    start = protocol["test_start_seed"]
    seeds = range(start, start + protocol["n_test"])
    expected = set(product(SCENARIOS, horizons, seeds))
    episodes = {}
    with (run / "episodes.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            scenario = row["scenario"]
            horizon, seed = int(row["action_steps"]), int(row["test_seed"])
            key = scenario, horizon, seed
            if key not in expected or key in episodes:
                raise ValueError(f"Unexpected or duplicate episode: {key}")
            if int(row["sampling_seed"]) != protocol["sampling_seed"] + seed - start:
                raise ValueError(f"Sampling seed does not match the protocol: {key}")
            episodes[key] = row
    if episodes.keys() != expected:
        raise ValueError(f"Incomplete run: {len(expected - episodes.keys())} missing episodes")
    return horizons, seeds, episodes


def estimates(values, draws):
    point = values.mean(axis=0)
    resampled = values[draws].mean(axis=1)
    low, high = np.quantile(resampled, [0.025, 0.975], axis=0)
    return {
        field: float(value)
        for metric, interval in zip(METRICS, zip(point, low, high))
        for field, value in zip((metric, f"{metric}_ci_low", f"{metric}_ci_high"), interval)
    }


def analyze(horizons, seeds, episodes):
    # Every replicate samples whole seed blocks, keeping all eight settings paired.
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(seeds), size=(BOOTSTRAP_DRAWS, len(seeds))
    )
    conditions, contrasts = [], []
    for horizon in horizons:
        values = {}
        for scenario in SCENARIOS:
            rows = [episodes[scenario, horizon, seed] for seed in seeds]
            observed = np.array([
                [float(row["max_normalized_coverage_score"]), row["success"] == "True",
                 float(row["total_inference_ms"]) / 1000]
                for row in rows
            ])
            values[scenario] = observed
            exposed = [row for row in rows if row["disturbance_applied"] == "True"]
            recovered = [row for row in exposed if row["post_disturbance_success"] == "True"]
            conditions.append({
                "scenario": scenario,
                "action_steps": horizon,
                "episodes": len(rows),
                "successes": int(observed[:, 1].sum()),
                "mean_inference_calls": np.mean([int(row["inference_calls"]) for row in rows]),
                "disturbance_exposed": len(exposed),
                "post_disturbance_successes": len(recovered),
                "post_disturbance_success_rate": len(recovered) / len(exposed) if exposed else None,
                **estimates(observed, draws),
            })
        contrasts.append({
            "contrast": "disturbed_minus_clean",
            "action_steps": horizon,
            "seed_pairs": len(seeds),
            **estimates(values["disturbed"] - values["clean"], draws),
        })
    return conditions, contrasts


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(horizons, seed_count, conditions, contrasts, path):
    ink, muted = "#16283c", "#4b6073"
    plt.rcParams.update({
        "font.size": 11,
        "text.color": ink,
        "axes.labelcolor": ink,
        "xtick.color": muted,
        "ytick.color": muted,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#c8d4df",
    })
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.1), facecolor="white")
    fig.subplots_adjust(top=0.76, bottom=0.14, left=0.10, right=0.97, hspace=0.48, wspace=0.28)
    fig.text(0.10, 0.942, "Replanning in simulated PushT", fontsize=23, weight="bold")
    fig.text(0.10, 0.899, f"{seed_count} paired initial conditions · one trained policy",
             fontsize=12, color=muted)

    positions = np.arange(len(horizons))

    def points(ax, rows, metric, x, color, marker, label=None, scale=1):
        center = np.array([row[metric] for row in rows]) * scale
        low = np.array([row[f"{metric}_ci_low"] for row in rows]) * scale
        high = np.array([row[f"{metric}_ci_high"] for row in rows]) * scale
        ax.vlines(x, low, high, color=color, linewidth=1.8, zorder=3)
        return ax.scatter(x, center, s=48, color=color, marker=marker, label=label, zorder=4)

    legend = []
    for ax, panel, title, metric, label, scale in zip(
        axes.flat, "ABCD",
        ("Peak score", "Task success", "Policy inference cost", "Change in peak score"),
        (*METRICS, "peak_score"),
        ("Mean score (0–1)", "Successful episodes (%)", "Mean seconds per episode",
         "Disturbed minus clean"),
        (1, 100, 1, 1),
    ):
        ax.set_title(f"{panel}  {title}", loc="left", weight="bold", pad=10)
        ax.set_ylabel(label)
        ax.set_xticks(positions, horizons)
        ax.set_xlim(-0.48, len(horizons) - 0.52)
        ax.tick_params(axis="both", length=0, pad=6)
        ax.grid(axis="y", color="#e3eaf1", linewidth=0.85)
        ax.set_axisbelow(True)
        if panel == "D":
            ax.axhline(0, color="#73879a", linewidth=1, linestyle="--")
            points(ax, contrasts, metric, positions, "#536590", "D")
        else:
            for scenario, offset, color, marker in (
                ("clean", -0.12, "#0b6b78", "o"), ("disturbed", 0.12, "#a44719", "s")
            ):
                rows = [row for row in conditions if row["scenario"] == scenario]
                artist = points(ax, rows, metric, positions + offset, color, marker,
                                scenario.capitalize(), scale)
                if panel == "A":
                    legend.append(artist)
            if panel in "BC":
                ax.set_ylim(bottom=0)
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(0.093, 0.844),
               ncol=2, frameon=False, columnspacing=1.5)
    fig.supxlabel("Actions executed before replanning", y=0.074, fontsize=11)
    fig.text(0.10, 0.026, "Dots are means; lines are pointwise 95% bootstrap intervals over seed blocks.",
             fontsize=9, color=muted)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", default=Path("results/horizons-test"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    horizons, seeds, episodes = read_run(args.run)
    conditions, contrasts = analyze(horizons, seeds, episodes)
    output = args.output_dir or args.run / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "condition_estimates.csv", conditions)
    write_csv(output / "paired_disturbance_effects.csv", contrasts)
    plot(horizons, len(seeds), conditions, contrasts, output / "horizon_tradeoffs.png")
    print(f"Analyzed {len(episodes)} episodes ({len(seeds)} paired seed blocks): {output}")


if __name__ == "__main__":
    main()
