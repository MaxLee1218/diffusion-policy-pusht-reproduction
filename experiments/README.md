# Experiments

Each named run directory contains training metrics in `logs.json.txt`, Hydra configuration, and checkpoints. New runs require a fresh directory. Resume uses an existing run directory and its `checkpoints/latest.ckpt`. Run outputs are ignored by Git unless explicitly selected for archival.

The selected `rtx4060-seed42` and `rtx4060-short-seed42` runs are archived with their logs, Hydra configuration, and videos. Their checkpoints and transient W&B state remain excluded from Git. Model checkpoints are retained locally, as requested; see the [results index](../results/rtx4060-seed42/README.md) for the local checkpoint inventory and checksums.
