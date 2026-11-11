# Experiments

Each named run directory contains training metrics in `logs.json.txt`, Hydra configuration, and checkpoints. New runs require a fresh directory. Resume uses an existing run directory and its `checkpoints/latest.ckpt`. These files are ignored by Git.
