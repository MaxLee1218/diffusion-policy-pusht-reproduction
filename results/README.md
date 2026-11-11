# Results

Each evaluation writes `eval_log.json` and rollout videos to a new directory. The GPU smoke test writes a JSON file containing CUDA compatibility, memory, and timing measurements. Selected completed experiment outputs are committed through explicit `.gitignore` exceptions. Large checkpoints remain local and are not uploaded.

[`rtx4060-validation.json`](rtx4060-validation.json) records the WSL 2 RTX 4060 Laptop smoke test and short training validation. Paths in that report refer to the original artifact locations. This short run is a functional check with four 32-step test episodes, not a converged policy or the full benchmark. When reporting a benchmark, record the training seed, configuration, checkpoint, and `test/mean_score` across the 50 published test seeds.

The [RTX 4060 seed-42 experiment archive](rtx4060-seed42/README.md) preserves the subsequent longer run, all raw training metrics, configurations, 33 videos across validation and training, per-episode evaluation results, stop status, and local checkpoint metadata/checksums. Small artifacts referenced by the earlier validation report are also committed. Model checkpoints from both runs remain local.
