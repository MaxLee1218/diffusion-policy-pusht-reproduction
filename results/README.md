# Results

Each evaluation writes `eval_log.json` and any rollout videos to a new directory. The GPU smoke test writes a JSON file containing CUDA compatibility, memory, and timing measurements. Keep raw logs, videos, and checkpoints local. Small validation summaries can be committed through an explicit `.gitignore` exception.

[`rtx4060-validation.json`](rtx4060-validation.json) records the WSL 2 RTX 4060 Laptop smoke test and short training validation. Paths in the report refer to local artifacts; the artifacts themselves are ignored. This is a functional check with four 32-step test episodes, not a converged policy or the full benchmark. When reporting a benchmark, record the training seed, configuration, checkpoint, and `test/mean_score` across the 50 published test seeds.
