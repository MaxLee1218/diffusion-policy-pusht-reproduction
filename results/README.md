# Results

Each evaluation writes `eval_log.json` and any rollout videos to a new directory. The GPU smoke test writes a JSON file containing CUDA compatibility, memory, and timing measurements. Keep these outputs local. When reporting a benchmark, record the training seed, configuration, checkpoint, and `test/mean_score` across the 50 published test seeds.
