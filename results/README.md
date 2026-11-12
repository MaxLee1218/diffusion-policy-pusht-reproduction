# Results

The checkpoint evaluator writes `eval_log.json` and rollout videos to a new directory. The action execution study writes its protocol, per-episode CSV, summary CSV, and optional videos. The GPU smoke test writes a JSON file containing CUDA compatibility, memory, and timing measurements. Selected completed experiment outputs are committed through explicit `.gitignore` exceptions. Large checkpoints remain local and are not uploaded.

[`rtx4060-validation.json`](rtx4060-validation.json) records the WSL 2 RTX 4060 Laptop smoke test and short training validation. Paths in that report refer to the original artifact locations. This short run is a functional check with four 32-step test episodes, not a converged policy or the full benchmark. When reporting a benchmark, record the training seed, configuration, checkpoint, and `test/mean_score` across the 50 published test seeds.

The [RTX 4060 seed-42 experiment archive](rtx4060-seed42/README.md) preserves the subsequent longer run, all raw training metrics, configurations, 33 videos across validation and training, per-episode evaluation results, stop status, and local checkpoint metadata/checksums. Small artifacts referenced by the earlier validation report are also committed. Model checkpoints from both runs remain local.

## RTX 4060 baseline analysis

![Training loss, validation loss and the five recorded rollout evaluations](rtx4060-seed42/analysis/baseline.png)

The [rollout summary](rtx4060-seed42/analysis/rollout_summary.csv) aggregates the 250 archived scores. The [paired scores](rtx4060-seed42/analysis/paired_episode_scores.csv) align all five evaluations by test seed and give the epoch 200 minus epoch 100 score for each initial condition.

Epoch 100 has the highest mean score (0.7913). At epoch 200, 27 seeds improved, 16 declined, and 7 tied. The number of scores at 1.0 rose from 11 to 17, while the mean fell to 0.7772. Seed 100044 fell from 0.9902 to zero and seed 100024 fell from 0.7003 to 0.0536. The archive records each episode's maximum clipped normalized coverage. Exact raw coverage above the clipping threshold and time to success require a new evaluation.

Regenerate the figure and tables from the repository root in the activated environment:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/analyze_baseline.py
```

## Action execution length development sweep

The development sweep completed on 2026-11-12 with five environment seeds (`200000–200004`), four action execution lengths, and paired clean/disturbed conditions: **40 episodes** in total. It used the epoch-100 EMA checkpoint, two observation steps, a 16-step prediction horizon, 100 diffusion inference steps, and a 300-control-step episode limit. The disturbed condition applies a +X velocity impulse of 120 simulation units/s before control step 50. All 20 disturbed episodes reached that step and received the impulse.

| Actions executed before replanning | Clean mean score | Clean successes | Disturbed mean score | Disturbed successes | Paired mean score change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.9079 | 1/5 | 0.6426 | 1/5 | -0.2653 |
| 4 | 0.8533 | 0/5 | 0.7690 | 1/5 | -0.0843 |
| 8 | 0.9397 | 1/5 | 0.9885 | 2/5 | +0.0489 |
| 12 | 0.7809 | 0/5 | 0.7793 | 2/5 | -0.0016 |

The mean score averages each episode's maximum clipped normalized coverage; success requires raw coverage strictly above 0.95. Peak scores span the entire episode, including time before the disturbance. The 8-action setting has the highest mean score in both conditions in this development sample. Five seeds and one trained checkpoint do not establish a general ranking or a benefit from disturbance. The two largest paired score drops in the 2-action setting occurred on seeds 200003 and 200004. High peak scores can coexist with later deterioration: the disturbed 8-action episode for seed 200004 reached 94.718% raw coverage but ended at 49.339%.

Mean cumulative policy inference time per clean episode was approximately 129, 74, 34, and 24 seconds for execution lengths 2, 4, 8, and 12. This is synchronous simulation: physics does not advance during inference, so these rollouts do not measure the effect of real-time inference delay. Post-disturbance time to success is averaged only over successful exposed episodes, which are different subsets across execution lengths.

- [Protocol and checkpoint SHA-256](horizons-dev/protocol.json), [40 per-episode results](horizons-dev/episodes.csv), [eight-condition summary](horizons-dev/summary.csv), and [console log](horizons-dev/console.log).
- [Validation record](horizons-dev/validation.json): seed pairing, episode counts, score/success consistency, summary values, disturbance exposure, and complete decoding of all 16 videos. Each video has 96 x 96 resolution and one frame per executed control step.
- [Clean videos](horizons-dev/videos/clean/) and [disturbed videos](horizons-dev/videos/disturbed/): seeds 200000 and 200001 are recorded for every execution length; filenames include the length and seed.
- [Artifact manifest and SHA-256 checksums](horizons-dev/manifest.json). Checkpoint weights remain local.

The [study command](../README.md#action-execution-length-study) reproduces this protocol with a fresh output directory and the local checkpoint. Environment seeds are a fixed reproducible list, with sampling seeds `42–46` reset per episode and shared across conditions for each environment seed.

## Independent 50-seed action execution length evaluation

The independent evaluation ran from 2026-11-12 17:29 to 2026-11-13 00:18 China Standard Time (UTC+8), finishing with exit code 0 after 6 h 48 min. It used the same epoch-100 EMA checkpoint and development protocol, with fresh environment seeds `300000–300049` and sampling seeds `42–91` paired across all eight conditions: **400 episodes** in total. These environment seeds are separate from both checkpoint selection and development.

| Actions executed before replanning | Clean mean score | Clean successes | Disturbed mean score | Disturbed successes | Paired mean score change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.7718 | 7/50 (14%) | 0.7841 | 8/50 (16%) | +0.0124 |
| 4 | 0.7624 | 10/50 (20%) | 0.7863 | 12/50 (24%) | +0.0240 |
| 8 | 0.7378 | 10/50 (20%) | 0.7164 | 10/50 (20%) | -0.0214 |
| 12 | 0.8079 | 5/50 (10%) | 0.7822 | 12/50 (24%) | -0.0257 |

The development sample's highest mean score at 8 actions did not persist on the independent seeds. On clean trials, 12 actions had the highest mean peak score but the lowest success rate. With disturbance, 4 and 12 actions tied for the highest overall success count. These are descriptive results for one training seed and checkpoint; they do not establish a statistically reliable ranking or a benefit from disturbance. Mean score measures peak clipped normalized coverage over the entire episode, while success requires raw coverage strictly above 0.95.

Of the 200 trials assigned to the disturbed condition, 199 received the impulse. The 8-action trial for seed `300040` succeeded at step 48, before the step-50 trigger. Its overall result remains included, but it has no post-disturbance result. Consequently, the 8-action post-disturbance success rate is **9/49 (18.37%)**, versus the overall **10/50 (20%)**. All 50 trials received the impulse at each other execution length.

Mean cumulative policy inference time per clean episode was approximately 127, 61, 32, and 22 seconds for 2, 4, 8, and 12 actions. Physics pauses during inference in this synchronous simulation, so these timings do not measure the effects of real-time inference delay.

- [Protocol and checkpoint SHA-256](horizons-test/protocol.json), [400 per-episode results](horizons-test/episodes.csv), and [eight-condition summary](horizons-test/summary.csv).
- [Completed run status](horizons-test-control/status.json), [console log](horizons-test-control/console.log), [launcher](horizons-test-control/launch.py), and [terminal monitor](horizons-test-control/monitor.py). Recorded process IDs refer to the original run.
- [Validation record](horizons-test/validation.json): seed pairing, episode counts, metric and summary consistency, disturbance exposure, checkpoint checksum, and full decoding of all 16 videos.
- [Clean videos](horizons-test/videos/clean/) and [disturbed videos](horizons-test/videos/disturbed/): seeds `300000` and `300001` for every execution length.
- [Artifact manifest and SHA-256 checksums](horizons-test/manifest.json). Model checkpoints remain local.
