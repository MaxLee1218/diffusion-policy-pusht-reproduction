# RTX 4060 seed-42 experiment archive

This archive preserves the run performed on 2026-11-11 through 2026-11-12 (Asia/Shanghai), including the preceding GPU smoke test and short training validation. The 300-epoch experiment was stopped by request after 232 complete epochs, during zero-based epoch 232. All metrics below come from the saved training logs; no new training or full-length evaluation was performed for publication.

The best saved EMA checkpoint is epoch **100**, with `test/mean_score = 0.791296` across 50 test initial conditions. The latest saved checkpoint is epoch 200. Later in-memory training progress was not checkpointed before stopping. Recorded start/finish timestamps span 8.75 hours.

## Configuration and metric

The model uses U-Net widths `[128, 256, 512]`, a ResNet-18 image encoder, batch size 8, AdamW, EMA, 100 diffusion inference steps, training seed 42, and two workers per data loader. Evaluation uses four parallel environments, six training initial conditions, and 50 test seeds (`100000` through `100049`), with up to 300 simulation steps per episode. The GPU is an RTX 4060 Laptop with 8 GiB VRAM; PyTorch is 2.4.1+cu121.

Each episode score is its maximum `clip(goal_overlap_fraction / 0.95, 0, 1)` over the trajectory. `test/mean_score` averages these scores across 50 test episodes. It is not a binary success rate. At the best checkpoint, 11 of 50 episode scores reached 1.0. This reduced model and single training seed are not a full reproduction of the published baseline. Checkpoint selection used the same test initial conditions.

| Saved epoch | Test mean score | Test episodes with score 1.0 | Validation loss |
| --- | ---: | ---: | ---: |
| 0 | 0.138479 | 0/50 | 0.076762 |
| 50 | 0.733019 | 4/50 | 0.046410 |
| **100** | 0.791296 | 11/50 | 0.054651 |
| 150 | 0.775580 | 9/50 | 0.072664 |
| 200 | 0.777166 | 17/50 | 0.101454 |

## Data and provenance

- [Machine-readable summary](summary.json), [232 completed-epoch metrics](epoch_metrics.csv), and [250 individual test episode scores](test_episode_scores.csv).
- [Complete training metrics, 312,437 rows](../../experiments/rtx4060-seed42/logs.json.txt), [effective configuration](../../experiments/rtx4060-seed42/.hydra/config.yaml), and [Hydra overrides](../../experiments/rtx4060-seed42/.hydra/overrides.yaml).
- [Full console output](console.log), [final status](status.json), [stop request](stop-request.json), and [background launcher](launch.py). Exit code 143 reflects the requested termination, not an unhandled training failure.
- [GPU smoke results](../local-validation/gpu-smoke-rtx4060-b8.json), [dependency versions](../local-validation/installed-packages.txt), and [official-replay pytest log](../local-validation/pytest.log).
- [Short-run summary](../local-validation/summary.json), [short-run effective configuration](../../experiments/rtx4060-short-seed42/.hydra/config.yaml), [short training log](../local-validation/short-train.log), and [independent short-checkpoint evaluation](../rtx4060-short-seed42-eval/eval_log.json).
- [File manifest with SHA-256 checksums](manifest.json), [video metadata](videos.json), and [local checkpoint inventory](checkpoints.json).

## Recorded videos

All **33 MP4 files** are committed to the repository (1,166,853 bytes total). The longer run accounts for 30 videos and 1,142,030 bytes. Each periodic evaluation records two training and four test episodes; all 50 test episodes contribute to the score even though only four are recorded. Videos are 96 x 96 at 10 FPS and have been decoded to verify integrity.

Video filenames are preserved from the upstream recorder. Epoch attribution was reconstructed from file recording times relative to checkpoint saves, and the index records that method. The archive does not claim an individual test-seed mapping for the randomly named clips.

| Evaluation epoch | Six recorded episodes |
| --- | --- |
| 0 | [2a9n32b2](../../experiments/rtx4060-seed42/media/2a9n32b2.mp4) / [bdzw1cij](../../experiments/rtx4060-seed42/media/bdzw1cij.mp4) / [i2icm6pq](../../experiments/rtx4060-seed42/media/i2icm6pq.mp4) / [log1ycur](../../experiments/rtx4060-seed42/media/log1ycur.mp4) / [mm14bn5m](../../experiments/rtx4060-seed42/media/mm14bn5m.mp4) / [mvu3sswu](../../experiments/rtx4060-seed42/media/mvu3sswu.mp4) |
| 50 | [3g3nmand](../../experiments/rtx4060-seed42/media/3g3nmand.mp4) / [4wn2kjv6](../../experiments/rtx4060-seed42/media/4wn2kjv6.mp4) / [awg5fqw1](../../experiments/rtx4060-seed42/media/awg5fqw1.mp4) / [glmh34ph](../../experiments/rtx4060-seed42/media/glmh34ph.mp4) / [hw4myo62](../../experiments/rtx4060-seed42/media/hw4myo62.mp4) / [iesfq51s](../../experiments/rtx4060-seed42/media/iesfq51s.mp4) |
| **100 (best)** | [1jybsvfg](../../experiments/rtx4060-seed42/media/1jybsvfg.mp4) / [540omh6i](../../experiments/rtx4060-seed42/media/540omh6i.mp4) / [cey90t6g](../../experiments/rtx4060-seed42/media/cey90t6g.mp4) / [errtjb9p](../../experiments/rtx4060-seed42/media/errtjb9p.mp4) / [lrnohlrm](../../experiments/rtx4060-seed42/media/lrnohlrm.mp4) / [ufj1mrw0](../../experiments/rtx4060-seed42/media/ufj1mrw0.mp4) |
| 150 | [62ljwnhq](../../experiments/rtx4060-seed42/media/62ljwnhq.mp4) / [a77n4iag](../../experiments/rtx4060-seed42/media/a77n4iag.mp4) / [h59zdwgq](../../experiments/rtx4060-seed42/media/h59zdwgq.mp4) / [p9r5oa7h](../../experiments/rtx4060-seed42/media/p9r5oa7h.mp4) / [t27h7ne1](../../experiments/rtx4060-seed42/media/t27h7ne1.mp4) / [tb9cm1b0](../../experiments/rtx4060-seed42/media/tb9cm1b0.mp4) |
| 200 | [5bsaje8w](../../experiments/rtx4060-seed42/media/5bsaje8w.mp4) / [9a7mrvqu](../../experiments/rtx4060-seed42/media/9a7mrvqu.mp4) / [bipknppy](../../experiments/rtx4060-seed42/media/bipknppy.mp4) / [d0iwvkso](../../experiments/rtx4060-seed42/media/d0iwvkso.mp4) / [dglnhgpn](../../experiments/rtx4060-seed42/media/dglnhgpn.mp4) / [f4qmlyss](../../experiments/rtx4060-seed42/media/f4qmlyss.mp4) |

The remaining three clips come from short training and its independent checkpoint evaluation:

| Source | Video | Duration |
| --- | --- | ---: |
| rtx4060-short-seed42, epoch 0 | [0nfowcxr.mp4](../../experiments/rtx4060-short-seed42/media/0nfowcxr.mp4) | 3.2 s |
| rtx4060-short-seed42, epoch 1 | [z94kuoal.mp4](../../experiments/rtx4060-short-seed42/media/z94kuoal.mp4) | 3.2 s |
| rtx4060-short-seed42-eval, epoch 1 | [kmtyusby.mp4](../../results/rtx4060-short-seed42-eval/media/kmtyusby.mp4) | 3.2 s |

## Local model checkpoints

All **nine** saved model/optimizer/EMA checkpoints remain on the training machine and are **not uploaded to GitHub**, as requested. Their total size is 4,908,166,758 bytes. The [checkpoint inventory](checkpoints.json) records original local paths, epochs, sizes, and SHA-256 hashes; [checksums](checkpoints.sha256) can be verified from the repository root on that machine:

```bash
sha256sum -c results/rtx4060-seed42/checkpoints.sha256
```

The best longer-run checkpoint is `experiments/rtx4060-seed42/checkpoints/epoch=0100-test_mean_score=0.791.ckpt`. The longer-run `latest.ckpt` contains epoch-200 state; it does not include the unsaved progress up to epoch 232. Checkpoint files are excluded by `.gitignore` and are absent from a fresh clone. The [reproduction guide](../../docs/reproduction.md#baseline-training-and-resume) describes the upstream resume counter and RNG limitations.
