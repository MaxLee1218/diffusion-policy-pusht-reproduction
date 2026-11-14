# Reproduction guide

This guide covers the [PushT reproduction and evaluation project](../README.md). The upstream code is pinned at commit `5ba07ac6661db573af695b419a7947ecb704690f` in `upstream/diffusion_policy`. The upstream MIT license covers the copied baseline configuration. See [Diffusion Policy by Chi et al., RSS 2023](https://diffusion-policy.cs.columbia.edu/) and the [official implementation](https://github.com/real-stanford/diffusion_policy).

## Baseline and CPU smoke configuration

`configs/baseline.yaml` is byte-for-byte the [published image PushT CNN configuration](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml). It retains the ResNet encoder, U-Net widths `[512, 1024, 2048]`, 100 diffusion steps, batch size 64, EMA, 3,050 epochs and 50 test seeds. The dataset split seed is 42 and is independent of the training seed. The local training launcher sets the output path and resume behavior, disables W&B, and keeps Hydra in the repository root.

`configs/smoke_test.yaml` inherits the baseline and uses a CPU, U-Net widths `[32, 64, 128]`, batch size 2, two training batches, one validation batch, and two reverse diffusion steps. It uses two training episodes and one short test rollout and disables EMA and top-k checkpointing.

## From demonstrations to rollout

1. [Dataset and sampler](../upstream/diffusion_policy/diffusion_policy/dataset/pusht_image_dataset.py). PushT reads `img`, `state`, `action`, and episode boundaries from the official Zarr replay. It selects 16-step windows within an episode and repeats the boundary frames when padding. A batch contains RGB tensors `[B, 16, 3, 96, 96]` scaled to `[0, 1]`, XY agent positions `[B, 16, 2]`, and XY target actions `[B, 16, 2]`. The upstream validation sampler uses the complement of the capped training mask.
2. [Observation encoder](../upstream/diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py). The policy normalizes observations and actions. For the first two observations it crops RGB to 84 × 84, encodes each crop with the robomimic ResNet-18 spatial-softmax encoder, and concatenates the corresponding XY position. The two feature vectors are flattened into one global conditioning vector per sample. Training uses random crops and inference uses a fixed crop.
3. [Conditional 1D U-Net](../upstream/diffusion_policy/diffusion_policy/model/diffusion/conditional_unet1d.py). The network convolves over the action trajectory's time axis. Residual blocks use learned scale and bias conditioned on the diffusion timestep embedding and global observation features. The U-Net predicts a noise tensor of shape `[B, 16, 2]`.
4. [Diffusion loss](../upstream/diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py). For each sample, draw a timestep and Gaussian noise, then use the DDPM scheduler to perturb the normalized actions. The loss is the mean squared error between the predicted and sampled noise. AdamW updates the weights and the baseline maintains an EMA copy for rollout.
5. [Rollout](../upstream/diffusion_policy/diffusion_policy/env_runner/pusht_image_runner.py). From Gaussian noise, iteratively denoise 16 candidate actions conditioned on the two latest observations. Unnormalize the trajectory and execute eight actions starting at index one. Repeat with new observations. `test/mean_score` averages the maximum goal overlap reached in each test episode.

## Apple Silicon CPU validation

Use Python 3.11 and activate a fresh virtual environment.

```bash
git submodule update --init
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.0 setuptools==65.5.0 wheel==0.38.4
python -m pip install -r requirements.txt -r requirements-torch.txt
python -m pip install --no-build-isolation gym==0.21.0
python -m pip install --no-deps robomimic==0.2.0

python -m pytest -q
./scripts/prepare_data.sh
./scripts/smoke_test.sh experiments/smoke-local
./scripts/evaluate.sh experiments/smoke-local/checkpoints/latest.ckpt results/smoke-local
```

The pinned build tools handle the legacy Gym package metadata. The separate robomimic install skips its EGL probe, which the Pygame/Pymunk PushT environment never uses. Pytest uses synthetic demonstrations by default. To run the same assertions with the official replay after preparing data, use the command below. The smoke and evaluation launchers use the real dataset. Data preparation verifies the official archive's SHA-256. Pass an archive path to use an existing download.

```bash
PUSHT_REPLAY_PATH=data/pusht/pusht_cchi_v7_replay.zarr python -m pytest -q
```

The GitHub Actions CPU workflow builds the Linux CPU image and runs the same tests, data preparation, training smoke and checkpoint evaluation.

Build the Linux x86_64 CPU image locally with the same pinned dependencies. On Apple Silicon, this was validated with Colima, Rosetta, and 8 GiB of VM memory. The official replay exceeded the 4 GiB VM limit under QEMU.

```bash
docker build --platform linux/amd64 --target cpu -t pusht:cpu .
docker run --rm --platform linux/amd64 --shm-size=1g \
  -e OMP_NUM_THREADS=2 \
  -e PUSHT_REPLAY_PATH=/work/data/pusht/pusht_cchi_v7_replay.zarr \
  -v "$PWD/data:/work/data:ro" pusht:cpu python -m pytest -q
```

## Linux NVIDIA GPU validation

The CUDA image targets Linux x86_64 with PyTorch 2.4.1, torchvision 0.19.1, and CUDA 12.1. NVIDIA A100 (Ampere, compute capability 8.0) is the reference target. Use a compatible host driver and NVIDIA Container Toolkit. The script checks the actual GPU's compute capability against the installed PyTorch CUDA architecture list before training. [PyTorch wheel matrix](https://pytorch.org/get-started/previous-versions/#v241), [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus), [CUDA 12.1 driver matrix](https://docs.nvidia.com/cuda/archive/12.1.1/cuda-toolkit-release-notes/index.html).

```bash
git submodule update --init
nvidia-smi
docker build --platform linux/amd64 --target cuda -t pusht:cuda .
docker run --rm --gpus all --shm-size=2g -v "$PWD:/work" pusht:cuda \
  ./scripts/prepare_data.sh
docker run --rm --gpus all --shm-size=2g -v "$PWD:/work" pusht:cuda \
  python smoke_test.py --steps 3 --batch-size 2 --seed 42 --output results/gpu-smoke.json
```

The GPU smoke command runs the full published model with FP32, AdamW, and EMA. It records each synchronized iteration time, median steady iteration time, and peak PyTorch allocated and reserved VRAM. Timing includes host-to-device transfer and training, with the batch already fetched. It skips validation, rollout, and checkpoint creation. Batch-size-2 memory measurements apply to that test.

## RTX 4060 Laptop experiment

The `rtx4060_laptop` profile uses CUDA, 300 epochs, U-Net widths `[128, 256, 512]`, and a training and validation batch size of 8. It runs rollouts in groups of four environments while keeping the baseline episode counts. The CUDA smoke command checks the training step at batch size 8 and records peak memory. It skips validation and rollout, so those stages have separate memory requirements.

Run on the laptop in Linux x86_64 with NVIDIA Docker support. On Windows, use a GPU-enabled WSL 2 setup and run these commands from its Linux shell.

```bash
nvidia-smi
docker build --platform linux/amd64 --target cuda -t pusht:cuda .
docker run --rm --gpus all --shm-size=2g --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" pusht:cuda bash -euc '
    ./scripts/prepare_data.sh
    python smoke_test.py --config-name rtx4060_laptop --steps 3 --batch-size 8 \
      --seed 42 --output results/gpu-smoke-rtx4060-b8.json
  '
```

After reviewing the smoke result, start the 300-epoch experiment with:

```bash
docker run --rm --gpus all --shm-size=2g --user "$(id -u):$(id -g)" \
  -e PUSHT_CONFIG=rtx4060_laptop -v "$PWD:/work" pusht:cuda \
  ./scripts/train.sh experiments/rtx4060-seed42 training.seed=42
```

### WSL 2 validation without Docker

The RTX 4060 Laptop profile was validated on WSL 2 with an 8 GiB GPU, Python 3.11.18, PyTorch 2.4.1+cu121, and torchvision 0.19.1+cu121 on 2026-11-11. The official replay contains 206 episodes and 25,650 steps and passed the archive checksum check. Measurements and short training metrics are in [`results/rtx4060-validation.json`](../results/rtx4060-validation.json).

With Python 3.11, `curl`, `unzip`, and the runtime libraries listed in the Dockerfile available, prepare a local environment:

```bash
git submodule update --init
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install pip==24.0 setuptools==65.5.0 wheel==0.38.4
python -m pip install -r requirements.txt
python -m pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-build-isolation gym==0.21.0
python -m pip install --no-deps robomimic==0.2.0
./scripts/prepare_data.sh

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PUSHT_REPLAY_PATH=data/pusht/pusht_cchi_v7_replay.zarr python -m pytest -q
python smoke_test.py --config-name rtx4060_laptop --steps 3 --batch-size 8 \
  --seed 42 --output results/gpu-smoke-rtx4060-b8.json
```

All seven tests in that validation passed using the official replay. The three-step GPU smoke test used FP32, AdamW, and EMA with batch size 8. It measured 0.682 GiB peak PyTorch allocated memory, 0.686 GiB peak reserved memory, and 0.0669 seconds median iteration time after the first iteration. These memory and timing numbers cover training steps with the batch already fetched; they exclude rollout, validation, and checkpoint saving.

The following short run retains the 4060 model, batch size 8, EMA, and 100 inference steps. It executes two epochs of 20 training batches and four validation batches each, with four 32-step test episodes per epoch, diffusion sampling, video recording, and checkpoint saving:

```bash
PUSHT_CONFIG=rtx4060_laptop ./scripts/train.sh experiments/rtx4060-short-seed42 \
  training.seed=42 training.num_epochs=2 \
  training.max_train_steps=20 training.max_val_steps=4 \
  training.lr_warmup_steps=5 training.rollout_every=1 \
  training.val_every=1 training.sample_every=1 training.checkpoint_every=1 \
  dataloader.num_workers=2 val_dataloader.num_workers=2 \
  task.env_runner.n_train=0 task.env_runner.n_train_vis=0 \
  task.env_runner.n_test=4 task.env_runner.n_test_vis=1 \
  task.env_runner.max_steps=32
./scripts/evaluate.sh experiments/rtx4060-short-seed42/checkpoints/latest.ckpt \
  results/rtx4060-short-seed42-eval --device cuda:0
```

The short run finished in 25.536 seconds, with finite training and validation losses in both epochs. The saved EMA checkpoint was loaded independently for GPU evaluation, and its video decoded successfully. This validates the local workflow. The subsequent longer experiment and its 50-seed evaluations are archived below.

To start a fresh 300-epoch experiment in the activated environment, use:

```bash
OMP_NUM_THREADS=2 PUSHT_CONFIG=rtx4060_laptop \
  ./scripts/train.sh experiments/rtx4060-seed42 training.seed=42 \
  dataloader.num_workers=2 val_dataloader.num_workers=2
```

### Recorded RTX 4060 training result

The seed-42 experiment completed 232 epochs and was stopped by request during zero-based epoch 232, before reaching the planned 300 epochs. The best saved EMA checkpoint, at epoch 100, achieved a mean score of **0.7913** over the 50 initial conditions used for checkpoint selection. This metric averages the maximum normalized coverage score in each episode; it is not a binary success rate. The experiment used the reduced 4060 model and one training seed.

| Saved epoch | Test mean score |
| --- | ---: |
| 0 | 0.1385 |
| 50 | 0.7330 |
| 100 | **0.7913** |
| 150 | 0.7756 |
| 200 | 0.7772 |

The [experiment archive](../results/rtx4060-seed42/README.md) contains the full training logs, effective configurations, per-epoch metrics, per-test-episode scores, stop status, and an index of all **33 videos**: 30 from the longer run and three from short training and independent evaluation. All videos are committed to this repository. The nine model/optimizer/EMA checkpoints are retained locally and were not published, as requested; the archive records their metadata and SHA-256 checksums.

## Baseline training and resume

Run these commands in the CUDA environment when ready to start a full experiment. Keep the project directory bind-mounted to retain output across container restarts.

```bash
./scripts/train.sh experiments/baseline-42 training.seed=42 training.device=cuda:0
./scripts/evaluate.sh experiments/baseline-42/checkpoints/latest.ckpt results/baseline-42 --device cuda:0
```

Use a new directory for every fresh run. With the same profile, seed, device and overrides, use `./scripts/train.sh experiments/baseline-42 --resume training.seed=42 training.device=cuda:0` to load `checkpoints/latest.ckpt` in place and append to the training metrics log. The upstream workspace restores model, EMA, optimizer, and counters, reconstructs the scheduler from the stored step, and treats `training.num_epochs` as the number of epochs for this invocation. Set the remaining budget explicitly when resuming. The checkpoint captures counters before the end-of-epoch increment, so the saved epoch number repeats. The upstream checkpoint does not save RNG, dataloader or top-k manager state. This makes resume a continuation with restored weights and counters, without bitwise equivalence to uninterrupted training. Use only trusted checkpoints because the upstream format uses pickle.

Evaluation reads the rollout profile and seed from the checkpoint and writes to a fresh results directory. For a smoke checkpoint, this evaluates one short episode. A baseline checkpoint evaluates the 50 published test seeds.

## Action execution length study

The [baseline analysis](../results/README.md#rtx-4060-baseline-analysis) plots the archived training run and pairs the five recorded evaluations by initial condition. This simulation study can run independently of robot bring-up. To compare execution lengths, use one trusted checkpoint on the machine that holds the weights. The following command runs a development sweep with five initial conditions:

```bash
checkpoint='experiments/rtx4060-seed42/checkpoints/epoch=0100-test_mean_score=0.791.ckpt'
./scripts/evaluate.sh horizons "$checkpoint" results/horizons-dev-rerun \
  --device cuda:0 --action-steps 2 4 8 12 --test-start-seed 200000 --n-test 5 \
  --disturbance-step 50 --disturbance-delta-v 120 --record-first 2
```

The held-out run used the same settings and a separate set of 50 initial conditions:

```bash
OMP_NUM_THREADS=2 ./scripts/evaluate.sh horizons "$checkpoint" results/horizons-test-rerun \
  --device cuda:0 --action-steps 2 4 8 12 --test-start-seed 300000 --n-test 50 \
  --sampling-seed 42 --max-steps 300 --disturbance-step 50 \
  --disturbance-delta-v 120 --record-first 2
```

Both seed ranges are separate from the archived checkpoint-selection seeds (100000–100049). The evaluator uses the checkpoint's weights, observation history, prediction horizon and diffusion inference steps. Each condition runs with batch size one and resets the model sampling seed per episode. Pair rows by `test_seed`; a given initial condition uses the same sampling seed across all settings.

The [completed independent 50-seed evaluation](../results/README.md#independent-50-seed-action-execution-length-evaluation) archives all 400 episodes, eight condition summaries, 16 videos, and validation records. The development sample's highest mean score at 8 actions did not persist on the independent seeds.

With `--disturbance-step`, the sweep adds a disturbed trial for every clean trial. The perturbation applies a fixed +X velocity impulse to the T block before the specified one-based control step. The trigger follows simulator steps across chunk boundaries. A trial that ends before that step contributes to overall success and has no post-disturbance result. Report the exposed count with the post-disturbance success rate and compare the overall results on all paired seeds. A fixed trigger can fall at different positions within different action chunks.

The output directory contains:

- `protocol.json` with the checkpoint hash, seed ranges and rollout settings.
- `episodes.csv` with raw peak and final coverage, the clipped normalized score, the environment's success flag (coverage strictly above 0.95), executed control steps, disturbance exposure, steps to success after exposure, and inference timing. Timing includes observation transfer, prediction and action transfer back to the CPU.
- `summary.csv` with one row per scenario and action execution length. Recovery time is averaged over exposed episodes that succeeded.
- `videos/` when `--record-first` is set. The first N seeds are the same across conditions, and each file name contains its action execution length and seed.

Use the archived scores for historical comparison. The study evaluates one episode at a time with independently seeded sampling, while the archived training runner evaluated in parallel.

## B601 real robot demonstration import

The local backup `D:\下载\b601-github-backup-20261112` contains six independent tar volumes of LeRobot v3 recordings. Install the optional Parquet dependency with `python -m pip install -r requirements-real-data.txt`. To restore a fresh copy and run the full audit, use an empty destination:

```bash
python scripts/prepare_b601.py --extract \
  --source '/mnt/d/下载/b601-github-backup-20261112' \
  --raw data/b601 --output results/b601-data-check
```

If the recordings have already been restored and only the merged output is absent, omit `--extract`. The script refuses to overwrite source recordings or an existing merge. It checks the archive SHA-256 values on extraction, recording file counts and sizes against the source manifest, every embedded image, finite joint features, frame order, episode metadata and capture timing. Failed recordings are excluded and cause a nonzero exit status after the report is saved. Pilot, unfinished and stationary recordings are retained separately from the formal merge.

The restored originals are in `data/b601/pick-place-*`; the start-pose reference is in `data/b601/references/`. The merged artifact is `data/b601/merged/demonstrations.zarr`, with `data/img` as 320×240 RGB, `data/state` and `data/action` as seven original joint position features, local timestamps and frame indices, and `meta/episode_ends` for episode boundaries. `data/b601/merged/episodes.json` maps each merged episode to its source recording. Original 640×480 image bytes and all source labels remain in the Parquet files. No action shifting, time resampling, coordinate conversion or trimming is applied.

The audit writes `results/b601-data-check/report.json`, `recordings.csv`, `summary.txt`, and sampled image previews. These structural checks do not establish task success. The `b601` and `b601_short` profiles use the local seven-joint dataset adapter and full 320×240 observations; the original PushT profiles remain configured for two-dimensional simulation actions.

For the additional action/command alignment audit and visual-review evidence, install `requirements-analysis.txt` and run:

```bash
MPLCONFIGDIR=/tmp/b601-matplotlib python scripts/review_b601.py
```

This exports seven-joint state/target plots and event-focused image sheets for all 40 formal episodes. For episodes 0–6, it also exports every frame in chronological sheets and seven 640×480 videos, verifying each video's decoded frame count. Video and sheet times use nominal 10 Hz; plot axes use the measured camera receive times. Numeric alignment checks detect action changes without a command, changes during held commands, and reversed feedback/command timestamp order. Source-code verification is recorded separately from log evidence.

The local visual-review record is `results/b601-data-check/visual_review.json`. Its provisional `review/training_candidates.json` selects 34 episodes / 19,269 frames by excluding original episode indices 0–5 from full-task training; all original data remain intact. This candidate list reflects visual-review decisions and is not regenerated automatically by the evidence exporter. Candidate status does not certify successful physical grasping or placement. The supplied collection source at `6416316` and B601 driver 1.0.5 confirm that stored actions are persistent, absolute motor-degree targets returned by the driver after clipping. See `review/source_verification.json` for evidence and limits; the archive does not identify its exact historical source commit.

### B601 offline short training

The user reported no known failed, retried or incomplete demonstrations in the retained set. The frozen seed-42 split uses 30 training episodes and four held-out episodes (original indices 8, 20, 27 and 30). The adapter fits joint/action normalization on training episodes only, reads images on demand from the original merged replay, keeps the full field of view, and preserves observation-before-command alignment. A batch contains images `[8, 2, 3, 240, 320]`, joint positions `[8, 2, 7]` and targets `[8, 16, 7]`.

In the activated training environment, prepare the immutable split, then use a fresh run directory:

```bash
export PYTHONPATH="$PWD:$PWD/upstream/diffusion_policy${PYTHONPATH:+:$PYTHONPATH}"
python -m real_robot.dataset
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PUSHT_CONFIG=b601_short \
  ./scripts/train.sh experiments/b601-short-new
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python scripts/validate_b601_training.py experiments/b601-short-new \
  --output results/b601-short-new --device cuda:0
```

`b601_short` runs three capped epochs of 50 batches each, with batch size 8, FP32, AdamW, EMA and 100 diffusion sampling steps. This is 150 optimizer updates, not three complete passes through the dataset. Its per-epoch main-model validation is capped at 16 batches. The independent checkpoint validator evaluates the saved EMA model over all 2,347 held-out windows, samples one sequence per held-out episode, and checks exact model/EMA/optimizer restoration followed by one in-memory optimizer update. It preserves the archived checkpoint.

The [recorded short-run validation](../results/b601-short-seed42/validation.json) completed on an RTX 4060 Laptop GPU in 26.13 seconds, with 1.094 GiB peak allocated and 1.316 GiB peak reserved PyTorch memory. Training epoch means decreased from 1.0354 to 0.5784; capped main-model validation noise MSE decreased from 0.9797 to 0.4249. The independently loaded EMA model's full held-out noise MSE was 0.5646. These are diffusion noise losses, not robot success rates. The short run validates the offline workflow; no robot was connected or moved. Checkpoints remain under the ignored local experiment directory.

The pinned diffusers 0.11.1 scheduler's final float32 coefficient is slightly above one. The sampled targets exceeded the training min/max by at most 0.000443 motor degrees; the validator records this deviation and checks the normalized output against the scheduler's derived coefficient. Prediction and motor execution limits are separate concerns. Stored actions are native absolute motor-degree targets: the direct SDK controller converts degrees to radians without changing direction/scale. Calling the higher-level collector driver instead requires compensating for its input direction/scale mapping exactly once. Both interfaces must retain execution limits.

### Full B601 training after the short run

`configs/b601_train.yaml` runs 100 complete epochs with the same 30/4 episode split, batch size 8, full images, FP32 and EMA. Each epoch has 2,086 optimizer updates and validates all 2,347 held-out windows. It saves the latest checkpoint every epoch and retains the three checkpoints with the lowest full main-model validation noise loss. The learning rate uses cosine decay with no second warmup when continuing the already warmed-up short run.

The B601 workspace restores the actual optimizer update count for EMA decay. Periodic checkpoints advance to the next epoch when loaded; the final checkpoint marks its counters as already advanced. Checkpoint writes finish before training continues. RNG and dataloader state are still not restored, so continuation is not bitwise identical to an uninterrupted process.

For a fresh output directory, the supervisor copies the source checkpoint, records its hash, starts training and then independently validates the final checkpoint:

```bash
python scripts/run_b601_training.py \
  --run-dir experiments/b601-new \
  --output results/b601-new \
  --source-checkpoint experiments/b601-short-seed42/checkpoints/latest.ckpt
```

The completed run is `experiments/b601-seed42`: 100 full epochs, 208,600 additional optimizer updates, and 26,018 seconds on the RTX 4060 Laptop GPU. Its [launch record](../results/b601-seed42/launch.json) preserves the implementation hashes and source snapshot, and its [supervisor status](../results/b601-seed42/training_status.json) records completion and evaluation recovery. Per-batch and per-epoch metrics are archived in `logs.json.txt`; console progress is in the experiment's `train.log`. The source short-run checkpoint remains intact.

The initial automatic validator incorrectly required `train_action_mse_error` on epochs where action sampling was not scheduled. Required losses and optional sampled metrics are now checked separately, with regression tests. The rerun passed exact checkpoint restoration, finite predictions/gradients, repeatable seeded sampling and an in-memory optimizer continuation. This script error did not interrupt training or modify saved weights. Both the original failure and successful recovery are retained in the status and validation logs. Final EMA noise MSE over all 2,347 validation windows is 0.026800, compared with 0.564569 after the short run.

To compare saved main/EMA weights and the short-run EMA with fixed action-prediction baselines:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python scripts/evaluate_b601_checkpoints.py experiments/b601-seed42 \
  --output results/b601-seed42/evaluation \
  --short-checkpoint experiments/b601-short-seed42/checkpoints/latest.ckpt
```

This evaluates 32 equal temporal bins in each of the four validation episodes, two sampling seeds per window, and the eight executable actions at indices `[1:9]` of the 16-step prediction. It compares against holding the latest measured joint positions and a constant training-mean target. Noise losses cover all 2,347 validation windows with matching seeds. The JSON report preserves checkpoint hashes, exact window indices, per-joint/per-episode errors and subsets where targets change by at least one motor degree. The same validation episodes selected checkpoints, so these comparisons are not an independent test set and provide no closed-loop robot success rate.

The [completed comparison](../results/b601-seed42/evaluation/summary.txt) found final EMA normalized action MSE of **0.002655**, versus **0.583900** after short training and **0.003349** for holding measured position. Final six-arm-joint MAE is **0.2921 motor degrees**; gripper MAE is **1.7549 motor degrees**, above the hold baseline's **1.3553**. Relative to holding, changing-target MSE improves by 37.65%, while held-target MSE worsens by 25.51%. Three validation episodes improve and original episode 8 worsens. Final main and EMA predictions are close; both outperform the checkpoint selected at formal epoch 32 by its minimum noise loss. The rising late noise validation curve therefore does not by itself select the best action predictor.

The final EMA is a candidate for subsequent robot validation, with substantial limits: the [20-run batch-one latency measurement](../results/b601-seed42/evaluation/latency.json) gives a 0.912-second median and 1.067-second P95 for 100 diffusion steps, including observation/action transfers but excluding camera and robot I/O. This exceeds the nominal 0.8-second eight-action execution interval. Inference/control timing, held-target errors and gripper behavior need attention before closed-loop task performance can be established. No robot was connected or moved during evaluation. Full metrics and predictions are in `evaluation/evaluation.json` and `evaluation/predictions.npz`, with overview and subset/episode plots alongside them.

To compare the final EMA with DDPM 100 steps and DDIM 100/50/20/10 steps, use an empty output directory:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/b601-matplotlib \
  python scripts/benchmark_b601_sampling.py \
  --reference results/b601-seed42/evaluation \
  --output results/b601-seed42/sampler-benchmark-new
```

The sampler benchmark reuses the exact 128 windows and two random seeds from the preceding comparison. It copies the training betas exactly into DDIM, uses deterministic DDIM transitions (`eta=0`) with the pinned scheduler's default leading timetable, and leaves model parameters and archived inference settings unchanged. The old DDPM implementation only supports the complete 100-step reverse chain here; strided DDPM is rejected. Analytic tests check DDIM transitions against a known clean sample, including the final transition.

Accuracy covers the eight executable actions, with separate changing-target, held-target, gripper-transition and per-episode metrics. Latency uses batch size one, three warmup calls per variant and 100 timed calls per variant. Every timed round uses the same observation and random seed for all five variants, and rotates variant order to balance measurement positions. Timing includes observation transfer, inference and action return with CUDA synchronization; it excludes disk/image preprocessing, camera/robot I/O and random-seed setup. The 0.8-second action interval is a possible overlap budget, not a guaranteed complete control period when inference and execution run in series. Reports, raw latency rows, predictions, source snapshots and checksums are saved under the chosen output directory.

The [completed sampler comparison](../results/b601-seed42/sampler-benchmark/summary.txt) contains 500 balanced, interleaved timing measurements:

| Sampler | Median seconds | P95 seconds | Normalized action MSE |
| --- | ---: | ---: | ---: |
| DDPM 100 | 0.935 | 1.125 | 0.0026547 |
| DDIM 100 | 0.932 | 1.167 | 0.0026628 |
| DDIM 50 | 0.459 | 0.674 | 0.0026517 |
| DDIM 20 | 0.186 | 0.276 | 0.0026357 |
| DDIM 10 | 0.096 | 0.196 | 0.0026247 |

DDIM 10 is the candidate for predict-only integration: median inference is 9.73 times faster than DDPM 100, while sampled overall action error is essentially unchanged. This does not establish a task-success improvement. Overall gripper MAE rises from 1.7549 to 1.7902 motor degrees (+2.01%); episode 30 gripper MAE rises by 9.45%. The episode 30 increase is already present with DDIM 100, so it cannot be attributed solely to reducing the step count. Held-target and episode 8 errors also remain above the hold-position baseline. The proposed [candidate parameters](../results/b601-seed42/sampler-benchmark/candidate.json) are an artifact for subsequent integration; no active inference configuration or checkpoint was modified.

An initial comparison stopped because a strict cross-process coordinate tolerance was exceeded by floating-point differences up to 0.005676 motor degrees. The original attempt is preserved. The complete rerun checks maximum motor/normalized coordinate differences and aggregate MSE consistency; its DDPM MSE differs from the previous evaluation by only 0.000561%. Every sampler passed finite-output and same-seed repeatability checks. All 44 repository tests passed. Exact settings, library versions, source snapshots, per-episode/per-joint results, raw timing rows and checksums are archived in `results/b601-seed42/sampler-benchmark/`. The existing four validation episodes were reused; no independent robot trials were performed.

The native WSLg window **B601 实时训练监控 | Live Training Monitor** refreshes every two seconds. It displays full-epoch progress, recent training loss, full held-out validation loss, GPU memory/load/temperature, elapsed time and an estimate of remaining time. The monitor reads logs incrementally and runs independently of the training supervisor. Closing the window leaves training running. To reopen it from the project root:

```bash
MPLCONFIGDIR=/tmp/b601-matplotlib python scripts/monitor_b601_training.py
```

Its live snapshot is `results/b601-seed42/monitor_status.json`. Monitor diagnostics are in `monitor.log`.

### B601 next steps

2026-11-14 进度交接：数据审计、短训练、100 轮完整训练、检查点复核、动作预测对比和五组采样器测试均已完成。当前候选为最终检查点的 EMA 权重和 DDIM 10 步。Windows 相机/关节反馈到 WSL GPU 的只预测链路现已实现，并完成停放姿态下的静态联调。后续已加入同一串口控制器内的受控单步模型执行入口，见文末；完整抓放成功率尚未验证。

后续用户指定以历史范围替代本次手动边界采点：采用 40 段完整正式示范的实测关节状态，计算 `end_link` 的 XYZ 包围盒，X/Y 两侧及 Z 上方向外扩展 1 cm，Z 下方扩展 0.5 cm。本机配置为 `collection/b601/workspace-bounds.local.json`；这个已确定的边界来源策略替代下文原计划中的现场测量步骤，后续仍需核对零位、参考起点和执行时序。

| 继续工作所需内容 | 位置或取值 |
| --- | --- |
| 本地检查点 | `experiments/b601-seed42/checkpoints/latest.ckpt`，权重键 `ema_model` |
| 检查点 SHA-256 | `5e19327cf1178586f360feecda4b832caddeb4ef32677104c46467c78a00c9f6` |
| 候选采样参数 | [`candidate.json`](../results/b601-seed42/sampler-benchmark/candidate.json)：DDIM 10 步，沿用训练的 beta，`eta=0` |
| 输入与输出 | 两帧 `320×240` RGB、七维关节反馈；预测 16 步，候选执行片段为索引 `[1:9]` 的 8 步绝对电机角度目标 |
| 采集与驱动语义依据 | [`b601_pick_place.py`](../collection/b601/b601_pick_place.py)、[采集指南](../collection/b601/README.md)、[源码核对记录](../results/b601-data-check/review/source_verification.json) |
| 固定数据划分 | [归档 split](../results/b601-short-seed42/dataset_split.json)，验证 episode 为 `8,20,27,30`；归一化仅使用训练 episode |
| 复核结果 | 全部 44 项 CPU 测试通过；采样器测试保存 500 条计时记录与原始预测数组 |

原始录制、合并 Zarr、模型权重和现场工作区配置由 `.gitignore` 排除。新克隆需要另外恢复数据和检查点；只有仓库中的报告不能直接重跑真实数据训练。动态监控快照及监控日志也保留在本地，完成的训练和评估报告已归档。

原定联调顺序如下。目前只预测链路、原生闭环执行、完整记录和时序复测已实现，现场边界采用用户指定的历史实测包围盒扩展。多轮真机试验仍未建立抓放成功，当前下一步是离线核对现场输入与训练示范、定位策略接近漂移；不能把接口和时序通过计作任务通过。

1. **实现只预测模式。** 复用采集工程的相机和反馈读取逻辑，接入最终 EMA＋DDIM 10 步。先确认读取连接流程不会隐式使能电机、发送目标或修改驱动状态；该模式不调用 `send_action`。以训练时相同的 RGB 转换、缩放、两帧顺序、关节顺序和归一化构造输入。先用归档帧比对离线输出，再连接现场输入。
2. **核对动作接口。** 同时记录模型输出的绝对电机角度和真正下发目标。训练标签是采集驱动裁剪后返回的原生电机目标；当前真机控制直接调用 SDK，只做度到弧度转换，不再套用 `joint_directions`。若另行调用会再次乘方向倍率的采集器高层接口，输入才需按实际方向倍率除回，核对版本中夹爪倍率为 `-6`。不能在原生 SDK 路径重复换算；接口应验证七维顺序、方向、倍率及限位。
3. **记录实时链路和预测稳定性。** 保存帧号、图像收到时间、反馈读取时间、推理开始/结束时间、预测目标及拟发送目标，并显示延迟和过期输入。覆盖静止、接近棋子、夹爪开合、搬运和释放阶段，重点复核 episode 8 的退化及 episode 30 的夹爪误差。先完成一次原流程演示的只预测记录，逐段检查是否出现异常跳变和保持阶段偏移。
4. **验证完整控制时序。** 当前推理中位数为 0.096 秒、P95 为 0.196 秒，不含相机、预处理和机器人 I/O。明确采集、推理和执行的串行或并行安排，测观测年龄、端到端延迟、动作间隔及停顿。8 动作按 10 Hz 执行约 0.8 秒只是动作块时长；不能据此认定整条闭环已满足 10 Hz。
5. **开展受控闭环试验。** 接口与时序核对通过后，使用现场确认的工作区、关节限位、速度和单周期动作增量限制，在有人接管的条件下从单步/短动作开始，再覆盖完整抓取与放置。逐次记录实际起始条件、抓取成功、放置成功、停顿、失败阶段和人工接管；区分命令发送完成与实测关节到达。使用新试验评估成功率，再按失败原因决定补采数据或调整模型。

当前仍需处理的质量问题：DDIM 10 步总体夹爪 MAE 为 1.7902 电机度，静止基线为 1.3553；episode 30 的夹爪误差比 DDPM 100 步高 9.45%。保持阶段与 episode 8 的归一化误差仍高于静止基线。采样加速没有解决这些问题，四个参与过选模型的验证 episode 也不能替代独立的真机成功率测试。

### Windows / WSL 只预测联调

[`b601_windows_bridge.py`](../scripts/b601_windows_bridge.py) 通过 Windows 原生 SDK 读取串口和外接 RealSense RGB 相机，将最近两帧图像及关节反馈传给 [`run_b601_shadow.py`](../scripts/run_b601_shadow.py) 中的 WSL GPU 推理。采集线程以标称 10 Hz 更新，推理按 0.8 秒间隔取最近的观测对。协议只接受 `ping`、`sample` 和 `stop`，没有运动目标或使能入口；退出不调用 SDK 的控制器上下文管理器或 shutdown，避免隐式改变电机状态。

Windows Python 需要 NumPy、`motorbridge==0.5.6`、`pyrealsense2==2.59.1.11053`。本机的两个官方 Windows wheel 校验后放在 Git 忽略的 `data/tools/windows-hardware/` 中，没有改训练环境或 Windows 全局依赖；驱动配置以 AST 读取 `data/tools/b601-driver-review/lerobot_robot_seeed_b601/config_seeed_b601_dm_follower.py` 中的字面量，不导入会使能电机的高层驱动。恢复到另一台机器时，需要准备相同的 SDK、已审计的驱动源码、数据及检查点。

在 WSL 的项目根目录运行。将 `USER`、串口和相机序列号替换为当前现场值，使用全新的输出目录：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/b601-matplotlib \
  python scripts/run_b601_shadow.py \
  --windows-python /mnt/c/Users/USER/AppData/Local/Programs/Python/Python311/python.exe \
  --port COM5 --camera-serial CAMERA_SERIAL \
  --output data/b601/shadow-session-new --duration 60
```

启动前先检查四段验证 episode 的原始全分辨率图像预处理与训练数据完全一致，再复现归档的 DDIM 预测。实时保存 `acquisition.jsonl`、`predictions.jsonl`、逐次输入/预测 NPZ、`latest-input.png`、`latest.json`、`status.json` 和 `summary.json`，全部位于本地忽略的数据目录。映射记录包含绝对电机目标、除以实际驱动方向倍率后的拟输入及限位投影；这些数值不会发给机械臂，也没有套用尚未验证的闭环相对目标限制。

首轮试运行暴露了 Windows QPC 与 WSL 单调时钟速率不同的问题，固定时钟偏移导致过期判断失真。修正后在每次取样前做往返校准，结束后用周期校准点插值估计相机接收至预测返回的延迟；过期判断分别使用采集端自身的帧年龄和 WSL 的桥接往返耗时，不用固定跨时钟偏移。该估计仍不等于曝光至电机执行的延迟。SDK 状态没有单电机接收时间戳，每次请求是否取得新反馈也仍未证明，日志明确保留这个限制。

2026-11-14 停放姿态复测在 60 秒内完成 **74 次有效预测**，Windows 保存 **574 条采集记录**，启动时一次等待预热的请求被丢弃。推理中位/P95 为 **138.8/236.7 ms**，有效请求的桥接往返为 **7.47/9.28 ms**，经周期时钟校准的相机接收至预测返回估算为 **222.3/307.8 ms**。四段原始图像预处理及归档模型输出复现完全一致，七关节方向倍率与已审计配置匹配，全程使能、失能、校准和运动指令均为零。

用户确认此时是停放姿态。五个关节的反馈超出训练状态范围，预测首步相对实测位置的最大差距为 **86.6 电机度**；这验证了链路，但不能用于判断模型在正常起点的抓放质量。随后用户选择了下节的历史范围派生边界方案。下一步是在该边界配置下核对零位与参考起点，再复核画面和动作阶段。边界采用底座坐标系的 `[XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX]`，单位米，约束模型的 `end_link` 参考点；历史包围盒本身没有验证夹指、棋子、桌面和其他物体的碰撞余量。

### 历史范围派生的工作区

在上述停放联调之后，用户明确指定使用历史实测到达范围并向外扩展，替代手动采点。运行 `python scripts/prepare_b601_workspace.py` 会从合并的 40 段完整正式示范、20,741 帧 `state` 计算末端参考点坐标，写入本机 `collection/b601/workspace-bounds.local.json`。不使用指令目标，也不混入停放联调记录。计算依据、极值所在 episode/帧和数据/URDF 校验和保存在 `data/b601/workspace-from-history/report.json`。

| 轴 | 历史最小值（米） | 历史最大值（米） | 扩展后最小值（米） | 扩展后最大值（米） |
| --- | ---: | ---: | ---: | ---: |
| X | 0.202040627 | 0.368957855 | 0.192040627 | 0.378957855 |
| Y | -0.080605857 | 0.102711972 | -0.090605857 | 0.112711972 |
| Z | 0.009389702 | 0.141013639 | 0.004389702 | 0.151013639 |

所有历史点和已保存的参考起始末端位置都位于该范围内。这是用户指定的历史包围盒扩展；未验证盒内其他位置、角点的可达性或无碰撞性，坐标仍依赖录制时零位和 URDF。生成配置本身不发送机械臂指令。

### Windows / WSL 受控模型短复现

2026-11-14 已实现并完成首段连续 **10 个模型动作**的真机闭环。Windows 的 `scripts/b601_manual_control.py --model-control` 独占 COM5，同时由相机线程获取 RGB；WSL 的 `scripts/run_b601_model_trial.py` 加载原检查点 EMA＋DDIM 10 步，使用最近两帧观测，每次重新推理并等待实测到位。执行端与键盘控制共用一个控制器，Esc 和窗口失焦仍会暂停；切换窗口进程必须先确认旧控制器退出。

用户要求加速后，本机 `setup.json` 的 `joint_velocity_deg_s` 设为 `[30,30,30,30,30,30,12]`，恢复采集时的速度设置，取消了额外的 5°/s 联调慢速档。没有无限速或改写硬件寄存器。单步仍保留最多 0.5 臂关节电机度、2 夹爪电机度和 6 mm 末端距离；关节限位、整条插值路径的历史工作区检查及 2° 臂关节跟踪保护继续有效。

首轮大段参考姿态插值触发肘关节 2.01° 跟踪保护，随后改为实际到位后再推进的小步对齐，保持夹爪开度。连续动作取样等待相机两帧刷新；输入最多允许 1 秒年龄和 0.3° 状态变化。控制请求采用 Windows 自身单调时钟签发的短期凭据，避免跨 Windows/WSL 墙上时钟比较。对于允许范围内的观测后微小漂移，执行端再次依据下发瞬间的反馈限幅，并同时记录请求目标、实际接受目标和实测到位位置；没有放宽保护阈值。

完成的会话保存在本地 `data/b601/model-trial-20261114-03/`：`model-trial.json` 为 `complete_holding`、10 步完成，`model-trial-events.jsonl` 记录每步输入、原始预测、有界目标和实测值，`verification.json`、`final-hardware-status.json` 及源码快照保存验证依据。最终七电机均使能、无本会话故障，窗口暂停保持；本地 `data/b601/active-controller.json` 指向实际控制进程。之前两次中止和诊断记录分别保留在 `model-trial-20261114-01/`、`model-trial-20261114-02/`。当前全部 69 项 CPU 测试通过。

这证明了受限幅的观测—推理—实际运动闭环，尚未验证完整抓取和放置成功。下一步继续从当前状态观察完整任务的接近、夹持、抬升、移动和释放阶段，记录失败或接管位置，再决定是否补采；不能把 10 步命令执行完成计作抓放成功。`--resume` 从当前位置继续，默认不带该参数时先对齐已保存的机械臂参考姿态。

后续按用户要求，将实时 RGB 嵌入同一操作窗口，并启动连续模型运行。本机 `data/b601/live-model-20261114-01/` 保留这次会话；预览直接使用同一相机管线，原生 Tk 显示约 15–20 帧/秒，推理输入仍取约 10 Hz 的两帧状态对。窗口按钮启动已训练 EMA＋DDIM 10 的 `--continuous --duration 300 --resume`，取消 10 步短试跑限制，持续闭环最多 5 分钟；没有宣称达到训练标称 10 Hz 的动作执行频率。Esc、窗口失焦和手动接管会推进控制代号，使正在加载或处于两步间隙的模型进程也永久取消本次运行。断流时暂停。新增暂停回归测试后，全部 71 项测试通过。

初版记录范围：采样 NPZ 有每次输入的两帧 RGB 和关节状态；事件日志有每步首个原始模型目标、限幅目标、接受目标及到位反馈，以及故障和运行日志。该轮在 61 步后由操作员暂停，缺失的连续视频等记录无法补录。

用户随后要求补齐，已在 `data/b601/live-model-recorded-20261114-01/` 启用完整记录：同一 RealSense 管线同时读取 30 fps RGB 与 Z16 深度，异步写入 5 秒分段的无损 RGB MKV、15 帧分块的无损深度 NPZ、逐帧时间和传感器标识；保存相机内参、深度单位及深度到彩色外参。逐控制周期另存关节位置、速度、扭矩、温度、状态、CAN 标识、目标和时序。推理在下发前保存全部 16×7 预测与 8×7 可执行片段，使用采样、运行和录制标识关联；原有事件和故障日志保留。记录失败或中断会暂停模型，下一次显式启动时先开记录。清单注明 SDK 单电机反馈新鲜度未知，不把缓存读取时间当作传感器采样时间。

74 项全量测试通过。真机静态录制读回 188 帧视频、188 帧深度、124 个约 19.86 Hz 控制周期且无源帧缺口；第二段读回 45 帧视频、45 帧深度、30 个周期及一次真实候选模型的完整预测，视频中的两帧模型输入逐像素一致。验证没有发送模型运动指令。曾发现 Windows 将 Linux 视频路径转换为反斜杠、导致视频落在仓库根目录，已改为平台无关的 POSIX 路径并归档原视频，修正后的第二段路径正常。验证报告与源码快照位于该会话，控制窗口保持暂停。

2026-11-15（上海时间）按用户要求开展第一次动作序列／滚动时域完整任务预实验，保存于 `data/b601/experiments/rhc-trial-001/`。对照原仓库 `eval_real_robot.py` 的运行时设置，输出 15 个未来目标，按 10 Hz 时间戳调度，每 6 步重新推理并替换未来目标，丢弃过期动作；没有逐步等待到位。继续使用已评估的最终 EMA＋DDIM 10 候选。控制端恢复采集驱动原有的每关节相对目标裁剪、保持速度、工作区和跟踪保护。序列覆盖按半个周期合并接收时间抖动下的同槽目标，每轮唯一标识隔离旧状态和新录制。

在保留工程预检及中止尝试之后，本轮实际运行 14.0997 秒，完成 23 次滚动推理和 129 个动作下发；动作间隔中位数 100.1071 ms、P95 130.3565 ms。随后触发“动作调度迟到超过 75 ms”保护，抓放未完成，结果为时序中断而非已验证的策略成功率。终止时跟踪误差未超过 2°，记录正常封存，七电机继续使能并暂停保持。完整结果、录制标识和记录完整性复核见本机会话的 `outcome.json`。下一步应将观测打包、IPC、文件操作移出执行线程，先验证完整记录下的调度稳定性，再开展后续任务试验。

按用户要求修复并复测后，请求文件轮询、图像序列化、输入打包、序列预检、回复及事件写盘均转移到后台线程。控制端通过有界内存队列接收请求，并在执行前重新验证控制代号和时效，避免暂停后落地旧请求；日志失败会停止执行。反馈/安全循环改为 50 Hz，动作时间戳继续标称 10 Hz，调度 75 ms 阈值、速度、力矩、关节和工作区保护均保留。新增异步 I/O、重启去重、取消及逐运行计数测试，共 85 项通过。

本地 `rhc-timing-validation-001` 完成 60 秒固定姿态满负载验证，最大下发迟到 30.8132 ms；该段 100 次真实模型推理与视频/深度等完整录制同时运行，调度目标保持常量。之后 `rhc-trial-002` 抓放复测运行 171.7282 秒，286 次推理、1,638 个动作，下发间隔中位数 101.5915 ms、P95 104.6192 ms，最大迟到 44.9732 ms，无调度、RPC 或事件日志错误。

操作员明确反馈“根本就没有碰到棋子”，因此结果标为 `failed_approach_no_contact`，任务未完成。最后输入的腕俯仰为 57.4291°，超过训练状态最大值 57.3198° 及 0.05° 容差，触发统计范围检查；物理关节与工作区约束均未越界。正常封存并回读 5,935 帧视频、5,935 帧深度、9,656 个控制周期和 286 条预测，无相机源帧缺口。该轮解决了观测到的时序中断，未证明抓放能力。下一步应对照实际训练示范核对初始画面、七维起点和接近目标的漂移，再决定针对性补采或训练调整。结果与操作员标签位于 `data/b601/experiments/rhc-trial-002/`，所有旧尝试保留。

第二次任务试验的首次启动 `rhc-trial-003` 在模型执行前中止：自动对齐旧静态参考的 39 次准备动作使 X 从约 332 mm 退到 285 mm，首段策略请求随后因观测过期被拒绝，模型序列动作数为零。操作员指出棋子位于 +X；视频与反馈核对确认这次退向底座由参考对齐造成。该段标为 `aborted_before_policy_execution`，不能用于判断策略方向。

用户随后指定当前位置为起点，保存于 `rhc-trial-004/start-pose.json`，末端参考点约为 `[286.28, 12.68, 24.99]` mm。该会话设置 `align_reference_on_start=false`，窗口以 `--resume` 启动；此模式只取当前观测，不发送任何参考对齐动作，原生执行端也拒绝 `reference_alignment` 请求。起点记录不修改编码器零位。相关检查包含在已通过的 85 项测试中。

这次实际策略运行 54.67 秒、91 次推理、517 个动作，自动对齐动作数为零。运行中 X 降至约 258 mm、Y 升至约 88 mm，与操作员指出的棋子方向不符，助手主动暂停，结果标为 `assistant_stopped_wrong_approach_direction`。517 个动作的六个臂关节目标均与原始策略输出在浮点精度内一致，最大差异约 2.78×10⁻¹⁷ 电机度；末段原始目标与实测位置同向，不能将本轮偏移归因于自动复位或臂关节符号翻转。起始预测的不同未来步也包含不同的 X 方向，不能将整条预测概括成单调负 X。完整读回 1,749 帧视频、1,749 帧深度及 2,852 个控制周期，无相机源帧缺口。

该轮完整记录和结果保存在本地 `data/b601/experiments/rhc-trial-004/`；`diagnostics/direction-audit.json` 与 `video-timeline.png` 保存方向及读回核对。暂停后电机保持使能，模型已退出。下一步对冻结的真机图像/状态与实际训练示范做离线预测对照，检查视觉条件与接近动作；尚未定位模型偏移根因，不应直接反转坐标或再次盲跑。

用户接着明确要求“先复位然后再跑一次”，`rhc-trial-005` 因此恢复到上述保存的起点，而非更早的静态参考。43 次准备动作完成复位；启动重试前七维反馈相对参考的最大误差为 0.044°。第一段预测因观测超过 500 ms 被拒绝，模型动作数为零，原始记录保留在 `attempts/reset-complete-first-plan-expired/`。启动器补充在实际复位后重新进行不下发动作的 GPU 预热，观测时效阈值不变；随后从已复位位置重新启动。

该轮模型实际运行 246.0007 秒，410 次推理、2,338 个序列动作，随后由暂停/接管控制代号变化结束。日志无法区分此次变化来自 Esc、人工接管还是窗口失焦。封存 7,200 帧 RGB/深度和 11,741 个控制周期；清单和 JSONL 读回数量一致，无相机源帧缺口。反馈末端由约 `[286.48,12.75,24.98]` mm 到 `[257.46,87.17,30.04]` mm，偏移仍存在。`outcome.json` 标为 `stopped_without_verified_task_success`，没有确认抓取或放置成功；结束后控制器暂停、七电机使能、无本轮硬件故障。

按用户要求，代码和小体积证据归档到 [`results/b601-hardware-20261115/`](../results/b601-hardware-20261115/README.md)。完整数据保留在原本机路径；大体积视频、原始深度、连续输入快照、示范数组/图像和权重不进入 Git，清单保留其大小与 SHA-256。新的归档文件上限为 5 MiB，超过上限的文本日志先进行无损 gzip 压缩。该归档覆盖启动中止、接管、失败诊断和完整控制记录，不只保留最后一次尝试。当前 85 项测试通过。
