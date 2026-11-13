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

The pinned diffusers 0.11.1 scheduler's final float32 coefficient is slightly above one. The sampled targets exceeded the training min/max by at most 0.000443 motor degrees; the validator records this deviation and checks the normalized output against the scheduler's derived coefficient. Prediction and motor execution limits are separate concerns; hardware inference must apply the driver's inverse direction/scale mapping and its execution limits.

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

2026-11-14 进度交接：数据审计、短训练、100 轮完整训练、检查点复核、动作预测对比和五组采样器测试均已完成。当前候选为最终检查点的 EMA 权重和 DDIM 10 步；实时推理接入与真机闭环执行尚未实现或验证。

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

下一步按下面顺序推进；以下是待实现计划，不是现有命令或已通过的验收项。

1. **实现只预测模式。** 复用采集工程的相机和反馈读取逻辑，接入最终 EMA＋DDIM 10 步。先确认读取连接流程不会隐式使能电机、发送目标或修改驱动状态；该模式不调用 `send_action`。以训练时相同的 RGB 转换、缩放、两帧顺序、关节顺序和归一化构造输入。先用归档帧比对离线输出，再连接现场输入。
2. **核对动作接口。** 同时记录模型输出的绝对电机角度和拟传给驱动的目标，验证正反向转换。训练标签是驱动裁剪后返回的电机目标；驱动输入需按实际 `joint_directions` 除回，当前核对版本中夹爪倍率为 `-6`。按现场驱动配置核对每个关节，不把该倍率当成所有机器的通用常量；模拟接口应验证七维顺序、方向、倍率及限位。
3. **记录实时链路和预测稳定性。** 保存帧号、图像收到时间、反馈读取时间、推理开始/结束时间、预测目标及拟发送目标，并显示延迟和过期输入。覆盖静止、接近棋子、夹爪开合、搬运和释放阶段，重点复核 episode 8 的退化及 episode 30 的夹爪误差。先完成一次原流程演示的只预测记录，逐段检查是否出现异常跳变和保持阶段偏移。
4. **验证完整控制时序。** 当前推理中位数为 0.096 秒、P95 为 0.196 秒，不含相机、预处理和机器人 I/O。明确采集、推理和执行的串行或并行安排，测观测年龄、端到端延迟、动作间隔及停顿。8 动作按 10 Hz 执行约 0.8 秒只是动作块时长；不能据此认定整条闭环已满足 10 Hz。
5. **开展受控闭环试验。** 接口与时序核对通过后，使用现场确认的工作区、关节限位、速度和单周期动作增量限制，在有人接管的条件下从单步/短动作开始，再覆盖完整抓取与放置。逐次记录实际起始条件、抓取成功、放置成功、停顿、失败阶段和人工接管；区分命令发送完成与实测关节到达。使用新试验评估成功率，再按失败原因决定补采数据或调整模型。

当前仍需处理的质量问题：DDIM 10 步总体夹爪 MAE 为 1.7902 电机度，静止基线为 1.3553；episode 30 的夹爪误差比 DDPM 100 步高 9.45%。保持阶段与 episode 8 的归一化误差仍高于静止基线。采样加速没有解决这些问题，四个参与过选模型的验证 episode 也不能替代独立的真机成功率测试。
