# Diffusion Policy PushT Reproduction

Reproduce the image-conditioned CNN Diffusion Policy on simulated PushT. The upstream code is pinned at commit `5ba07ac6661db573af695b419a7947ecb704690f` in `upstream/diffusion_policy`. The upstream MIT license covers the copied baseline configuration. See [Diffusion Policy by Chi et al., RSS 2023](https://diffusion-policy.cs.columbia.edu/) and the [official implementation](https://github.com/real-stanford/diffusion_policy).

## Baseline and CPU smoke configuration

`configs/baseline.yaml` is byte-for-byte the [published image PushT CNN configuration](https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml). It retains the ResNet encoder, U-Net widths `[512, 1024, 2048]`, 100 diffusion steps, batch size 64, EMA, 3,050 epochs and 50 test seeds. The dataset split seed is 42 and is independent of the training seed. The local training launcher sets the output path and resume behavior, disables W&B, and keeps Hydra in the repository root.

`configs/smoke_test.yaml` inherits the baseline and uses a CPU, U-Net widths `[32, 64, 128]`, batch size 2, two training batches, one validation batch, and two reverse diffusion steps. It uses two training episodes and one short test rollout and disables EMA and top-k checkpointing.

## From demonstrations to rollout

1. [Dataset and sampler](upstream/diffusion_policy/diffusion_policy/dataset/pusht_image_dataset.py). PushT reads `img`, `state`, `action`, and episode boundaries from the official Zarr replay. It selects 16-step windows within an episode and repeats the boundary frames when padding. A batch contains RGB tensors `[B, 16, 3, 96, 96]` scaled to `[0, 1]`, XY agent positions `[B, 16, 2]`, and XY target actions `[B, 16, 2]`. The upstream validation sampler uses the complement of the capped training mask.
2. [Observation encoder](upstream/diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py). The policy normalizes observations and actions. For the first two observations it crops RGB to 84 × 84, encodes each crop with the robomimic ResNet-18 spatial-softmax encoder, and concatenates the corresponding XY position. The two feature vectors are flattened into one global conditioning vector per sample. Training uses random crops and inference uses a fixed crop.
3. [Conditional 1D U-Net](upstream/diffusion_policy/diffusion_policy/model/diffusion/conditional_unet1d.py). The network convolves over the action trajectory's time axis. Residual blocks use learned scale and bias conditioned on the diffusion timestep embedding and global observation features. The U-Net predicts a noise tensor of shape `[B, 16, 2]`.
4. [Diffusion loss](upstream/diffusion_policy/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py). For each sample, draw a timestep and Gaussian noise, then use the DDPM scheduler to perturb the normalized actions. The loss is the mean squared error between the predicted and sampled noise. AdamW updates the weights and the baseline maintains an EMA copy for rollout.
5. [Rollout](upstream/diffusion_policy/diffusion_policy/env_runner/pusht_image_runner.py). From Gaussian noise, iteratively denoise 16 candidate actions conditioned on the two latest observations. Unnormalize the trajectory and execute eight actions starting at index one. Repeat with new observations. `test/mean_score` averages the maximum goal overlap reached in each test episode.

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

## Baseline training and resume

Run these commands in the CUDA environment when ready to start a full experiment. Keep the project directory bind-mounted to retain output across container restarts.

```bash
./scripts/train.sh experiments/baseline-42 training.seed=42 training.device=cuda:0
./scripts/evaluate.sh experiments/baseline-42/checkpoints/latest.ckpt results/baseline-42 --device cuda:0
```

Use a new directory for every fresh run. With the same profile, seed, device and overrides, use `./scripts/train.sh experiments/baseline-42 --resume training.seed=42 training.device=cuda:0` to load `checkpoints/latest.ckpt` in place and append to the training metrics log. The upstream workspace restores model, EMA, optimizer, and counters, reconstructs the scheduler from the stored step, and treats `training.num_epochs` as the number of epochs for this invocation. Set the remaining budget explicitly when resuming. The checkpoint captures counters before the end-of-epoch increment, so the saved epoch number repeats. The upstream checkpoint does not save RNG, dataloader or top-k manager state. This makes resume a continuation with restored weights and counters, without bitwise equivalence to uninterrupted training. Use only trusted checkpoints because the upstream format uses pickle.

Evaluation reads the rollout profile and seed from the checkpoint and writes to a fresh results directory. For a smoke checkpoint, this evaluates one short episode. A baseline checkpoint evaluates the 50 published test seeds.
