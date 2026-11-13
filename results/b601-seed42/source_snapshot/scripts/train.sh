#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if (( $# < 1 )); then
  echo "Usage: $0 RUN_DIR [--resume] [Hydra overrides...]" >&2
  exit 2
fi
run_dir=$1
shift
resume=false
if [[ ${1:-} == --resume ]]; then
  resume=true
  shift
fi
if [[ ! -f upstream/diffusion_policy/train.py ]]; then
  echo 'Initialize upstream with git submodule update --init.' >&2
  exit 1
fi

if [[ $resume == true ]]; then
  if [[ ! -s "$run_dir/checkpoints/latest.ckpt" ]]; then
    echo "No checkpoint to resume: $run_dir/checkpoints/latest.ckpt" >&2
    exit 1
  fi
else
  mkdir -p -- "$(dirname "$run_dir")"
  if ! mkdir -- "$run_dir"; then
    echo "Choose a new run directory or use --resume: $run_dir" >&2
    exit 1
  fi
fi

lock="$run_dir/.training-lock"
if ! mkdir -- "$lock"; then
  echo "Run is locked: $lock" >&2
  exit 1
fi
trap 'rmdir -- "$lock"' EXIT

export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
export PYTHONPATH="$PWD:$PWD/upstream/diffusion_policy${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
"${PYTHON:-python}" upstream/diffusion_policy/train.py \
  --config-dir "$PWD/configs" --config-name "${PUSHT_CONFIG:-baseline}" \
  "$@" "hydra.run.dir=\"$run_dir\"" hydra.job.chdir=false \
  "training.resume=$resume" logging.mode=disabled
