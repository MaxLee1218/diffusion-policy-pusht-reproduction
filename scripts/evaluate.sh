#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f upstream/diffusion_policy/train.py ]]; then
  echo 'Initialize upstream with git submodule update --init.' >&2
  exit 1
fi

export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export SDL_AUDIODRIVER=${SDL_AUDIODRIVER:-dummy}
export PYTHONPATH="$PWD/upstream/diffusion_policy${PYTHONPATH:+:$PYTHONPATH}"
if [[ ${1:-} == horizons ]]; then
  shift
  exec "${PYTHON:-python}" scripts/evaluate_horizons.py "$@"
fi
exec "${PYTHON:-python}" scripts/evaluate.py "$@"
