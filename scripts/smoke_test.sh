#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export PUSHT_CONFIG=smoke_test
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-2}
exec scripts/train.sh "$@"
