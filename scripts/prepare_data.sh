#!/usr/bin/env bash
set -euo pipefail
if (( $# > 1 )); then
  echo "Usage: $0 [path/to/pusht.zip]" >&2
  exit 2
fi

if (( $# == 1 )); then
  case "$1" in
    /*) archive=$1 ;;
    *) archive="$PWD/$1" ;;
  esac
fi
cd "$(dirname "$0")/.."

dataset=data/pusht/pusht_cchi_v7_replay.zarr
if [[ -f "$dataset/.zgroup" ]]; then
  printf '%s\n' "$dataset"
  exit 0
fi

mkdir -p data
if [[ -e data/pusht ]]; then
  echo 'data/pusht already exists but is missing the Zarr replay.' >&2
  exit 1
fi

staging=$(mktemp -d data/.pusht.XXXXXXXX)
trap 'rm -rf -- "$staging"' EXIT
if (( $# == 0 )); then
  archive="$staging/pusht.zip"
  curl --fail --location --silent --show-error --retry 3 \
    https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip \
    --output "$archive"
fi

expected=63d52a114a3f010861f0181309d165b7d69133ccae426ece2fc94caed147bdf9
if command -v sha256sum >/dev/null; then
  actual=$(sha256sum < "$archive")
else
  actual=$(shasum -a 256 < "$archive")
fi
if [[ ${actual%% *} != "$expected" ]]; then
  echo "PushT archive SHA-256 mismatch: $archive" >&2
  exit 1
fi

unzip -q "$archive" -d "$staging"
for entry in .zgroup data/img/.zarray data/state/.zarray data/action/.zarray meta/episode_ends/.zarray; do
  if [[ ! -f "$staging/${dataset#data/}/$entry" ]]; then
    echo "Incomplete PushT archive: missing $entry" >&2
    exit 1
  fi
done
mv "$staging/pusht" data/
printf '%s\n' "$dataset"
