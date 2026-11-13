#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${GENESIS_PYTHON:-}" ]]; then
  GENESIS_RUNTIME="$GENESIS_PYTHON"
elif [[ -x .venv/bin/python ]]; then
  GENESIS_RUNTIME="$PWD/.venv/bin/python"
else
  GENESIS_RUNTIME="$(command -v python3 || true)"
fi
if [[ -z "$GENESIS_RUNTIME" ]] || ! command -v "$GENESIS_RUNTIME" >/dev/null 2>&1; then
  printf '%s\n' 'Python was not found. Set GENESIS_PYTHON to a Python executable with Genesis and CUDA PyTorch installed.' >&2
  exit 1
fi
# With no arguments, preserve the checked-in reference run.
if [[ $# -eq 0 ]]; then
  set -- --out output/reproduced
fi
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
exec "$GENESIS_RUNTIME" -u run.py "$@"
