#!/usr/bin/env bash
# Verify the shared numerical evidence without EEG recordings or CUDA.
set -Eeuo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${AGFL_PYTHON:-"$repo_root/.venv/bin/python"}
if [[ ! -x $python_bin ]]; then python_bin=python3; fi
mode=verify
case ${1:---verify} in
  --verify) [[ $# == 0 ]] || shift ;;
  --unpack) mode=unpack; shift ;;
  --help|-h) printf '%s\n' 'Usage: bash scripts/show_completion_results.sh [--verify|--unpack]' 'Set AGFL_PYTHON to choose Python. Verification needs NumPy only; no CUDA or GDF files.'; exit 0 ;;
  *) printf 'Unknown mode: %s\n' "$1" >&2; exit 2 ;;
esac
exec "$python_bin" "$repo_root/scripts/completion_bundle.py" "$mode" "$@"
