#!/usr/bin/env bash
# Use the project environment from any working directory; real tasks default CUDA.
set -Eeuo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${AGFL_PYTHON:-"$repo_root/.venv/bin/python"}
command -v "$python_bin" >/dev/null || { printf 'Python not found: %s\nSet AGFL_PYTHON to a prepared interpreter.\n' "$python_bin" >&2; exit 1; }
# Resolve relative executable overrides before switching to the repository.
python_bin=$("$python_bin" -c 'import sys; print(sys.executable)')
cd -- "$repo_root"
export AGFL_PYTHON="$python_bin"
if [[ ${1:-} == --check-cuda ]]; then
    [[ $# == 1 ]] || { printf '%s\n' '--check-cuda is a standalone operation.' >&2; exit 2; }
    exec "$python_bin" "$script_dir/check_cuda.py"
fi
if [[ $# == 0 ]]; then
    printf '%s\n' 'Usage: bash scripts/run_baselines.sh --check-cuda | BASELINE_CLI_OPTIONS' \
        'Uses AGFL_PYTHON or the repo .venv. Real tasks default to CUDA; --smoke uses CPU.'
    exit 0
fi
# Let the CLI validate its options. Probe CUDA before valid real-task requests so
# a sandbox/driver failure is diagnosed before any experiment artifacts are made.
args=("$@")
device=cuda
real_task=0
help=0
for ((i=0; i<${#args[@]}; i++)); do
    case ${args[i]} in
        --task-index|--task-index=*) real_task=1 ;;
        --device) device=${args[i+1]:-} ;;
        --device=*) device=${args[i]#--device=} ;;
        --help|-h) help=1 ;;
    esac
done
if [[ $real_task == 1 && $device == cuda && $help == 0 ]]; then
    "$python_bin" "$script_dir/check_cuda.py"
fi
exec "$python_bin" -m inm.baselines "${args[@]}"
