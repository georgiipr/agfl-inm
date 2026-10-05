#!/usr/bin/env bash
# Explicit scientific execution; all model fitting is CPU deterministic, sequential.
set -Eeuo pipefail
usage() {
    cat <<'HELP'
Usage: bash scripts/run_tensor_temporal_experiments.sh MODE [options]
  --pilot             Train A01/seed0, five fixed arms
  --cohort            Verify completed pilot, then train A02–A09/seed0 and summarize
  --summarize         Verify and recompute existing evidence without fitting
  --config PATH       Default configs/tensor-temporal-screen.json
  --python PATH       Default AGFL_PYTHON, then repo .venv/bin/python
  --dry-run           Print commands without fitting or filesystem writes
  --help              Show help
Only real configurations run through this script. Synthetic fixtures use the Python CLI.
Incomplete or incompatible fit artifacts stop execution and are never repaired.
HELP
}
die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n $2 && $2 != --* ]] || die "$1 requires a value"; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${AGFL_PYTHON:-"$repo_root/.venv/bin/python"}
config="$repo_root/configs/tensor-temporal-screen.json"
mode=
dry_run=0
while (($#)); do
    case "$1" in
        --pilot|--cohort|--summarize)
            [[ -z "$mode" ]] || die 'Choose exactly one mode'
            mode="$1"; shift ;;
        --dry-run) dry_run=1; shift ;;
        --config) need_value "$@"; config="$2"; shift 2 ;;
        --python) need_value "$@"; python_bin="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
[[ -n "$mode" ]] || { usage; exit 0; }
command -v "$python_bin" >/dev/null || die "Python not found: $python_bin"
cd -- "$repo_root"
base=("$python_bin" -m inm.tensor_temporal --config "$config")
run() {
    if ((dry_run)); then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
if ((!dry_run)); then
    mkdir -p .session-runs
    exec 9> .session-runs/accuracy.lock
    flock -n 9 || die 'Another session or experiment runner is using this workspace'
fi
if [[ "$mode" == --pilot ]]; then
    run "${base[@]}" --task-index 0
elif [[ "$mode" == --cohort ]]; then
    run "${base[@]}" --verify-pilot
    for task in 1 2 3 4 5 6 7 8; do run "${base[@]}" --task-index "$task"; done
    run "${base[@]}" --summarize-only
else
    run "${base[@]}" --summarize-only
fi
