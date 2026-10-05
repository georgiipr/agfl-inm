#!/usr/bin/env bash
# Explicit scientific execution, separate from Codex coding/review sessions.
set -Eeuo pipefail
usage() {
    cat <<'HELP'
Usage: bash scripts/run_candidate_experiments.sh MODE [options]
  --pilot             Train only A01/seed0: five neural fits and four probes
  --cohort            Run tasks 1–26 sequentially after verified pilot review
  --summarize         Validate/aggregate existing outputs without training
  --device cpu|cuda   Default cpu; device must have readiness evidence
  --python PATH       Default AGFL_PYTHON, then repo .venv/bin/python
  --state-dir PATH    Default .session-runs/candidate-followup
  --dry-run           Print commands; no models, training or filesystem writes
  --help              Show help
This script launches real training only with explicit --pilot or --cohort.
It never calls Codex, changes configs, retries failures, or repairs artifacts.
HELP
}
die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n $2 && $2 != --* ]] || die "$1 requires a value"; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
python_bin=${AGFL_PYTHON:-"$repo_root/.venv/bin/python"}
state_dir=.session-runs/candidate-followup
phase=
device=cpu
dry_run=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --pilot|--cohort|--summarize) [[ -z $phase ]] || die 'Choose exactly one mode'; phase=${1#--}; shift ;;
        --device) need_value "$@"; device=$2; shift 2 ;;
        --python) need_value "$@"; python_bin=$2; shift 2 ;;
        --state-dir) need_value "$@"; state_dir=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
[[ -n $phase ]] || { usage; exit 0; }
[[ $device == cpu || $device == cuda ]] || die 'Device must be cpu or cuda'
command -v "$python_bin" >/dev/null || die "Python not found: $python_bin"
python_bin=$("$python_bin" -c 'import sys; print(sys.executable)')
[[ $state_dir = /* ]] || state_dir="$repo_root/$state_dir"
state_dir=$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$state_dir")
helper="$script_dir/followup_session_state.py"
config="$repo_root/configs/encoder-candidates-reproducible.json"
base=("$python_bin" -m inm.encoder_candidates --config "$config")
gate_command=("$python_bin" "$helper" execution-gate --root "$repo_root" --state "$state_dir" --session 02 --phase "$phase" --device "$device")
print_command() { printf '%q ' "$@"; printf '\n'; }
if [[ $dry_run == 1 ]]; then
    print_command "${gate_command[@]}"
    case $phase in
        pilot) print_command "${base[@]}" --task-index 0 --device "$device" ;;
        cohort)
            for ((i=1;i<27;i++)); do print_command "${base[@]}" --task-index "$i" --device "$device"; done
            print_command "${base[@]}" --summarize-only ;;
        summarize) print_command "${base[@]}" --summarize-only ;;
    esac
    exit 0
fi
cd -- "$repo_root"
mkdir -p .session-runs
exec 9> .session-runs/accuracy.lock
flock -n 9 || die 'Another session or experiment runner is using this workspace'
export AGFL_PYTHON="$python_bin"
if [[ $device == cuda ]]; then export CUBLAS_WORKSPACE_CONFIG=:4096:8; fi
"${gate_command[@]}"
if [[ $phase != summarize && $device == cuda ]]; then
    "$python_bin" "$script_dir/check_cuda.py"
fi
mkdir -p "$state_dir/experiments"
run_logged() {
    local tag=$1
    shift
    local log
    log=$(mktemp "$state_dir/experiments/$tag-XXXXXX.log")
    printf 'Running %s. Log: %s\n' "$tag" "$log"
    if "$@" > "$log" 2>&1; then
        tail -n 8 "$log"
    else
        local status=$?
        tail -n 30 "$log" >&2
        printf 'Stopped (%s). Preserve artifacts; do not retry incompatible/partial outputs.\n' "$status" >&2
        exit "$status"
    fi
}
case $phase in
    pilot)
        run_logged pilot "${base[@]}" --task-index 0 --device "$device"
        "$python_bin" "$helper" gate --root "$repo_root" --session 03
        printf 'Pilot finished. Run follow-up session 03 to review before the cohort.\n' ;;
    cohort)
        for ((i=1;i<27;i++)); do
            # Detect source/config/package changes before each next task.
            "${gate_command[@]}"
            run_logged "task-$i" "${base[@]}" --task-index "$i" --device "$device"
        done
        run_logged summary "${base[@]}" --summarize-only
        "$python_bin" "$helper" gate --root "$repo_root" --session 04
        printf 'Cohort evidence verified. Run follow-up session 04 for verdicts.\n' ;;
    summarize) run_logged summary "${base[@]}" --summarize-only ;;
esac
