#!/usr/bin/env bash
# Explicit, sequential CUDA fitting with timeouts, fresh logs and independent audits.
set -Eeuo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${AGFL_PYTHON:-"$repo_root/.venv/bin/python"}
config=; mode=; smoke_output=; dry=0
die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
usage() {
  cat <<'HELP'
Usage: bash scripts/run_completion_experiments.sh MODE --config FILE [options]
Modes: --plan --preflight --smoke --pilot --cohort --audit --summarize
Options: --python PATH  --output-dir FRESH_SMOKE_DIRECTORY  --dry-run
Pilot fits A01/seed0 and independently audits it before recording acceptance.
Cohort requires that receipt, then fits/audits tasks 1-26 and produces the report.
Audit replays all 27 selected models without fitting. Smoke uses synthetic data.
Requires GNU timeout and flock for execution. No automatic retry or artifact repair.
See docs/task-driven-completion-cuda/reproduce.md for input and environment setup.
HELP
}
while (( $# )); do
  case $1 in
    --plan|--preflight|--smoke|--pilot|--cohort|--audit|--summarize) [[ -z $mode ]] || die 'Choose one mode'; mode=${1#--}; shift ;;
    --config|--python|--output-dir)
      [[ $# -ge 2 && -n $2 && $2 != --* ]] || die "$1 requires a value"
      case $1 in --config) config=$2;; --python) python_bin=$2;; --output-dir) smoke_output=$2;; esac; shift 2 ;;
    --dry-run) dry=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done
[[ -n $mode && -n $config ]] || { usage; exit 2; }
[[ $mode == smoke || -z $smoke_output ]] || die '--output-dir is for synthetic smoke only'
[[ $mode != smoke || -n $smoke_output ]] || die 'Smoke requires --output-dir'
cd -- "$repo_root"
[[ -f $config ]] || die 'Configuration file is missing'
command -v "$python_bin" >/dev/null || die 'Python not found; set AGFL_PYTHON or --python'
config=$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$config")
base=("$python_bin" -m inm.task_driven_completion_cuda --config "$config")
audit=("$python_bin" -m inm.task_driven_completion_cuda.audit --config "$config")
helper=("$python_bin" "$repo_root/scripts/completion_bundle.py")
print_command() { printf '%q ' "$@"; printf '\n'; }
if (( dry )); then
  case $mode in
    plan|preflight) print_command "${base[@]}" "--$mode" ;;
    smoke) print_command "${base[@]}" --smoke --output-dir "$smoke_output" ;;
    pilot) print_command "${base[@]}" --task-index 0; print_command "${audit[@]}" --task-index 0 ;;
    cohort) for ((i=1;i<27;i++)); do print_command "${base[@]}" --task-index "$i"; print_command "${audit[@]}" --task-index "$i"; done; print_command "${base[@]}" --summarize-only ;;
    audit) for ((i=0;i<27;i++)); do print_command "${audit[@]}" --task-index "$i"; done ;;
    summarize) print_command "${base[@]}" --summarize-only ;;
  esac
  exit 0
fi
if [[ $mode == plan ]]; then exec "${base[@]}" --plan; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ $mode == preflight ]]; then
  "${helper[@]}" check-env --config "$config"
  exec "${base[@]}" --preflight
fi
command -v timeout >/dev/null || die 'GNU timeout is required'
command -v flock >/dev/null || die 'flock is required'
key=$("$python_bin" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()[:20])' "$config")
state="$repo_root/.session-runs/completion-reproduction/$key"
mkdir -p "$state"
exec 9> "$repo_root/.session-runs/completion-reproduction/runner.lock"
flock -n 9 || die 'Another completion reproduction runner is active'
active=
cleanup() { if [[ -n $active ]]; then kill -TERM "$active" 2>/dev/null || true; wait "$active" 2>/dev/null || true; fi; }
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
run_logged() {
  local tag=$1 seconds=$2; shift 2
  [[ ! -e $state/$tag.stdout.log && ! -e $state/$tag.stderr.log ]] || die "Attempt already exists: $tag; preserve it and use a fresh identity after failure"
  printf 'Running %s; logs: %s/%s.*.log\n' "$tag" "$state" "$tag"
  timeout --signal=TERM --kill-after=15s "$seconds" "$@" > "$state/$tag.stdout.log" 2> "$state/$tag.stderr.log" &
  active=$!
  local rc=0
  wait "$active" || rc=$?
  active=
  printf '%s\n' "$rc" > "$state/$tag.exit-code"
  if (( rc )); then tail -n 15 "$state/$tag.stderr.log" >&2; die "Stopped at $tag (exit $rc); no automatic retry"; fi
}
if [[ $mode == pilot || $mode == cohort ]]; then
  "${helper[@]}" check-run --config "$config"
elif [[ $mode != smoke ]]; then
  "${helper[@]}" check-env --config "$config"
fi
case $mode in
  smoke)
    run_logged smoke 1200 "${base[@]}" --smoke --output-dir "$smoke_output"
    run_logged smoke-audit 1200 "${audit[@]}" --synthetic-smoke --output-dir "$smoke_output" ;;
  pilot)
    run_logged pilot-fit 7200 "${base[@]}" --task-index 0
    run_logged pilot-audit 1200 "${audit[@]}" --task-index 0
    "${helper[@]}" accept-pilot --config "$config" --state "$state" ;;
  cohort)
    "${helper[@]}" check-pilot --config "$config" --state "$state"
    for ((i=1;i<27;i++)); do
      run_logged "task-$i-fit" 7200 "${base[@]}" --task-index "$i"
      run_logged "task-$i-audit" 1200 "${audit[@]}" --task-index "$i"
    done
    run_logged report 1200 "${base[@]}" --summarize-only ;;
  audit) for ((i=0;i<27;i++)); do run_logged "replay-$i" 1200 "${audit[@]}" --task-index "$i"; done ;;
  summarize) run_logged report 1200 "${base[@]}" --summarize-only ;;
esac
printf 'Completed %s. Receipts: %s\n' "$mode" "$state"
