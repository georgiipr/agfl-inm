#!/usr/bin/env bash
# Run one fresh Codex conversation per bounded task; advance only after checks.
set -Eeuo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/run_accuracy_sessions.sh [options]
  --list                List all session plans; no model call
  --dry-run             Show selected commands; no calls or filesystem changes
  --from ID             First session to run (default 01); predecessors required
  --through ID          Last session (default 10; 11–12 require real pilot data)
  --model NAME          Codex model (default gpt-6-luna, or AGFL_SESSION_MODEL)
  --effort LEVEL        low, medium, high, xhigh (default high)
  --python PATH         Python (default AGFL_PYTHON, repo .venv, then python3)
  --codex PATH          Codex executable (default codex / AGFL_CODEX_BIN)
  --state-dir PATH      Logs/receipts (default .session-runs/accuracy, relative to repo)
  --timeout SECONDS     Per model session and acceptance phase (default 1800)
  --help                Show this help
Completed sessions are automatically rechecked and skipped. No auto-retries.
HELP
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n $2 && $2 != --* ]] || die "$1 requires a value"; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
helper="$script_dir/accuracy_session_state.py"
model=${AGFL_SESSION_MODEL:-gpt-6-luna}
effort=high
if [[ -n ${AGFL_PYTHON:-} ]]; then
    python_bin=$AGFL_PYTHON
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    python_bin="$repo_root/.venv/bin/python"
else
    python_bin=python3
fi
codex_bin=${AGFL_CODEX_BIN:-codex}
first=01
last=
state_dir=.session-runs/accuracy
timeout_seconds=1800
list_only=0
dry_run=0
active_pid=

while [[ $# -gt 0 ]]; do
    case $1 in
        --list) list_only=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --from) need_value "$@"; first=$2; shift 2 ;;
        --through) need_value "$@"; last=$2; shift 2 ;;
        --model) need_value "$@"; model=$2; shift 2 ;;
        --effort) need_value "$@"; effort=$2; shift 2 ;;
        --python) need_value "$@"; python_bin=$2; shift 2 ;;
        --codex) need_value "$@"; codex_bin=$2; shift 2 ;;
        --state-dir) need_value "$@"; state_dir=$2; shift 2 ;;
        --timeout) need_value "$@"; timeout_seconds=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

command -v "$python_bin" >/dev/null || die "Python not found: $python_bin"
python_bin=$(command -v "$python_bin")
# Resolve relative executable paths before changing into the repository.
python_bin=$("$python_bin" -c 'import sys; print(sys.executable)')
[[ $effort =~ ^(low|medium|high|xhigh)$ ]] || die "Invalid reasoning effort: $effort"
[[ $timeout_seconds =~ ^[1-9][0-9]*$ ]] || die "Timeout must be a positive integer"
[[ $state_dir = /* ]] || state_dir="$repo_root/$state_dir"
state_dir=$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$state_dir")

if [[ $list_only == 1 ]]; then
    "$python_bin" "$helper" list --root "$repo_root"
    exit 0
fi
[[ -n $last ]] || last=$("$python_bin" "$helper" default --root "$repo_root")
[[ $first =~ ^[0-9]{1,2}$ && $last =~ ^[0-9]{1,2}$ ]] || die "Session IDs must be numbers 01–12"
printf -v first '%02d' "$((10#$first))"
printf -v last '%02d' "$((10#$last))"
ids_text=$("$python_bin" "$helper" ids --root "$repo_root")
mapfile -t session_ids <<< "$ids_text"
[[ " ${session_ids[*]} " == *" $first "* && " ${session_ids[*]} " == *" $last "* ]] || die "Unknown session range"
[[ $first < $last || $first == "$last" ]] || die "--from must not exceed --through"

make_command() {
    local report=$1
    session_command=("$codex_bin" --ask-for-approval never exec
        --cd "$repo_root" --sandbox workspace-write --model "$model"
        --config "model_reasoning_effort=\"$effort\"" --color never --json
        --output-schema "$repo_root/plans/accuracy/result.schema.json"
        --output-last-message "$report" -)
}

if [[ $dry_run == 1 ]]; then
    printf 'Workspace: %s\nModel: %s (%s)\nState: %s\n' "$repo_root" "$model" "$effort" "$state_dir"
    for sid in "${session_ids[@]}"; do
        [[ $sid < $first || $sid > $last ]] && continue
        make_command "$state_dir/<attempt-$sid>/result.json"
        printf 'Session %s: ' "$sid"
        printf '%q ' timeout --kill-after=30s "${timeout_seconds}s" "${session_command[@]}"
        printf '< prompt.txt\n'
    done
    printf 'Real execution requires verified predecessors and runs independent acceptance checks.\n'
    exit 0
fi

"$python_bin" -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ required"'
for executable in git flock timeout "$codex_bin"; do
    command -v "$executable" >/dev/null || die "Required executable not found: $executable"
done
codex_bin=$(command -v "$codex_bin")
codex_bin=$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).absolute())' "$codex_bin")
cd -- "$repo_root"
git rev-parse --show-toplevel >/dev/null
mkdir -p -- "$state_dir"
# All state directories share a workspace lock, preventing two runs in this tree.
mkdir -p -- "$repo_root/.session-runs"
exec 9> "$repo_root/.session-runs/accuracy.lock"
flock -n 9 || die "Another accuracy-session runner is using this workspace"
export AGFL_PYTHON="$python_bin"

interrupted() {
    trap - INT TERM
    if [[ -n $active_pid ]]; then
        kill -TERM "$active_pid" 2>/dev/null || true
        wait "$active_pid" 2>/dev/null || true
    fi
    printf '\nInterrupted. Partial edits and attempt logs are preserved.\n' >&2
    exit 130
}
trap interrupted INT TERM

for sid in "${session_ids[@]}"; do
    [[ $sid > $last ]] && break
    if [[ -f "$state_dir/completed/$sid.json" ]]; then
        "$python_bin" "$helper" check-receipt --root "$repo_root" --state "$state_dir" --session "$sid"
        printf 'Rechecking completed session %s...\n' "$sid"
        if ! timeout --kill-after=30s "${timeout_seconds}s" "$python_bin" "$helper" verify \
            --root "$repo_root" --session "$sid" > "$state_dir/recheck-$sid.log" 2>&1; then
            cat "$state_dir/recheck-$sid.log" >&2
            die "Session $sid no longer passes. Repair the regression before resuming."
        fi
        continue
    fi
    [[ $sid < $first ]] && die "Session $sid is unfinished; --from cannot skip prerequisites"
    attempt=$(mktemp -d "$state_dir/$sid-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
    "$python_bin" "$helper" prompt --root "$repo_root" --state "$state_dir" --session "$sid" > "$attempt/prompt.txt"
    expected_fingerprint=$("$python_bin" "$helper" fingerprint --root "$repo_root" --session "$sid")
    git status --short --untracked-files=all > "$attempt/before.status"
    git diff --binary > "$attempt/before.patch"
    make_command "$attempt/result.json"
    printf 'Starting session %s with %s. Logs: %s\n' "$sid" "$model" "$attempt"
    timeout --kill-after=30s "${timeout_seconds}s" "${session_command[@]}" \
        < "$attempt/prompt.txt" > "$attempt/events.jsonl" 2> "$attempt/stderr.log" &
    active_pid=$!
    exit_code=0
    wait "$active_pid" || exit_code=$?
    active_pid=
    git status --short --untracked-files=all > "$attempt/after.status"
    if [[ $exit_code != 0 ]]; then
        tail -n 30 "$attempt/stderr.log" >&2
        die "Session $sid exited $exit_code (124 usually means timeout). No completion receipt; inspect $attempt"
    fi
    actual_fingerprint=$("$python_bin" "$helper" fingerprint --root "$repo_root" --session "$sid")
    [[ $actual_fingerprint == "$expected_fingerprint" ]] || die "Session $sid modified its plan or runner; review before continuing"
    "$python_bin" "$helper" validate-report --root "$repo_root" --session "$sid" --report "$attempt/result.json"
    printf 'Checking session %s independently...\n' "$sid"
    if ! timeout --kill-after=30s "${timeout_seconds}s" "$python_bin" "$helper" verify \
        --root "$repo_root" --session "$sid" > "$attempt/verification.log" 2>&1; then
        cat "$attempt/verification.log" >&2
        die "Session $sid failed acceptance checks. No receipt; inspect $attempt"
    fi
    actual_fingerprint=$("$python_bin" "$helper" fingerprint --root "$repo_root" --session "$sid")
    [[ $actual_fingerprint == "$expected_fingerprint" ]] || die "Acceptance execution modified the plan or runner; review session $sid"
    "$python_bin" "$helper" receipt --root "$repo_root" --state "$state_dir" --session "$sid" \
        --report "$attempt/result.json" --model "$model" --effort "$effort"
    printf 'Session %s completed and verified.\n' "$sid"
done
printf 'Requested sessions through %s are complete. Real EEG experiments remain a separate runbook step.\n' "$last"
