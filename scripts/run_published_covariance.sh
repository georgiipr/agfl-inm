#!/usr/bin/env bash
# Published covariance study launcher. Safe to source for run_bounded().

run_bounded() {
  if (( $# < 3 )); then echo "usage: run_bounded LOG_PREFIX TIMEOUT_SECONDS COMMAND..." >&2; return 2; fi
  local prefix="$1" seconds="$2"; shift 2
  [[ "$seconds" =~ ^[1-9][0-9]*$ ]] || { echo "timeout must be positive integer seconds" >&2; return 2; }
  local parent; parent="$(dirname -- "$prefix")"
  mkdir -p -- "$parent" || return
  # A unique prefix makes every command, stream and exit receipt immutable.
  local cmdfile="${prefix}.command.json" out="${prefix}.stdout" err="${prefix}.stderr" status="${prefix}.exit" started="${prefix}.started.utc" finished="${prefix}.finished.utc"
  if [[ -e "$cmdfile" || -e "$out" || -e "$err" || -e "$status" || -e "$started" || -e "$finished" ]]; then echo "refusing to overwrite launcher receipt: $prefix" >&2; return 2; fi
  python3 -c 'import json,sys; open(sys.argv[1],"x").write(json.dumps(sys.argv[2:])+"\n")' "$cmdfile" "$@" || return
  date -u +%Y-%m-%dT%H:%M:%SZ > "$started" || return
  local rc=0
  timeout --signal=TERM --kill-after=20s "$seconds" setsid bash -c 'trap '\''trap "" TERM; kill -TERM -- -$$; ( sleep 20; kill -KILL -- -$$ ) >/dev/null 2>&1 & wait'\'' TERM; "$@" & child=$!; wait "$child"; rc=$?; exit "$rc"' wrapper "$@" >"$out" 2>"$err" || rc=$?
  date -u +%Y-%m-%dT%H:%M:%SZ > "$finished" || return
  ( umask 077; set -o noclobber; printf '%s\n' "$rc" > "$status" ) || return
  return "$rc"
}

published_covariance_main() {
  [[ $# -ge 1 ]] || { echo "usage: $0 MODE [--output PATH] [--device cpu|cuda:0] [--synthetic]" >&2; return 2; }
  local mode="$1"; shift
  mkdir -p .session-runs
  exec 9>.session-runs/accuracy.lock
  if ! flock -n 9; then echo "another accuracy study holds .session-runs/accuracy.lock" >&2; return 3; fi
  local python="${PUBLISHED_COVARIANCE_PYTHON:-.venv-published-covariance/bin/python}"
  [[ -x "$python" ]] || { echo "runtime executable missing: $python" >&2; return 4; }
  local logroot=".session-runs/published-checkpoint-covariance/launcher"
  mkdir -p "$logroot"
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  if [[ " $* " != *" --synthetic "* && ( "$mode" == "pilot" || "$mode" == "train-cohort" || "$mode" == "reference" || "$mode" == "evaluate-cohort" || "$mode" == "audit" ) ]]; then
    local output="" device="cuda:0" asset_args=() rest=("$@") i=0
    while (( i < ${#rest[@]} )); do
      case "${rest[i]}" in
        --output) output="${rest[i+1]}"; i=$((i+2));;
        --device) device="${rest[i+1]}"; i=$((i+2));;
        --assets) asset_args+=(--assets "${rest[i+1]}"); i=$((i+2));;
        *) i=$((i+1));;
      esac
    done
    [[ -n "$output" ]] || { echo "real pilot/cohort requires --output" >&2; return 2; }
    if [[ "$mode" == "pilot" ]]; then
      for seed in 0 1 2; do
        run_bounded "$logroot/$stamp-A01-s$seed-fit" 3600 "$python" -m published_covariance fit-task --output "$output" --device "$device" "${asset_args[@]}" --subject A01 --seed "$seed" || return $?
        run_bounded "$logroot/$stamp-A01-s$seed-fit-audit" 600 "$python" -m published_covariance audit-fit-task --output "$output" --device "$device" "${asset_args[@]}" --subject A01 --seed "$seed" || return $?
      done
      run_bounded "$logroot/$stamp-pilot-seal" 600 "$python" -m published_covariance mark-pilot --output "$output" --device "$device" "${asset_args[@]}"
      return $?
    fi
    if [[ "$mode" == "reference" ]]; then
      for subject_index in 1 2 3 4 5 6 7 8 9; do
        printf -v subject 'A%02d' "$subject_index"
        run_bounded "$logroot/$stamp-$subject-T-reference" 600 "$python" -m published_covariance reference-task --output "$output" --device "$device" "${asset_args[@]}" --subject "$subject" || return $?
      done
      return 0
    fi
    if [[ "$mode" == "evaluate-cohort" ]]; then
      for subject_index in 1 2 3 4 5 6 7 8 9; do
        printf -v subject 'A%02d' "$subject_index"
        for seed in 0 1 2; do
          run_bounded "$logroot/$stamp-$subject-s$seed-E-evaluate" 600 "$python" -m published_covariance evaluate-task --output "$output" --device "$device" "${asset_args[@]}" --subject "$subject" --seed "$seed" || return $?
          run_bounded "$logroot/$stamp-$subject-s$seed-E-audit" 600 "$python" -m published_covariance audit-evaluation-task --output "$output" --device "$device" "${asset_args[@]}" --subject "$subject" --seed "$seed" || return $?
        done
      done
      run_bounded "$logroot/$stamp-evaluation-aggregate" 600 "$python" -m published_covariance aggregate-evaluation --output "$output" --device "$device" "${asset_args[@]}" || return $?
      return 0
    fi
    if [[ "$mode" == "audit" ]]; then
      run_bounded "$logroot/$stamp-final-audit" 600 "$python" -m published_covariance audit --output "$output" --device "$device" "${asset_args[@]}" || return $?
      return $?
    fi
    for subject_index in 1 2 3 4 5 6 7 8 9; do
      printf -v subject 'A%02d' "$subject_index"
      for seed in 0 1 2; do
        run_bounded "$logroot/$stamp-$subject-s$seed-fit" 3600 "$python" -m published_covariance fit-task --output "$output" --device "$device" "${asset_args[@]}" --subject "$subject" --seed "$seed" || return $?
        run_bounded "$logroot/$stamp-$subject-s$seed-fit-audit" 600 "$python" -m published_covariance audit-fit-task --output "$output" --device "$device" "${asset_args[@]}" --subject "$subject" --seed "$seed" || return $?
      done
    done
    run_bounded "$logroot/$stamp-cohort-seal" 600 "$python" -m published_covariance seal-cohort --output "$output" --device "$device" "${asset_args[@]}"
    return $?
  fi
  local timeout=3600
  case "$mode" in reference|audit) timeout=600;; evaluate-cohort|summarize|review-pilot|preflight) timeout=600;; esac
  run_bounded "$logroot/$stamp-$mode" "$timeout" "$python" -m published_covariance "$mode" "$@"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cd "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit
  published_covariance_main "$@"
fi
