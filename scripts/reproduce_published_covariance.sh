#!/usr/bin/env bash
# Staged reproduction helper for the audited published-checkpoint study.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/reproduce_published_covariance.sh prepare-assets
  scripts/reproduce_published_covariance.sh plan
  scripts/reproduce_published_covariance.sh STAGE results/published-covariance-friend-NAME

STAGE is one of: preflight, smoke, reference, pilot, review-pilot,
train-cohort, evaluate-cohort, audit, summarize, verify.
Use a new output directory beginning with published-covariance-friend-.
EOF
  exit 2
}

[[ $# -ge 1 ]] || usage
stage="$1"
shift
asset_root="$ROOT/.session-runs/published-checkpoint-covariance/assets"
output="${1:-}"
python="${PUBLISHED_COVARIANCE_PYTHON:-$ROOT/.venv-published-covariance/bin/python}"

if [[ "$stage" == "prepare-assets" ]]; then
  source_dir="$asset_root/EEG-ATCNet"
  revision="65162fb359ea46a2f62c885a9987247ab491ee7a"
  mkdir -p "$asset_root"
  if [[ ! -d "$source_dir/.git" ]]; then
    git clone --no-checkout https://github.com/Altaheri/EEG-ATCNet.git "$source_dir"
  fi
  actual_revision="$(git -C "$source_dir" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$actual_revision" != "$revision" ]]; then
    git -C "$source_dir" checkout --detach "$revision"
  fi
  git -C "$source_dir" checkout --detach "$revision"
  mkdir -p "$source_dir/results/saved models/run-1" "$asset_root/recordings"
  for subject in {1..9}; do
    printf -v subject_name 'A%02d' "$subject"
    cp "$ROOT/docs/published-covariance/checkpoints/${subject_name}.h5" \
      "$source_dir/results/saved models/run-1/subject-${subject}.h5"
  done
  cp "$ROOT/docs/published-covariance/checkpoints-origin.json" "$asset_root/origin.json"
  python3 "$ROOT/plans/published-checkpoint-covariance/fetch_recordings.py"
  (
    cd "$asset_root/recordings"
    sha256sum --check "$ROOT/docs/published-covariance/recordings-sha256.txt"
  )
  alignment="$ROOT/.session-runs/published-checkpoint-covariance/02/native-recording-alignment.json"
  mkdir -p "$(dirname "$alignment")"
  cp "$ROOT/docs/published-covariance/native-recording-alignment.json" "$alignment"
  echo "Prepared pinned source, nine audited checkpoints and verified recordings."
  exit 0
fi

if [[ "$stage" == "plan" ]]; then
  python3 -m published_covariance plan
  exit 0
fi

[[ -n "$output" ]] || usage
case "$output" in
  results/published-covariance-friend-*) ;;
  *) echo "Output must be a fresh results/published-covariance-friend-* path." >&2; exit 2 ;;
esac
[[ -x "$python" ]] || { echo "Missing runtime: $python" >&2; exit 2; }
[[ -d "$asset_root/EEG-ATCNet" && -f "$asset_root/origin.json" ]] || {
  echo "Run prepare-assets first." >&2; exit 2;
}

cuda_root="$ROOT/.venv-published-covariance/lib/python3.11/site-packages/nvidia/cuda_nvcc"
if [[ -d "$cuda_root" ]]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=$cuda_root"
  export PATH="$cuda_root/bin:$PATH"
fi
export PUBLISHED_COVARIANCE_PYTHON="$python"

case "$stage" in
  preflight) "$python" -m published_covariance preflight --device cuda:0 ;;
  smoke) "$ROOT/scripts/run_published_covariance.sh" smoke --native --device cuda:0 --output "$output" ;;
  reference|pilot|review-pilot|train-cohort|evaluate-cohort|audit|summarize)
    "$ROOT/scripts/run_published_covariance.sh" "$stage" --output "$output" --device cuda:0 ;;
  verify) "$python" "$ROOT/plans/published-checkpoint-covariance/verify_real_report.py" "$output" ;;
  *) usage ;;
esac
