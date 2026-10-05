# Encoder audit execution runbook

This runbook executes the fixed train/validation diagnostic only. It does not
retrain encoders or classifiers. The audit output is the separate
`results/encoder-audit-v1` tree; keep it outside the baseline, legacy, and EEG
data directories. The source/config/package identity is recorded in every task
and the report. **Freeze all Python source and the config before the first real
task and keep them unchanged through summary generation.** A code or config
change requires a new output directory and a complete rerun.

Use the repository interpreter explicitly (the same value can be supplied via
`AGFL_PYTHON`):

```bash
export AGFL_PYTHON="${AGFL_PYTHON:-.venv/bin/python}"
export AUDIT_CONFIG="configs/encoder-audit.json"
```

## Preflight and independent smoke

Plan is dependency-light and performs no fits:

```bash
"$AGFL_PYTHON" -m inm.encoder_audit --config "$AUDIT_CONFIG" --plan
```

Inspect source and artifact readiness before any real task. A blocked inventory
is a stop condition for real replay; retain its exact historical source mismatch
paths and do not substitute current source files:

```bash
"$AGFL_PYTHON" -m inm.encoder_audit --config "$AUDIT_CONFIG" --inventory
```

Run the synthetic smoke into a fresh temporary directory. It writes a clearly
marked synthetic task and report and must never point at the configured real
output:

```bash
SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agfl-encoder-audit-smoke.XXXXXX")"
"$AGFL_PYTHON" -m inm.encoder_audit --config "$AUDIT_CONFIG" --smoke --output-dir "$SMOKE_DIR"
```

The smoke is CPU-only and uses tiny fixtures. It checks artifact writing and
report structure, not numerical replay or research behavior.

## Real pilot and cohort

Only proceed when the required data and historical replay dependencies have
been independently resolved without editing the original manifests. Run task
index 0 (A01, seed 0) on CPU and inspect its receipt before starting the rest:

```bash
"$AGFL_PYTHON" -m inm.encoder_audit --config "$AUDIT_CONFIG" --task-index 0 --device cpu
cat results/encoder-audit-v1/tasks/A01_seed_0/audit.json
find results/encoder-audit-v1/tasks/A01_seed_0/probes -maxdepth 1 -type f -print
```

The audit saves four fixed CPU logistic fits for the task and replays three
saved baseline neural checkpoints over train and validation. It does not launch
GPU training. Memory use is bounded by one participant/seed's source arrays,
checkpoint batches, and feature caches; tasks run one at a time. Runtime depends
on recording length, machine CPU, and I/O; no wall-time estimate is asserted.

Review task identity, train/validation counts and stable IDs, checks, source and
artifact hashes, replay errors, convergence, per-sample diagnostic files, mask
pairing, and any errors. Confirm no test metrics or rows appear. Do not continue
if the task is failed, incomplete, mismatched, or surprising in a way that
requires source changes. A corrected implementation needs a fresh output path.

After accepting the pilot, explicitly run the remaining 26 subject-major tasks
sequentially. `--task-index` ordering follows subjects 1–9 and seeds 0–2:

```bash
for task_index in $(seq 1 26); do
  "$AGFL_PYTHON" -m inm.encoder_audit --config "$AUDIT_CONFIG" \
    --task-index "$task_index" --device cpu || exit $?
done
```

Safe resume validates each task identity and its recorded output hashes. A
changed config/source or a tampered artifact blocks reuse; remove nothing and
do not replace an incompatible receipt. Fix the issue under a new output identity.

Build the evidence JSON, Markdown summary, and CSV after tasks finish:

```bash
"$AGFL_PYTHON" -m inm.encoder_audit --config "$AUDIT_CONFIG" --summarize-only
cat results/encoder-audit-v1/report/summary.md
```

Partial reports retain task findings and unavailable comparisons but withhold
cohort means. Repeats are aggregated within task, then seeds within participant,
then participants equally. Negative completion effects and probe convergence
failures remain visible. Never select architecture, rank, ridge, preprocessing,
or budget settings from these reused-validation summaries.

## Evidence review

Only after all 27 real tasks are complete, hashes verify, and the report says
`complete`, invoke coding session 08:

```bash
bash scripts/run_encoder_sessions.sh --session 08
```

The session runner independently checks the exact evidence location, real flag,
train/validation-only partitions, nine subjects, three seeds, source byte map,
task identities, task checks, and file hashes. A synthetic smoke or partial
pilot is expected to fail that gate. Session 08 reviews evidence; it does not
start a model search or training run.
