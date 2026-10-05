# Encoder candidate runbook

This is a new, full-input supervised study. It compares `local_power` with its
fresh `local_control`, and compares `spatial_filterbank` and
`spatial_transformer` with the fresh `spatial_eegnet` reference. The fixed
matrix is 27 subject/seed tasks, five neural fits per task (135 total), and four
CPU probes per task (108 total). There are no test scores in this study's
outputs or report.

## Before execution

Use the declared environment and verify it without loading EEG:

```bash
export AGFL_PYTHON=/path/to/the/project/python
$AGFL_PYTHON -m inm.encoder_candidates --config configs/encoder-candidates.json --plan
$AGFL_PYTHON -m inm.encoder_candidates --config configs/encoder-candidates.json --preflight
```

The plan must show 27 tasks, 135 neural fits, 108 CPU probe fits, and zero
Tucker fits. Preflight inventories the nine `AxxT.gdf` recordings, persisted
split files, metadata and numerical packages. It does not check file contents,
recording checksums, or execute a model. Resolve every unavailable or conflicting
input before a real task; do not regenerate persisted split membership.

## Synthetic CPU smoke

Choose a new empty path outside the repository, recording data, split inputs,
and real output. For example:

```bash
SMOKE_DIR="$(mktemp -d /tmp/agfl-encoder-candidate-smoke.XXXXXX)"
rmdir "$SMOKE_DIR"
$AGFL_PYTHON -m inm.encoder_candidates --config configs/encoder-candidates.json \
  --smoke --device cpu --output-dir "$SMOKE_DIR"
```

The smoke fits all five arms on synthetic data with a tiny training budget,
checks reload, hidden-NaN masking, probes, and paired masks, then writes a
partial report. Its task is explicitly synthetic and cannot enter
`evidence.json` as real evidence. It is not an accuracy result.

## One real pilot, then the remaining tasks

Freeze this checkout, the raw config bytes, and package versions before starting.
Use a new output path for any source, config, or package change. A source change
also changes the declared study identity. Old tensor-study checkpoint/source
mismatches do not authorize changing historical artifacts: this candidate study
trains new models from scratch, does not consume those checkpoints, and records
its own source and input identities.

Run exactly one task first: task index 0 is A01, seed 0. It performs five neural
fits and four CPU probe fits. Real commands are explicit and are never run by
the coding-session workflow:

```bash
$AGFL_PYTHON -m inm.encoder_candidates --config configs/encoder-candidates.json \
  --task-index 0 --device cpu
```

Inspect `results/encoder-candidates-v1/tasks/A01_seed_0/task.json`,
`data_provenance.json`, all five `ARMS/<arm>/result.json` files, selected
checkpoints, histories, prediction/probe NPZ files, validation mask hashes,
training-only normalization, selected epochs, parameter counts, convergence,
and failure journals. Check that the task has status `complete`, all seven
verification checks are true, all validation IDs and mask hashes match across
arms, and all artifact digests verify. Stop on any disagreement; preserve that
output and diagnose the input or implementation issue before using a new output
identity.

After the pilot has been reviewed, execute the other 26 tasks sequentially. Each
index is one complete subject/seed unit:

```bash
for task_index in $(seq 1 26); do
  $AGFL_PYTHON -m inm.encoder_candidates --config configs/encoder-candidates.json \
    --task-index "$task_index" --device cpu || break
done
```

The task order is subject-major, with seeds 0, 1, 2 for each participant. A
failed or partial directory is preserved and cannot be resumed as success. Set
`output_dir` in a copied config to a fresh, non-overlapping location before
restarting. Do not rerun a task into incompatible results.

## Summarize and review

Once all tasks are present, independently validate and aggregate them:

```bash
$AGFL_PYTHON -m inm.encoder_candidates --config configs/encoder-candidates.json \
  --summarize-only
```

Exit status 0 means the complete 27-task real evidence gate passed. Exit status
1 means a partial report was written; inspect `report/issues.csv` and
`report/summary.md`. The summarizer writes `evidence.json`, Markdown, and CSVs
for all three paired contrasts, participant/seed scores, training gaps, ordered
and shuffled local probes, and validation robustness. It checks task/fit
identity, exact arm and condition coverage, paired validation IDs and masks,
selected history epoch, numeric metrics, and every declared checkpoint/history/
NPZ checksum. It rejects synthetic tasks and suppresses cohort means unless the
entire real cohort is valid.

For each contrast, the candidate-minus-control difference is averaged across
seeds within participant, then the nine participants are weighted equally.
Log-loss gains use control loss minus candidate loss, so positive values favor
the candidate. The exploratory 95% intervals resample participants in paired
draws (2000 repetitions, seed 20261003). They are descriptive intervals in a
small validation cohort; three comparisons are reported, and validation also
selects epochs. They do not establish confirmation. The predeclared screening
rule marks a candidate only when the complete cohort mean BA gain is at least
0.02 and at least six participant BA differences are positive. Report negative
and mixed results as observed; a screen is a reason to seek independent
confirmation, not a claim of statistical significance or strong BCI performance.

Session 10 reviews `results/encoder-candidates-v1/report/evidence.json` and the
underlying task artifacts. The evidence manifest certifies software-level
completeness and identity checks, not independent scientific truth.
