# Encoder candidate protocol declaration

This package declares a new supervised study, separate from the legacy tensor
experiment, the baseline accuracy study, and the encoder audit. No candidate
fits or measurements have run. The five ordered arms are `local_control`,
`local_power`, `spatial_eegnet`, `spatial_filterbank`, and
`spatial_transformer`. The cohort is A01–A09 with seeds 0, 1, and 2: 27 tasks,
135 neural fits, 108 ordered/shuffled CPU probe fits, and zero Tucker fits.

The machine-readable declaration is
[`configs/encoder-candidates.json`](../../configs/encoder-candidates.json).
`load_config(path)` in `inm.encoder_candidates.protocol` rejects unknown keys
and settings, altered arm order or IDs, test exposure, non-full training,
search, real-config budget changes, duplicate cohort IDs, and overlapping
output/input paths.
Relative paths are resolved against the config file after symlink resolution.
Outputs must use a fresh directory. The input split and recording locations are
read-only.

The checked-in config has `synthetic: false`. A test fixture may set
`synthetic: true` and use a subset of declared subjects/seeds and reduced
training budgets; all other protocol settings remain fixed. Such a config is
not eligible for real evidence.

The dependency-light commands are:

```bash
python -m inm.encoder_candidates --plan
python -m inm.encoder_candidates --preflight
```

Plan enumerates the fixed fit matrix and does no fitting. Preflight checks for
the nine T-session GDF paths, persisted `artifacts/Axx_seed_s/split.json` and
`dataset.json` metadata, and numerical package availability. It does not load
recordings or checkpoints; actual metadata contents and recording hashes are
verified by the data implementation. A historical checkpoint source mismatch
does not block fresh fits: old checkpoints are not inputs. Missing real inputs
are readiness findings, not software acceptance failures.

The protocol API is `load_config(path)`, `tasks(cfg)` (ordered dictionaries
with `subject` and `seed`), `arms(cfg)` (ordered arm ID strings),
`study_identity(cfg)`, and `inspect_inputs(cfg)`. Study identity includes the
raw config-byte SHA-256, canonical resolved config SHA-256, SHA-256 map of root
Python files plus all `agfl/` and `inm/` Python files, package versions
discovered without importing numerical packages, and Python version. It does
not claim verified input identities; the data session must add those before any
real task identity is finalized.

`--task-index`, `--smoke`, and `--summarize-only` currently fail explicitly as
not implemented. `--output-dir` is reserved for synthetic smoke and rejected
for other operations. No synthetic result or real-study completion can be
claimed from this session.
