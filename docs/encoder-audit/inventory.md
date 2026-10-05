# Encoder investigation protocol and input inventory

Session 01, observed 2026-10-02 on `research/baseline-accuracy`, implements the
read-only declaration and inventory. No audit measurement, checkpoint replay,
probe fit, neural fit, or GPU job was performed. This is software acceptance,
not an empirical result. The original study files and reports were not updated.

## Declared audit

[The config](../../configs/encoder-audit.json) uses schema
`agfl-encoder-audit-v1`, participants 1–9, seeds 0–2, and the existing T-session
within-participant splits. Only training and validation enter analysis. Reused
validation is exploratory; existing selected validation maxima are
selection-biased. Inventory omits every score, including validation scores.

The protocol fixes 22 channels, four 250-sample windows, 250 Hz, cue offset zero,
and window-local 2–30 Hz filtering. Baseline raw normalization has epsilon
`1e-8`; legacy raw and feature floors are `1e-12` and `1e-6`. Their conventions
remain separate. The inventory checks finite saved JSON statistics; independent
training-only recomputation belongs to replay, not this session.

Each task declares three full-input logistic probes: ordered features (2816),
window-mean features (704), and ordered frozen Tucker core (64), plus the
ordered-feature shuffled-training-label control (seed + 700001). All use a
training-fitted StandardScaler, L2, C=1, L-BFGS, max_iter=1000, tol=1e-4,
random_state=0, and no class weights or search. No removed sklearn
`multi_class` option is prescribed. Total: 108 CPU probe fits, zero neural
refits. These are future audit budgets, not executed fits.

The same ordered-feature probe will compare full input, normalized zero-fill,
and frozen Tucker completion. Validation conditions are `full_22` (one repeat)
and `random_static_16`, `dynamic_random_16`, `random_static_6`,
`dynamic_random_6` (five repeats each). Spatial losses, 11-channel conditions,
and test results are excluded. Feature/probability tolerances are
rtol=1e-4, atol=1e-5; normalization tolerances are rtol=1e-6, atol=1e-8.
Weak linear probes cannot prove absence of nonlinear information, and an NRMSE
threshold cannot establish classification adequacy.

Paths resolve from the configuration directory, including symlinks:

| Key | Declaration | Resolved location in this checkout |
|---|---|---|
| `data_dir` | `../../ml` | `/home/kalexu97/Projects/ml` |
| `baseline_dir` | `../results/baselines-reproducible-v1` | `results/baselines-reproducible-v1` |
| `legacy_dir` | `../results/inm-tensor-followup-v1` | `results/inm-tensor-followup-v1` |
| `output_dir` | `../results/encoder-audit-v1` | `results/encoder-audit-v1` |

Output may neither equal, contain, nor be contained by either study input or
the data directory; it may not equal the repository root. Unknown fields,
selection/search overrides, nonfinite settings, test partitions, cross-session
input, invalid masks, and duplicate subjects/seeds are rejected. A subset of the
configured subjects/seeds can be declared for a bounded task fixture; the
supplied cohort config remains 9 × 3. No output directory is created by plan
or inventory.

## Observed identities and coverage

Directory names were not used as evidence. The original manifests,
per-task JSON identities, split records, histories, and recorded binary hashes
were read. Recorded calibration/checkpoint/history/model byte digests were
verified without tensor deserialization. Dataset paths were inventoried without
GDF parsing; recording checksums still require the later artifact reader.

| Input | Observed metadata/file coverage | Manifest byte SHA-256 |
|---|---|---|
| Baseline | 27 tasks, 108 result records; 81 selected neural checkpoints and histories; 27 saved covariance models | `bbffbdc8717b7d9ba7db52a2a9702b57b797611383798798f57142e30dd12bba` |
| Legacy tensor follow-up | 27 calibration records/caches and histories; 162 downstream result/history records | `f41e37af1be8470949c0aa9483605dd7bf060f69da9a9d466c72bd423e6888e6` |

Coverage means metadata and required files are available, not that classifiers
have been replayed or that the studies can yet be paired. The legacy pilot
`results/inm-v2` is not an input substitute.

The legacy stored study ID is
`fb7ada96d388b9de4540d7e45e37df5239311f2917ab660c4ce458f146f73960`.
Its recorded source-map digest is
`c5dad41b7140308df3ab3d2bdce693bbe49e020ee2bd2362b2f6d93c2960ecda`;
all recorded source paths match current bytes.

The baseline manifest has no stored `study_id`. Inventory explicitly derives
its input ID as SHA-256 of canonical JSON of its recorded `identity`:
`29a1b6c867ccc2609cddf77ef2f0621afa8beb14a244da72e4ad38680e282ccb`.
Its recorded config checksum is
`29787dad80a59172091fa6d6417585c3c83fe55fc572544f8aaf586130277f91`
and matches the original config file. The canonical source-map digest, derived
by this inventory, is
`915be9abb74cedc6d2fe803bdfa711ec4ecf4e38a98fdf02fee5cfbfe72df83c`.
Six baseline historical source paths differ:

- `inm/baselines/diagnostics.py`
- `inm/baselines/eegnet.py`
- `inm/baselines/protocol.py`
- `inm/baselines/reporting.py`
- `inm/baselines/study.py`
- `inm/training.py`

The known replay dependencies `inm/baselines/eegnet.py` and `inm/training.py`
therefore produce explicit `replay_source_mismatch` blockers. Inventory status
is **blocked**, and its CLI exits 1. Faithful baseline replay cannot silently
substitute the current code. Later replay must verify every additional source
file it imports against the historical map. Newly added audit source is given
its own identity rather than compared to the historical whole-checkout digest.
No historical manifest or source file was repaired in this session.

All nine expected T-session recording paths exist. Required packages are
discoverable under `AGFL_PYTHON`: numpy 2.5.3, scipy 1.18.1, torch 2.14.1+cu130,
scikit-learn 1.9.1, mne 1.13.2, tqdm 4.70.1. Discovery/version metadata does not
establish numerical runtime readiness, and no numerical package was imported
by planning or inventory.

## Capabilities and limits

The legacy cache writer persists encoder weights, factors, normalized learned
features, saved statistics, and selection summaries. Its cache container
includes all partitions: later readers must immediately extract training and
validation and must not expose test rows. Inventory does not inspect tensor
contents. `capabilities.checkpoint_kind` declares
`encoder_only_with_factors_and_feature_cache`;
`pretraining_classifier_reloadable` and `downstream_classifiers_reloadable`
are both false. The whole pretraining MHA head and selected downstream heads
were never persisted. They cannot be reconstructed from the encoder state.

Baseline neural files are selected classifier checkpoints, with verified saved
hashes. Their `classifier_checkpoint_available` flag describes file integrity;
`classifier_replay_verified` remains false until actual evaluation. Covariance
is saved context only and will not be refitted. Cross-study trial alignment,
normalization replay, dropout/BatchNorm checks, probes, and completion diagnostics
are subsequent sessions. None is replaced by this inventory.

## API and CLI handoff

`protocol.load_config(path)` returns a primitive dictionary with absolute path
strings and an added `config_path`. Top-level keys are `schema_name`, `name`,
`subjects`, `seeds`, the four paths above, `protocol`, `partitions`,
`preprocessing`, `normalization`, `budgets`, `probes`, `conditions`, and
`tolerances`. Nested keys are fixed by the checked-in config;
`protocol.tasks(cfg)` returns `(subject, seed)` pairs in declared subject-major
order. `protocol.audit_identity(cfg)` returns raw/resolved config digests,
the exact root/agfl/inm Python source map and digest, package versions, Python
version, and a separate `audit_id`. Original input IDs remain separate.

`inventory.inspect_inputs(cfg)` returns primitive JSON metadata with `status`,
`input_study_ids`, `studies`, `dependencies`, `data`, `audit_identity`, and
structured `blockers` containing `code`, `path`, and `message`. It copies no
metrics, labels, feature values, or test counts into public results.
`fits_available` counts file/metadata availability separately from source
readiness. Missing files, corrupt JSON, missing/mismatched checksums, incorrect
identities, split overlap, saved-ID reorder, and nonfinite normalization metadata
produce blockers. No original experiment writer or resume routine is invoked.

```bash
"$AGFL_PYTHON" -m inm.encoder_audit --config configs/encoder-audit.json --plan
"$AGFL_PYTHON" -m inm.encoder_audit --config configs/encoder-audit.json --inventory
"$AGFL_PYTHON" -m unittest discover -s tests/encoder_audit -p test_protocol.py -v
"$AGFL_PYTHON" run.py --plan
```

Plan prints `Tasks: 27; CPU probe fits: 108; neural refits: 0` and
`Partitions: train/validation only; exploratory reused validation`. Inventory
prints JSON to stdout, exits 0 when ready and 1 when blocked. Invalid CLI/config
arguments exit 2. CLI actions are mutually exclusive; default device is CPU,
with explicit `--device cuda` accepted for future replay. `--task-index`,
`--smoke`, and `--summarize-only` fail explicitly as unimplemented in session 01.
The reserved `--output-dir` argument is only accepted for future synthetic
smoke; no synthetic study is implemented here.

Tests use temporary stdlib fixtures with deliberately unreadable-as-tensor
checkpoint bytes. Planning is exercised in a fresh interpreter with numerical
imports blocked. Test metrics, labels, and feature fields are poisoned with
NaN/infinity and must leave the entire inventory unchanged. Bad saved
normalization statistics instead block inventory. Fixtures retain original
input bytes and test both symlink containment directions. The legacy planner
still declares 27 tasks and 378 classifier fits. This session does not interpret
test results or create a research-success receipt.
