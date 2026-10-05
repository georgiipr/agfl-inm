# Session 01 handoff: protocol and provenance

Completed 2026-10-05. The session added a fixed, dependency-light protocol
declaration and a metadata-only inventory for the completion/Transformer replay.
The implementation is confined to the session's allowed package, config, tests
and documentation paths.

## Files added

- `configs/completion-transformer.json` declares nine participants, three seeds,
  two historical candidate backbones, three input strategies and the five
  fixed availability conditions. Paths resolve relative to this config.
- `inm/completion_transformer/__init__.py` exports the protocol API.
- `inm/completion_transformer/protocol.py` provides strict JSON/config
  validation, subject-major tasks, a stable current study identity, and a
  read-only inventory of recordings, candidate artifacts and baseline split
  metadata. Inventory checksums are byte hashes only; it does not deserialize
  data, checkpoint or prediction arrays and imports only the standard library.
- `tests/completion_transformer/test_protocol.py` covers config validation,
  path resolution, task planning, output overlap/occupancy, deterministic
  identity and opaque-file inventory behavior.
- `docs/completion-transformer/branch-review.md` records the local
  `origin/main` review and cited historical scores.
- `docs/completion-transformer/protocol.md` records the fixed estimands,
  completion settings and input lineage.

No pre-existing session handoffs were present in `docs/completion-transformer/`.

## APIs and commands

The package exports `load_config(path, allow_existing_output=False)`,
`tasks(cfg)`, `study_identity(cfg)`, and `inspect_inputs(cfg)`, along with fixed
`ARM_IDS`, `STRATEGY_IDS`, and `CONDITIONS`. `study_identity` hashes the exact
config, current Python source files, installed package versions and the
historical evidence/split manifest files. Input inventory includes SHA-256
checksums for the nine recordings and the expected historical artifact files.

Executed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/completion_transformer -v
.venv/bin/python -m compileall -q inm/completion_transformer tests/completion_transformer
git diff --check
```

All five synthetic unittest cases passed; compilation and whitespace checks
passed. The config plans 27 tasks. The actual read-only inventory reported all
9/9 recordings, 27/27 candidate tasks with both backbone artifact sets,
27/27 split metadata pairs, both historical metadata manifests, and all five
required packages present. The study identity generated in this checkout was
`bb0311510f5fbb2d42a7dda40cfbd6f66f1f465d1be0bee56db2a87c266e8233`.

## Evidence limits and next-session blockers

The 64.87% Transformer and 59.50% spatial EEGNet validation balanced accuracies
are cited from the existing three-seed candidate report in
[`candidate-followup/findings.md`](../candidate-followup/findings.md); this
session did not recompute them. Their +5.37 percentage point descriptive
difference motivated the question, and their full-input validation-selected
checkpoints make this replay non-independent confirmation.

Input inventory is not replay authorization or a compatibility verdict. The
next replay session must verify each saved fit/task/checkpoint checksum,
constructor, selected epoch, split and data IDs, training normalization,
full-input prediction arrays, and relevant historical source/package identity
before opening numerical arrays or running completion. The candidate study's
`report/evidence.json` is its manifest; there is no required top-level
`study.json`. The baseline directory supplies persisted split and dataset
metadata only. Historical inputs were not modified. No factors were fit and no
real evaluation was run.
