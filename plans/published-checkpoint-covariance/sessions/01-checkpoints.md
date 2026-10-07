# Session 01: published assets and native recipe

Read ../CONTRACT.md and ../ACCEPTANCE.md. Default20minutes; fresh worker;
supervisor checks<=60seconds. No covariance fitting or E outcome metrics.

Worker write scope: `docs/published-covariance/checkpoint-audit.md`,
`docs/published-covariance/checkpoint-audit.json`,
`docs/published-covariance/session-01-handoff.md`, and a proposed dependency
specification under `docs/published-covariance/`. No other code or plans.
Supervisor owns downloaded assets, dependency installation, inventories and receipts.

Audit the official ATCNet run-1 nine-subject weights first. Request/download through
the supervisor as needed; retain immutable URLs, commit IDs, license and hashes.
Inspect HDF5 tensor names/shapes and the matching author model/preprocessing code.
Determine architecture, classes, expected input, normalization reconstruction,
training/epoch/run selection history and native runtime compatibility. Trace the
asset revision rather than assuming today's README describes older saved weights.

Primary references, checked during planning on 2026-10-06:

- [ATCNet repository](https://github.com/Altaheri/EEG-ATCNet).
- [Nine subject weight files](https://github.com/Altaheri/EEG-ATCNet/tree/main/results/saved%20models/run-1).
- [Native preprocessing](https://github.com/Altaheri/EEG-ATCNet/blob/main/preprocess.py).
- [Older training script](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainTest.py).
- [Revised training script](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainValTest.py).
- [Official TCFormer fallback](https://github.com/Altaheri/TCFormer).

The older ATCNet script feeds test data into validation-based selection. The
authors now recommend the revised split. This is evidence to investigate the
published assets, not proof that every checkpoint has the same history.

Fill a concrete admission table: subject, hash, compatible architecture, input
recipe, training sessions, selection sessions, normalization source, provenance
grade, unresolved facts. State which complete set was chosen and why, without
using local accuracy. Recommend native TensorFlow versus an audited port with
resource requirements. No speculative model conversion or installation into the
existing environment. Native missing-data preprocessing must satisfy the contract.

Handoff: authoritative source links, complete metadata, proposed runtime,
missing inputs, admission recommendation and exact evidence. Supervisor verifies
assets/interpretation and freezes the recipe before session02. If no compatible
set exists after the bounded fallback, stop; do not create infrastructure for an
unavailable classifier or start a replacement backbone-training project.
