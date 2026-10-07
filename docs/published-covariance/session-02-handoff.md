# Session 02 handoff: native data and static masks

**Decision:** the session-02 data and mask implementation passes its synthetic
checks and the independent A01 T native-array comparison. It is ready for
supervisor review. This does not close the cohort-wide input-origin gate: only
A01 T was available in the audit assets. The worker did not fit covariance,
load E data, run the classifier, or calculate E outcomes.

The adapter in `published_covariance/data.py` returns native trial arrays in
chronological source order, labels 0–3, stable run/trial IDs, run IDs, artifact
flags, and source/hash/protocol metadata. It uses the author's direct stored
trial index, first 22 channels, `[375:1500]` slice, 250 Hz, and retains flagged
trials. The original channel/class maps and exact slice are described in
[data.md](data.md). Normalization replays the authored shuffle-42,
per-channel/per-timepoint T StandardScaler and supports hash-bound save/load.
The 80/20 split follows the prescribed class-ordered PCG64 allocation. Static
masks cover 22/16/6 electrodes, key on study identity and original trial ID,
and use disjoint subset pools for train, validation, and E.

Verification used the sealed runtime
`.venv-published-covariance/bin/python`:

- `-m unittest discover -s tests/published_covariance -v`: 7 local tests passed.
- `PYTHONPATH=. .venv-published-covariance/bin/python plans/published-checkpoint-covariance/acceptance_data.py`: all 5 supervisor-authored independent acceptance tests passed, including real A01 T array/label equality with the pinned author's loader.
- Synthetic fixtures explicitly test channel offsets/order, cue slice, labels, artifact retention, NaN/Inf hidden sanitization, exact normalization and split membership, provenance archive binding, deterministic masks and cross-partition subset disjointness.

The predeclared A01 comparison tolerance was native extraction bitwise exact;
same-runtime normalization bitwise exact; GDF microvolt absolute `1e-8`,
relative `1e-10`; and labels, channel identities and trial alignment exact.
The native-array check met the bitwise sample and label criterion for all 288
A01 T trials. No GDF comparison was run by this worker. The A01 historical
diagnostic reported 15 flagged trials retained and a one-sample offset from GDF;
this implementation preserves the audited offset. A02–A09 structured MAT/GDF
arrays were absent, so their alignment/locality and per-trial accounting still
need verification by the supervisor before freezing the full cohort data
identity. This session's native check was data-only, with no model scoring.

Implementation uses no new dependencies and does not alter historical packages.
