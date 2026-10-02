# Session 03 — Reuse loading/splits without leakage

Read `inm/data.py` and tracked `agfl/datasets/{base,splits,eeg}.py` (use Git inventory or explicit paths; ignore rules hide this directory from ordinary rg). Scientific dependencies are required from this session onward. If absent, stop as blocked instead of skipping tests.

Implement PreparedData and prepare_subject as defined in CONTRACT.md. Reuse load_subject and get_split; translate the new config into their actual expected dictionaries. Within-session: T only and existing stratified split. Cross-session: T+E loading with explicit labels_dir; existing session split reserves 20% of T for validation and E for test. Preserve exclusions, IDs, source checksums, channel order and class counts.

Fit raw channel mean/std on TRAIN indices only, saving stats and applying them to all partitions. Keep filtered unnormalized signals separately for the classical pipeline. Do not call per-sample normalization and accidentally erase amplitude information. Preserve independent 250-sample filtering for fair dynamic availability. Reject overlapping sample IDs, missing classes where BA requires four classes, nonfinite observations and mismatched sampling/channel metadata.

Scope: new data adapter and tests; small config fixes only if necessary.

Acceptance: synthetic SignalDataset with injected loader, no GDF download; correct within/session partition roles; modifying validation/test values cannot change training statistics; T/E labels required and never guessed; deterministic persisted splits; config data provenance survives preparation.

Handoff: PreparedData fields/types, normalization convention, fake-data construction helper for later smoke tests.
