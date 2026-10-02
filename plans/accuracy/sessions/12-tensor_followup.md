# Session 12 — Optional tensor follow-up with a fair feature source

Prerequisite: real baseline pilot and the session-09 diagnostic report. If data/results needed to ground the follow-up are absent, return blocked. The purpose is to reduce the OLD representation matrix, not bolt Tucker onto a CNN whose spatial channel axis has already collapsed.

Implement a narrowly scoped new entry/config for the legacy channel-local frozen feature source. Reuse the existing Tucker2, FeatureClassifier, calibration mathematics and mask rules without changing legacy schema 2 or legacy result directories. Declare MHA only; baseline, tensor_completion, tensor_core as initial representations, with mean_completion and linear_core available as controls; full and mixed regimes. Rank/ridge remain explicit. All compared representations share the identical frozen encoder, features, factors where applicable, masks and head-selection policy.

Keep this study's provenance and report grouping separate from end-to-end spatial EEGNet. Do not label cross-backbone differences a tensor effect. Preserve observations exactly in completion and no autograd through detached Tucker solves. Support a small explicit rank/ridge candidate list for later validation-only runs, not an automated test-score search or unconstrained sweep. Validation selection must respect held-out mask patterns.

Write docs/baselines/tensor-followup.md with the smaller study matrix, control rationale, limits, commands and a decision rule: retain tensor complexity only with measured paired robustness benefits and an explicitly discussed full-input cost. If no data support that decision yet, leave it open.

Acceptance: config/matrix and separate output identity; frozen factors stay unchanged under classifier optimization; hidden NaN isolation; observed completion entries preserved; every pair shares feature/calibration and mask hashes; existing full 378-fit experiment unchanged. Tests use tiny synthetic tensors and must not require real EEG.

This task may touch a small adapter plus new config, never rewrite the tensor solver. If reuse requires a large legacy refactor, return blocked with the smallest proposed interface change rather than making an architectural rewrite.
