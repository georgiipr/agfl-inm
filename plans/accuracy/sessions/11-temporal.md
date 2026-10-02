# Session 11 — Optional temporal head, only after the real baseline pilot

Prerequisite: a real completed A01/seed-0 baseline pilot and valid reports/histories per CONTRACT.md. If absent, return blocked immediately with the runbook command that supplies it. Synthetic smoke is insufficient. Read validation evidence only for design choices; no test-based tuning.

Implement ONE optional small temporal-convolution head over the EEGNet feature sequence instead of the flatten readout. Keep raw masking, spatial front-end, split/preprocessing, training budget and selection policy fixed. Retain the existing flatten mode as the default and comparator. Declare head width/kernel/dilation and parameter counts. No extra attention, larger encoder, augmentation or rank search in this ablation.

Write configs/baselines-temporal.json with a NEW output identity, matching full/mixed CNN arms and both heads. Adapt planning/reporting generically only as needed. Record in docs/baselines/temporal-ablation.md why the experiment is being run, what is held fixed, and the validation-only decision rule. Do not claim a benefit without all required measured results.

Acceptance: temporal logits/gradients, hidden-value invariance and checkpoint roundtrip; configuration enumeration and pairing prevent pooling results from different head types; legacy and default baseline configurations unchanged. Run the relevant earlier tests. No full GPU experiment is launched.

Handoff: ready-to-run paired ablation commands and the evidence still needed to choose a head.
