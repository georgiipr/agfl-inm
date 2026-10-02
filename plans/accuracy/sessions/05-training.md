# Session 05 — Training and selection with a small API

Read legacy `inm/training.py`, `agfl/optimization.py`, `agfl/reproducibility.py`, and the shared contract. Reuse low-level metrics/loss helpers; avoid changing the legacy fit function.

Implement fit_classifier for neural models. Use the declared optimizer, schedule, clipping and early stopping, deterministic trial order and existing training_mask_bank. Run validation in eval/no_grad mode. Implement explicit full and robust validation selection exactly as defined in CONTRACT.md; fixed validation masks, no 11-channel/spatial/test conditions. All compared arms with the same selection policy must share these banks. Allow tiny synthetic epoch settings in tests.

Save selected checkpoint weights, constructor, preprocessing stats, class/channel order, protocol identity and selection evidence. Histories distinguish masked training loss from clean eval metrics. Reload selected weights before returning. Keep classifier serialization/load helpers simple and explicit.

Acceptance: tiny deterministic CPU model/fixtures; verify a later worse epoch does not replace the best checkpoint; tie-breaking by log loss; exact robust average; selection never reads test labels (change/remove them in the fixture); train-only statistics unchanged; checkpoint reload predictions match; minimum-epoch/patience behavior; invalid empty classes fail visibly.

Scope: training module and tests, minimal model checkpoint adapter if needed. Do not implement real-data orchestration yet.

Handoff: return dict/checkpoint/history formats and actual test commands.
