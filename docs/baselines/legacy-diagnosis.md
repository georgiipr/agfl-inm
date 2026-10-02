# Diagnosing the legacy pipeline by stage

The real A01/seed-0 pilot has now been reviewed; see
[legacy-pilot-review.md](legacy-pilot-review.md) for its validated stage scores
and the session-12 command.

This report helps locate a possible weakness in the existing shared-encoder,
baseline-head, and Tucker-head pipeline. It reads saved JSON artifacts only; it
does not load checkpoints, tensors, predictions, or numerical packages, and it
does not make automatic architecture recommendations.

## View saved evidence

From the repository root, run:

```sh
"$AGFL_PYTHON" -c 'import json; from inm.baselines.diagnostics import inspect_legacy; print(json.dumps(inspect_legacy("results/inm-v2")["stage_comparison"], indent=2))'
```

Set `AGFL_PYTHON` to the configured interpreter path. The programmatic API is
`inm.baselines.diagnostics.inspect_legacy(output_dir) -> dict`; inspect its
`stage_comparison` member. Each task has `validation_stages`, separately labeled
`test_metrics`, `matched_head_pairs`, and `missing_sources`. The shared
pretraining score comes from `calibration.json`; its history is checked at the
selected epoch in `encoder_history.json`. Baseline and Tucker scores come from
full-input `VALIDATION` result rows and their histories at each result's
selected epoch. Test rows are retained only under `test_metrics` and explicitly
excluded from architecture recommendations.

## How to read it

1. **Weak shared-pretraining validation performance** can point to an issue in
   representation learning, training, or data handling. These are hypotheses;
   this score alone cannot distinguish among them.
2. **Strong pretraining but a weak retrained baseline head** can point to the
   head, feature normalization, optimization, or the classifier training setup.
3. **A strong baseline head but a weak Tucker head** can point to compression,
   regularization, or the core inference/classification path.

The configured tensor ranks are `rank_channels=4` and `rank_features=4`, with
ridge `0.001`. The Tucker core has 4×4×4 = 64 entries compared with 22×4×32 =
2816 entries in the original channel/window/feature tensor. This 2816→64
reduction describes the representation size; it does not establish information
preservation or explain any score difference. Rank or ridge changes belong to
a separate, controlled investigation.

The shared encoder and downstream heads select their epochs independently on
validation data. Comparing their respective selected validation maxima is
therefore diagnostic and selection-biased, not independent test evidence. A
test score is not a substitute for stage-local validation evidence, and must
not enter an architecture recommendation. Unless common held-out raw
predictions and labels were explicitly saved, summary metrics cannot recreate
paired predictions, paired uncertainty, or example-level error overlap.

Only pair baseline and Tucker heads when calibration identity matches and
attention and training regime match. A different model family, including the
new spatial EEGNet baseline, does not share the legacy frozen feature space and
must never be paired with a legacy Tucker head as a matched representation
comparison. Missing, corrupt, or mismatched calibration/result identities are
reported as unavailable evidence rather than silently compared.

The legacy writer stores the frozen calibration digest as top-level
`cache_sha256` in `calibration.json` and as `identity.calibration_sha256` in arm
results. The reader normalizes those names; conflicting values are rejected.
