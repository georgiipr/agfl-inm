# Legacy frozen-feature tensor follow-up

The six-arm cohort is now complete (162/162 fits). Read
[the results review](tensor-results-review.md) before launching further work;
it recommends stopping tensor expansion and auditing the simpler baseline.
The commands below document the completed study and optional controls.

This is a reduced comparison in the legacy channel-local frozen feature space.
It is separate from the end-to-end spatial EEGNet studies and their reports.
The A01/seed-0 pilot and corrected session-09 diagnosis are available in
`results/inm-v2` and reviewed in [legacy-pilot-review.md](legacy-pilot-review.md).
The one-participant diagnosis is operational evidence only; it does not establish
an effect or justify choosing a rank or ridge value.

## Matrix and controls

`configs/tensor-followup.json` declares MHA only. The initial matrix uses
`baseline`, `tensor_completion`, and `tensor_core`, each with `full` and `mixed`
classifier training (six heads per task). `mean_completion` and `linear_core`
are available controls, added only with the explicit `--include-controls` flag.
They then use the same two regimes and a separate
`results/inm-tensor-followup-controls-v1` directory. The default output identity
is `results/inm-tensor-followup-v1`, separate from `results/inm-v2` and every
end-to-end EEGNet output.

Baseline retains the observed feature tokens and learned missing placeholders.
Tensor completion fills missing entries from the training-fitted Tucker model
while preserving observed entries. Tensor core changes to inferred latent
channel/feature components. Mean completion checks whether a simple training
mean fill explains an effect; linear core checks shape-matched learned
compression without a Tucker reconstruction objective. Keep the original
channel/window identities and matched mask banks.

Every representation must use one common frozen encoder, feature tensor,
training calibration, split, and evaluation mask bank. Tucker factors are fitted
once from training-only features and reused unchanged; classifier optimization
must not update them. Pair comparisons only when feature, calibration, and mask
hashes match. Hidden entries must be selected out before arithmetic, completion
must copy measured entries exactly, and the Tucker core solve remains detached
from classifier autograd.

The explicit starting setting is rank `(4,4)` and ridge `0.001`. The config
also records a small candidate list for separately declared validation-only
runs. It is not an automatic sweep. Use held-out validation mask patterns for
selection, with the declared robust balanced-accuracy mean and validation
log-loss tie-breaker. Never inspect test results while choosing a candidate.

## Run and interpret

The base protocol remains the supplied legacy T-session, stratified split,
window-local filtering, and frozen-feature source. Use the project interpreter
and keep the output directory separate:

```sh
"$AGFL_PYTHON" -m inm.tensor_followup --config configs/tensor-followup.json --plan
"$AGFL_PYTHON" -m inm.tensor_followup --config configs/tensor-followup.json --task-index 0 --device cuda
"$AGFL_PYTHON" -m inm.tensor_followup --config configs/tensor-followup.json --task-index 0 --include-controls --device cuda
"$AGFL_PYTHON" run.py --config configs/study.json --plan
"$AGFL_PYTHON" -m unittest discover -s tests/baselines -p 'test_tensor_followup.py' -v
```

The default follow-up has 27 tasks and six fits per task; `--include-controls`
selects a distinct ten-fit matrix and output identity. The original experiment
remains 27 tasks and 378 classifier fits. This follow-up document does not claim new
measurements. Retain tensor complexity only if a completed, paired evaluation
shows robustness benefits on held-out loss patterns and the full-input cost is
explicitly discussed. If those data do not yet exist, leave the decision open.

Do not compare this study's scores to end-to-end EEGNet as a tensor effect: its
backbone, learned feature source, and calibration are different. This reduced
matrix also does not establish cross-subject or cross-session performance.
