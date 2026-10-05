# Branch and historical evidence review

Reviewed the checked-out `research/baseline-accuracy` implementation and the
local Git reference `origin/main` at `3c9a170` (`v0.3`). This records the local
reference available in this checkout; it does not claim to have checked a newer
remote revision.

The local `origin/main` implementation attaches `Tucker2` through its shared
`SignalInput` layer. It first applies the original Boolean availability mask,
fits the factors when the tensor model is calibrated, and fills the absent raw
channel samples before the backbone. Its EEGNet applies a whole-trial temporal
convolution, attention over electrode identities at each time point, then
spatial EEGNet mixing. Its Signal Transformer instead tokenizes each electrode
over the whole trial and applies spatial attention across electrodes. The two
backbones therefore have different attention domains, and the shared completion
entry point does not itself constitute the matched frozen-checkpoint replay in
this protocol. No code from that branch was merged.

The local upstream schema/configuration also describes a different preprocessing
and training design, including run-scope filtering and a 500-epoch budget. This
successor fixes the historical within-T, window-filtered, 250-sample window
preprocessing and replays already selected checkpoints. Those upstream defaults
do not alter this study's declaration.

The current branch's selected candidate `spatial_transformer` is an EEGNet
front end followed by temporal Transformer blocks. Both frozen backbones come
from the same candidate task: `spatial_transformer` and `spatial_eegnet` each
have their own checkpoint and saved full-input validation predictions. The
baseline reproducible study supplies the persisted split and dataset metadata
used when the candidate study prepared each task. Their subject/seed split and
dataset identities must be checked before any replay. The original Boolean
`[B,22,4]` availability flags remain attached to all six cells. Completed
values enter the raw EEGNet input while those flags continue to identify which
measurements were actually observed.

The cited three-seed validation report gives 64.87% balanced accuracy for
`spatial_transformer` and 59.50% for `spatial_eegnet`, a descriptive difference
of +5.37 percentage points (95% participant bootstrap interval −0.42 to
+12.00; 7/9 participants positive). These values are reported in the existing
[candidate follow-up findings](../candidate-followup/findings.md) and are not
recomputed here. They motivated the matched completion question; they do not
establish that completion will help. Checkpoint selection used full-input
validation, so this replay reuses selected validation checkpoints and is not
independent confirmation.

The three strategies are normalized zero fill, a training-only channel
second-moment conditional ridge predictor, and training-only Tucker completion.
Covariance uses ridge `0.001 * mean(diagonal)` without per-mask tuning. Tucker
reuses `inm.tensor_attention.Tucker2` with channel/feature ranks 4/16, ridge
0.001, 30 factor epochs, learning rate 0.01 and batch size 64. Its seed is the
task seed plus 810001, with ambient RNG restored. Full input bypasses completion,
so the study makes no clean-accuracy gain claim. Entirely missing windows remain
invalid. Completion is independent by window and the shared EEGNet remains a
whole-trial convolutional model; this protocol does not claim causal online
processing or recovery of a fully absent window.

The plan keeps historical inputs immutable and gives all replay artifacts a
new `results/completion-transformer-v1` identity. Before numerical replay, the
later replay session must verify fit/task/checkpoint hashes, constructors,
split and data IDs, normalization, saved full-input predictions, and the
relevant historical source/package identities. The metadata inventory in this
session only records paths, presence and byte checksums; it does not deserialize
EEG, checkpoints or prediction arrays.
