# Fixed replay protocol

`configs/completion-transformer.json` declares 27 tasks: nine participants by
three historical T-session seeds. Each task replays two frozen neural fits
across three input strategies, for six cells. It uses training and validation
trials only. There is no encoder or classifier training, checkpoint reselection,
test access, architecture search, or rank tuning.

Both frozen fits come from each matching task under
`results/encoder-candidates-reproducible-v1/tasks/Axx_seed_n/ARMS/`, in
`spatial_eegnet/` and `spatial_transformer/`. The persisted split and dataset
metadata come from
`results/baselines-reproducible-v1/artifacts/Axx_seed_n/`. Each fit must be
linked to those original participant/seed split and dataset identities, with
its saved normalization and source/package identities verified before replay.
The model constructors, selected epoch and full-input validation prediction
arrays are part of that verification. Results and histories are metadata
evidence, not authorization to change a fixed setting.

The normalized raw input has shape `[N,22,4,250]`. Every arm receives the
original Boolean `[B,22,4]` mask as an availability feature. Completion first
selects observed samples with `torch.where`; zero fill preserves observed
values; imputed samples never turn into reported observations. Every window
must retain at least one channel. Full-input input bypasses completion and must
match the historical model's ordinary forward path exactly.

Zero strategy leaves normalized missing samples at zero. Covariance estimates
the channel second moment on the original normalized training trials, then uses
an observed-channel conditional ridge solve with ridge `0.001` times the mean
diagonal. Tucker fits factors once per task from the original normalized
training `[N,22,4,250]`, shared by both backbones. It uses read-only
`inm.tensor_attention.Tucker2`, channel/feature ranks 4/16, ridge 0.001, 30
epochs, learning rate 0.01 and batch size 64. Its scoped CPU RNG seed is task
seed + 810001 and ambient RNG state is restored after fitting. Neither
completion method gets validation labels or classifier gradients.

Evaluation conditions are `full_22` and five-repeat `random_static` and
`dynamic_random` masks at 16 and 6 channels. Mask realizations and validation
trial order are paired across every cell. Report balanced accuracy and log loss
by participant, seed, strategy, backbone and condition, preserving incomplete
coverage and negative effects.

The primary contrast is the equally averaged four degraded-condition balanced
accuracy difference, Tucker minus zero within the Transformer. Secondary
contrasts are that difference within EEGNet, Tucker minus covariance within the
Transformer, and the difference in Tucker benefit between backbones. The last
is an interaction contrast; it does not prove synergy. Average repeats, then
seeds within participant, then participants equally. The exploratory screen is
a mean gain of at least 2 percentage points and positive gain for at least 6/9
participants. Participant bootstrap intervals use 2000 resamples and seed
20261005. They remain exploratory and validation checkpoint selection has
already occurred on these same participants.

Paths are resolved relative to this JSON file. The new output identity includes
this declaration, current source hashes, package versions and historical
manifest hashes. The read-only inventory records byte checksums without
loading numerical artifacts. Replay must stop with a concrete incompatibility
if any historical fit, split, data, normalization, prediction, source or
package identity cannot be verified.
