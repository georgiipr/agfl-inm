# Signal availability protocol

The experiment trains BCI Competition IV 2a participants **individually**, for all
nine participants. This deliberately replaces the proposal's subject-disjoint
folds, following the user's choice. Trials, their labels and their split remain
identical across the compared attention and tensor arms. An observation mask is
a synthetic intervention on an otherwise available trial, not a new participant
split or a new source of labels.

## Masks and known electrode identity

The canonical tensor has axes `trial × original electrode × non-overlapping
window × feature`. Its mask `M[n,c,p]` is Boolean. An unavailable electrode keeps
its original index and identity; surviving channels are never renumbered.
Only observed entries may enter feature inference, tensor fitting or prediction.
Artificially hidden features may be used separately as reconstruction scoring
targets. No operation should filter across a window boundary into an unavailable
window or present hidden values to the model before masking them.

The retained counts are **22, 16, 11 and 6**. They correspond to 0%, 27.27%, 50%
and 72.73% missing channels. Removing exactly 75% of 22 channels is impossible:
retaining six is the most severe integer setting without exceeding 75% removal.
Every window retains the declared count, not merely that count on average.

## Conditions

Full-input training always presents all 22 channels. Mixed-availability training
chooses 22, 16 or 6 channels with equal probability independently per training
trial and epoch. At 16 or 6 channels, it chooses random static masking or random
dynamic outages with equal probability. Thus the full condition has probability
1/3 and each combination of degraded count and training pattern has probability
1/6. Training never presents an 11-channel trial or samples from the spatial-loss
generator (a random subset can still be spatially clustered by chance). The
complete banks are deterministic given subject, seed, epoch and
original trial IDs; model, attention and tensor settings do not enter that key.

Evaluation comprises 13 scenarios: one full-input scenario and the following
four patterns at each of 16, 11 and 6 retained channels:

| Pattern | Construction |
|---|---|
| `random_static` | One random observed subset throughout a trial. |
| `spatial_static` | Remove the electrodes nearest one center in schematic scalp coordinates, throughout a trial. |
| `dynamic_random` | A random subset A, a different random subset B, then A again. |
| `dynamic_spatial` | Spatially grouped loss A, a different spatially grouped loss B, then A again. |

For `P` windows the ABA transitions occur at `P//3` and `2*P//3`. Each phase is
nonempty, so dynamic scenarios require at least three windows. Both A and B
retain exactly `k` electrodes. Electrodes in A but not B disappear for one
contiguous middle interval and then return. Electrodes in B but not A become
available during that interval. Electrodes absent from both stay absent.
This matched-count schedule isolates changing channel identity from the amount
of information measured per window. It is one specified outage model; results
should not be described as covering arbitrary real-world outage durations.

## Separate combination banks

Each nonfull subset is represented by a 22-bit integer using canonical electrode
IDs. A fixed SHA-256-based rule assigns this integer to exactly one of training,
validation or test. Sampling rejects subsets assigned to another partition.
The assignment does not depend on pattern, seed or model. Therefore a subset
seen during training cannot recur as an evaluation subset even if its pattern
label changes. Both A and B in a dynamic schedule obey the partition rule.
The unique full-input mask necessarily occurs in all partitions and is the
explicit exception. Different seeds/repeats within one partition need not have
disjoint subsets; enforcing this is neither required nor feasible indefinitely.

Sampling allows at most 4096 attempts per subset. It raises an error if it cannot
meet the constraints instead of silently borrowing masks from another partition.
Banks are reproducible by subject, seed, partition, pattern, retained count,
repeat and original trial ID. Pass original trial IDs when data order may change.
Store the mask digest with each result to audit equality across comparison arms.

## Fixed spatial coordinates

The spatial-loss generator uses this deliberately simple fixed 10–20 projection.
Positive `x` is rightward, positive `y` is anterior. Units are arbitrary. These
coordinates define synthetic geometric losses only: they are not individual
electrode measurements, inferred anatomical connectivity or functional brain
connections.

| Channels, in left-to-right order | `x` positions | `y` |
|---|---|---:|
| Fz | 0 | 2 |
| FC3, FC1, FCz, FC2, FC4 | −2, −1, 0, 1, 2 | 1 |
| C5, C3, C1, Cz, C2, C4, C6 | −3, −2, −1, 0, 1, 2, 3 | 0 |
| CP3, CP1, CPz, CP2, CP4 | −2, −1, 0, 1, 2 | −1 |
| P1, Pz, P2 | −1, 0, 1 | −2 |
| POz | 0 | −3 |

For a spatial mask, sample one center uniformly in `[-4,4] × [-4,3]`; mark the
`22-k` nearest electrodes as missing. Negligible deterministic random jitter
breaks distance ties. This generates spatially grouped losses. It is not an
anatomically validated lesion or a strictly contiguous region on a cortical mesh.

## Validation and interpretation

Keep checkpoint selection on the declared **full-input validation** criterion for
both training regimes. After selecting the checkpoint, report all 13 scenarios
on validation and then on the held-out test set. Consequently full-input-only
training is explicitly assessed under a train/evaluation channel-count mismatch,
and mixed training is assessed on the unseen 11-channel count and unseen spatial
patterns. Degraded validation scores are descriptive outputs of this fixed
experiment, not an additional checkpoint-search opportunity. Held-out test scores
must never select epochs, ranks, regularisation, patterns or experiment arms.

Compare tensor versus non-tensor arms within the same subject, seed, training
regime, attention, evaluation scenario and mask repeat. Average mask repeats
within each subject before subject-level inference; repeats are not independent
participants. Report full-input accuracy, balanced accuracy, macro-F1,
degradation and paired gains, including unfavorable outcomes. The mask generator
does not establish that a tensor model can recover lost information; it provides
a reproducible way to measure whether it helps.

## Reviewable implementation

The implementation is in `inm/availability.py`. Public helpers are
`make_mask_bank`, `training_mask_bank`, `evaluation_scenarios`,
`availability_metadata`, `subset_partition` and `mask_bank_digest`. No tests,
training or model inference were run locally; numerical and runtime behavior
remain to be validated on the cluster.
