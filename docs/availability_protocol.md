# Signal availability protocol

The experiment trains BCI Competition IV 2a participants **individually**, for
all nine participants, using within-participant trial splits. Trials, labels and
splits remain identical across both backbones and their tensor variants. An observation mask
is a synthetic intervention on an otherwise available trial, not a new
participant split or a new source of labels.

## Masks and known electrode identity

The canonical tensor has axes `trial × original electrode × non-overlapping
window × within-window sample`. Its mask `M[n,c,p]` is Boolean. An unavailable
electrode keeps its original index and identity; surviving channels are never
renumbered. Only observed samples may enter tensor inference or prediction;
fitting uses full-channel training samples only. Hidden signal samples are
accessed separately as post-fit reconstruction scoring targets. No operation
should filter across an unavailable interval or present hidden values to the
model before masking them. Full-input preprocessing uses the original run-level
bandpass before trial extraction. For degraded input, remove raw missing
intervals first and filter contiguous observed spans per electrode. An observed
window boundary is not itself a filter boundary. Native gaps and run boundaries
remain filter boundaries. Outages apply inside the cue-aligned trial; raw
recording context outside it remains observed. This is offline zero-phase
preprocessing, not causal streaming. Preprocessed-input hashes accompany mask
hashes to verify that all model variants receive identical observations.

The retained counts are **22, 16, 11 and 6**. They correspond to 0%, 27.27%, 50%
and 72.73% missing channels. Removing exactly 75% of 22 channels is impossible:
retaining six is the most severe integer setting without exceeding 75% removal.
Every window retains the declared count, not merely that count on average.

## Conditions

Every model trains only on all 22 channels, once per participant/seed. Only
full-channel validation runs during training and selects its checkpoint. The
selected weights are then frozen for the entire degraded validation/test sweep.
No channel subset triggers a new training run. Mask seeds exclude model
identity, so both backbones and their tensor versions receive the same
evaluation banks.

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
available during that interval. Electrodes absent from both stay absent. This
matched-count schedule isolates changing channel identity from the amount of
information measured per window. It is one specified outage model; results
should not be described as covering arbitrary real-world outage durations.

## Separate combination banks

Each nonfull subset is represented by a 22-bit integer using canonical electrode
IDs. A fixed SHA-256-based rule assigns this integer to exactly one of training,
validation or test. Sampling rejects subsets assigned to another partition. The
assignment does not depend on pattern, seed or model. Training presents no
degraded subsets. Validation and test degraded subsets are disjoint under the
fixed allocation rule; the reserved training bucket is unused in this full-
channel-only protocol. Both A and B in a dynamic schedule obey the partition
rule. The unique full-input mask necessarily occurs in all partitions and is the
explicit exception. Different seeds/repeats within one partition need not have
disjoint subsets; enforcing this is neither required nor feasible indefinitely.

Sampling allows at most 4096 attempts per subset. It raises an error if it
cannot meet the constraints instead of silently borrowing masks from another
partition. Banks are reproducible by subject, seed, partition, pattern, retained
count, repeat and original trial ID. Pass original trial IDs when data order may
change. Store the mask digest with each result to audit equality across models.

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
anatomically validated lesion or a strictly contiguous region on a cortical
mesh.

## Validation and interpretation

Keep checkpoint selection on **full-input validation** for every model. After
selection, report all 13 scenarios on validation and then held-out test. The
train/evaluation channel-count mismatch is deliberate: each model is trained on
22 channels and assessed under decreasing availability. Degraded validation is
descriptive, not an additional checkpoint or factor-search opportunity. Test
scores never select epochs, ranks, regularization, patterns or models.

Compare each tensor model with its baseline backbone for the same participant,
seed, scenario and mask repeat. Average mask repeats, then seeds within each
participant, before equally weighting all declared participants. Repeats are not
independent participants. Report accuracy, balanced accuracy, macro-F1,
degradation and paired gains, including unfavorable outcomes. The mask generator
does not establish that a tensor model can recover lost information; it provides
a reproducible way to measure whether it helps.

## Reviewable implementation

The implementation is in `inm/availability.py`. Public helpers are
`make_mask_bank`, `evaluation_scenarios`, `availability_metadata`,
`subset_partition` and `mask_bank_digest`. See the [README](../README.md) for
dataset setup and usage instructions.
