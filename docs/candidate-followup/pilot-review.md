# Candidate pilot review: A01 / seed 0

**Decision: proceed to the cohort.** The A01/seed0 task is complete real evidence
for the frozen reproducible-v1 study. Its identity and artifacts verify, its
five selected checkpoints reproduce the saved clean validation predictions, and
the required local probes converged. No operational failure blocks the cohort.
This decision authorizes only the separately gated cohort launcher; it is not an
architecture selection.

## Identity and integrity

- Study ID: `b72da29fe20c355bed86810c5a865595678e2b476f0f91f7083437da163246dc`
- SHA-256 of the exact `tasks/A01_seed_0/task.json` bytes:
  `a0129e80fd54b4469b4986ffeb18ce089c81558f383c29987fad6bf05d2979ed`
- The task is `complete`, `synthetic: false`, subject 1, seed 0, with only the
  declared train and validation partitions. All seven task checks are true.
- Current config, source-file map, and package versions match readiness and task
  provenance. The recorded A01 recording and source metadata hashes were checked
  by reloading the real input; the regenerated `data_provenance.json` matched the
  saved file exactly. Normalization uses training trials/time only, and the
  split ID, data ID, normalization, and ordered train/validation sample IDs
  reproduce exactly. There are 163 training and 55 validation trials.
- All five fit-result hashes and every artifact hash declared by those results
  verify, including checkpoints, histories, prediction arrays, and both probe
  arrays for the local arms. The official candidate fit verifier accepted the
  histories, finite numeric arrays, metrics, and ordered sample identities.
  No test output is present. No pilot failure journal was present.
- Every fit and checkpoint records CPU execution and the same versioned RNG
  policy: task seed 0, fit seed 1,900,001, fit stream offset 1,900,001, and
  per-epoch data-order offset 700,001 with the declared formula. Deterministic
  algorithms are enabled. CPU is the only readiness-verified device; CUDA
  remains unverified.

## Selected checkpoints and clean metrics

Selection was recomputed from each full-input validation history using balanced
accuracy, then lower validation log-loss, then earlier epoch. Exactly one row is
marked selected per arm, and its epoch and metrics agree with the result and
checkpoint. Saved validation sample IDs and labels match the regenerated split.
After loading each checkpoint and replaying raw validation input on CPU, all
five probability arrays matched their saved arrays exactly (maximum absolute
difference `0.0`).

| Arm | Selected epoch | Train BA | Validation BA | Train − validation BA | Train log-loss | Validation log-loss |
|---|---:|---:|---:|---:|---:|---:|
| `local_control` | 44 | 0.4665 | 0.5426 | -0.0761 | 1.0799 | 1.1071 |
| `local_power` | 137 | 0.7741 | 0.5673 | +0.2068 | 0.6095 | 1.2262 |
| `spatial_eegnet` | 162 | 0.9508 | 0.8530 | +0.0977 | 0.3944 | 0.5857 |
| `spatial_filterbank` | 134 | 0.5631 | 0.4560 | +0.1071 | 1.1460 | 1.2862 |
| `spatial_transformer` | 184 | 0.9017 | 0.8365 | +0.0651 | 0.2609 | 0.4593 |

The local-control validation BA exceeds its training BA on this task. The local-
power validation log-loss is higher than its training log-loss; these differences
are reported descriptively and do not indicate an artifact-integrity failure.
They are not evidence for a cohort-level effect.

## Pairing and probes

For every arm, the full-input and four masked validation conditions have the
expected repeats and identical mask hashes across arms. Both local arms' ordered
and shuffled-label probes converged, and their stored arrays carry the exact
train and validation sample IDs. Ordered-probe validation BA was 0.4931 for
`local_control` and 0.6538 for `local_power`; shuffled-label-control validation
BA was 0.2033 and 0.2170, respectively. These are single-task diagnostics, not
significance tests.

## Scope and interpretation

This pilot covers one participant and one seed, with 55 validation trials. Its
ranking and scores cannot establish participant-level generalization, a reliable
architecture winner, or the predeclared cohort screening rule. `spatial_eegnet`
and `spatial_transformer` scored higher here, while `spatial_filterbank` scored
lower; a model losing on one participant is not an operational reason to stop or
redesign the fixed study. Proceed because the artifacts, identities, selection,
pairing, replay, and probe convergence passed—not because of any positive score.

The exact task binding, decision, device, and machine-readable checks are in
[`pilot-review.json`](pilot-review.json). No model was trained during this
review, and no source, config, tolerance, or readiness artifact was changed.
