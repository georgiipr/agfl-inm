# Session 03 Review the real pilot before cohort execution

Required artifacts and bounded paths:

- `docs/candidate-followup/pilot-review.md`
- `docs/candidate-followup/pilot-review.json`

Read the complete real A01/seed0 five-arm task from the reproducible output.
Verify all hashes/provenance and RNG metadata, train/validation membership,
checkpoint selection, clean metrics, reload predictions, local probe convergence
and mask pairing. Preserve any original failure journals. Do not train or change
scientific source/config. Use existing readers/evaluators only where faithful.

Report five model scores and gaps descriptively, with explicit single-participant
limits. Decide operational proceed/stop; a model losing is not a reason to stop
or redesign a pilot. An actual identity, leakage, reproducibility or convergence
failure is. Write pilot-review.json bound to exact task bytes and current study
ID. A stop review is a valid completed review but prevents the cohort launcher.
Do not write a full-cohort evidence report or select an architecture here.
