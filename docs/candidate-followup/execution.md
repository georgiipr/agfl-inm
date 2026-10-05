# Candidate execution after readiness

The candidate contract fixes nine participants and three seeds: 27 tasks, five
neural fits per task (135 total), and four local logistic probes per task (108
CPU fits total). This coding session performed synthetic software checks and
verified A01/seed0 inputs; it did not train a real EEG model.

## Review sequence

Run the explicit pilot first, then review its saved evidence:

```bash
bash scripts/run_candidate_experiments.sh --pilot
bash scripts/run_followup_sessions.sh --session 03
```

The pilot fits A01/seed0's five neural arms and four local probes on CPU. Inspect
the review's checks, provenance, selected epochs, paired masks and artifact
hashes. A `stop` decision ends real execution. Only a verified `proceed` review
authorizes the explicit cohort command:

```bash
bash scripts/run_candidate_experiments.sh --cohort
bash scripts/run_followup_sessions.sh --session 04
```

The cohort command processes the other 26 tasks sequentially and summarizes once.
The final session reviews all 27 tasks and their 135 neural and 108 probe fits.
There is no implicit training command, automatic retry, or permission to replace
failed output. Complete compatible tasks can be verified and skipped; partial,
failed, corrupt, checksum-mismatched, source/config/package-mismatched, or
synthetic artifacts stop execution and remain available for review.

## Readiness and device

The fixed experiment config is
`configs/encoder-candidates-reproducible.json`, with output
`results/encoder-candidates-reproducible-v1`. The launcher verifies the frozen
source/config/package identity and readiness hashes before every real task.
This session verified CPU only. CUDA must not be requested unless a future
readiness artifact records successful tiny CUDA repeatability checks; pilot and
cohort must use the same verified device type.

No candidate training occurred during session 02. Real fits begin only with the
explicit `--pilot` command after this readiness review.
