# Corrected baseline: completed cohort and optional next step

The corrected study in `results/baselines-reproducible-v1` contains 108/108
valid fits: nine participants, three seeds, four arms. Task 0 is included.
Rebuilding its report succeeded with no missing, invalid, failed, or excluded
neural pairs. Sessions 01–10 receipts remain valid.

Full-input balanced accuracy (%), averaged over seeds within participant and
then equally over participants:

| Arm | Validation | Test |
|---|---:|---:|
| EEGNet reference / full training | 57.43 | 51.17 |
| Mask-conditioned EEGNet / full training | 60.86 | 53.40 |
| Mask-conditioned EEGNet / mixed training | 51.08 | 43.52 |
| Covariance classifier / full training | 56.12 | 56.35 |

Use the corrected study for subsequent diagnostics; the previous study is
archived evidence of the earlier initialization behavior. Both studies reuse
the same recordings and split seeds, so they are not independent datasets.
Test values are evaluation outcomes, not permission to tune against test data.

Clean training balanced accuracy at selected checkpoints is 88.35%, 91.28%,
and 73.53% for the three neural arms respectively. Their sizable gaps to
validation warrant training/generalization investigation; greater complexity
is not justified merely by completing the baseline. For the five allowed
robustness validation banks, descriptive means are 38.65%, 40.05%, and 41.20%.
All these checkpoints were selected using full-input validation, not robust
selection. No SOTA comparison or significance claim follows from these values.

## Continuing the approved implementation plan

The real-pilot prerequisite for optional session 11 is satisfied. To implement
the planned temporal-head comparison, run only that coding session:

```bash
bash scripts/run_accuracy_sessions.sh --from 11 --through 11
```

The session must treat the corrected study as the current baseline evidence,
retain the flatten head, keep the initialization fix, and introduce just the
planned small temporal head with a new output identity. It must document a
validation-only comparison rule before experiments and keep the CNN front-end,
splits, preprocessing, and training budget matched. The existing train/validation
gap means this is an optional testable hypothesis, not an established remedy.
An investigation of training/generalization remains a reasonable priority over
any architecture extension. Do not use 11-channel/spatial validation conditions
or test scores to choose settings.

Session 11 implements and tests ablation support; it does not run the real GPU
ablation. Read its resulting `docs/baselines/temporal-ablation.md` before using
the new experiment commands. Do not change the completed baseline config or
write new fits into its output directory after source changes.

Session 12 is a separate optional tensor investigation. No legacy experiment
results are present to ground that diagnosis. Cross-session evaluation likewise
remains separate: all E recordings are present, but official E-label files were
not found under the data directory and must be explicitly provided.
