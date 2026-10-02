# Session 10 integration review

## Scope and result

Reviewed the baseline package and its tests as an independent integration
pass. The package keeps the new baseline study separate from the legacy
experiment. **Software ready for a real pilot; real pilot not run.** Numerical
baseline scores remain unknown until experiments run.

The baseline EEGNet is an architecture reproduction used within this
checkout's declared protocols. The architecture alone does not reproduce a
paper protocol or establish a paper's reported score. In particular, this
study uses participant-specific T-session splits (or a separately configured
T-to-E protocol), its own filtering/training/selection details, and the
declared baseline matrix. It makes no SOTA claim.

## Independent boundary checks

- Train-only normalization was exercised by the preparation regression tests:
  changes to validation/test samples leave training statistics unchanged.
- Raw-input masking is applied before temporal/spatial learned mixing. Tests
  check hidden NaNs/random values cannot change logits and hidden inputs receive
  zero gradient.
- Full and robust epoch-selection tests use validation evidence only; changing
  test labels does not change training or selection. Robust selection equally
  averages the five fixed validation banks and uses mean validation log loss
  as its tie-breaker.
- Checkpoint round trips are tested, and the integration test reloads the
  selected synthetic checkpoint twice and verifies identical predictions.
- Study/result identities, incomplete artifacts, corrupt checkpoints, changed
  manifests, and synthetic/real collisions are covered by regression tests.
- The synthetic smoke is written to a source-identified sibling output and
  records `synthetic: true`. Reporting rejects synthetic records. Integration
  checks verify all 13 neural conditions are present and synthetic smoke data
  does not enter the real summary.
- Existing reporting tests check equal mask-repeat, seed, and participant
  weighting; incomplete cohorts remain blank and the classical arm is
  full-input only.
- A missing-recording integration case confirms preflight names the blocker,
  real task execution leaves a `failed` record with `synthetic: false`, and no
  `result.json` is emitted. A mocked unavailable CUDA device confirms explicit
  CUDA requests fail before creating a study output; no CPU fallback occurs.
- Code inspection confirms real input calls the GDF loader; synthetic data is
  created only by the explicit smoke entry point and its injected fixture.
- The legacy planner remains unchanged at 27 tasks / 378 classifier fits.

No concrete production defect required a model or protocol change. Session 10
adds independent CLI integration checks in
`tests/baselines/test_integration.py`; these exercise plan, smoke, summary,
selected-checkpoint reload, real/synthetic separation, missing-data handling,
and CUDA failure behavior.

## Checks run

Using `AGFL_PYTHON=/home/kalexu97/Projects/AGFL/.venv/bin/python`:

| Command/check | Result |
|---|---|
| `"$AGFL_PYTHON" -m unittest discover -s tests/baselines -p 'test_*.py' -v` | 50 tests passed; no skips |
| `"$AGFL_PYTHON" run.py --plan` | 27 tasks, 378 classifier fits |
| Integration CLI `--plan` against `configs/baselines.json` | 27 tasks, 108 baseline fits |
| `"$AGFL_PYTHON" -m inm.baselines --config configs/baselines-cross-session.json --plan` | 27 tasks, 108 fits; resolved paths printed; explicit E-label prerequisite printed |
| Integration CLI `--smoke`, `--summarize-only`, checkpoint reload | Synthetic CPU smoke completed; 13 neural conditions recorded; loaded predictions matched; real summary counted zero synthetic fits |
| Integration CLI `--preflight` and `--task-index 0 --device cpu` with a temporary missing data directory | Actionable blocked preflight; failed real task record; no complete result |

No GDF files, package installation, GPU scheduling, real pilot, or accuracy
measurement was used in this review.

## Remaining empirical questions and next gate

Software checks cannot answer whether any arm improves balanced accuracy,
whether the architecture is competitive on these recordings, whether
cross-session performance differs from within-session performance, or whether
robust validation selection helps. These remain empirical questions. They must
not be inferred from the synthetic smoke or architecture details.

Before sessions 11–12, produce and inspect a **real complete A01/seed-0 task**
under the declared within-session baseline config. Evidence required:

1. Four arm `result.json` records with complete declared metric coverage and
   non-synthetic data/split/source identities.
2. Readable per-neural-arm `history.json` and selected `checkpoint.pt` files,
   plus the classical fitted state where applicable.
3. A successful `--summarize-only` report whose progress identifies the
   completed pilot without implying full-cohort completion.
4. Validation-only configuration evidence across the declared cohort for any
   selection decision; do not choose settings from one favorable participant
   or test scores.

See [the baseline runbook](runbook.md) for setup, commands, paths, and resume
behavior.
