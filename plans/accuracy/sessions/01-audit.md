# Session 01 — Audit before changing models

Read: `docs/investigation.md`, `configs/study.json`, `inm/study.py` calibration/result formats, and `inm/reporting.py` output names. Numerical dependencies and recordings may be absent; this session must remain stdlib-only.

Implement `inm/baselines/diagnostics.py` with `inspect_environment(config_path=None)` and `inspect_legacy(output_dir)`. The environment report lists Python/package discoverability, resolved input/output paths, and whether expected recordings exist. Do not import torch or MNE to generate it, scan credentials, install dependencies, or read raw GDF content.

For saved legacy results, extract available per-participant/seed full-input baseline/core scores and encoder selection metrics; explicitly distinguish encoder VALIDATION from downstream TEST scores. Include completion state and source/split/calibration identity. Missing files yield a structured 'unavailable' finding, not zeros or invented diagnoses. Corrupt JSON is a visible issue and never a fabricated success.

Create `docs/baselines/audit.md` with observed readiness, known architectural concerns, and the exact missing evidence. No SOTA judgment without comparable results.

Scope: new package marker, diagnostics module, audit document, and tests only.

Acceptance: synthetic temporary-directory fixtures for missing results, corrupt records, partial results, and validation/test labeling. `test_diagnostics.py` must run without scientific packages. Confirm the legacy planner still returns 378 fits.

Handoff: exact diagnostic JSON keys, artifact formats discovered, and package/data gaps. Do not make later sessions re-investigate these formats.
