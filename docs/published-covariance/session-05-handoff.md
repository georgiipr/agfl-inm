# Session 05 handoff: execution and reporting

**Status: implementation delivered; integration acceptance remains open.** No
real covariance fits or classifier outcome scores were run. This handoff does
not claim CUDA native readiness or real-cohort readiness.

The new execution layer provides a dependency-light `plan`, explicit
preflight/smoke/reference/pilot/review/train/evaluate/audit/summarize commands,
identity-bound output roots, exclusive NPZ/JSON persistence, a pilot review
gate, a 27-task fit seal, paired arm evaluation, saved-mask/state replay, and
hierarchical participant reporting. The real path hashes the pinned upstream
source/checkpoint files and all 18 MAT recordings when it constructs the study
identity; the data loader also checks each recording against its origin
manifest when used. The native T references use an isolated subprocess running
the author's unmodified graph. E full-input native references are created only
after the complete fit seal. Checkpoint grade B and unknown prior E exposure
are included in human and machine reports.

The synthetic lifecycle acceptance passed once during implementation, but the
supervisor invalidated that receipt because source files changed during that
run. The updated full lifecycle acceptance was not rerun after the final source
edits. The supervisor's persistence fault acceptance passed 3 cases before the
last source edits; treat that receipt as provisional. Latest direct checks were:

- `bash -n scripts/run_published_covariance.sh` — passed.
- `python3 -I -S plans/published-checkpoint-covariance/check_plan.py` — passed: 27 covariance fits, 0 backbone fits, 81 arm evaluations, with no scientific imports.
- `.venv-published-covariance/bin/python -m unittest discover -s tests/published_covariance -v` — 19 tests passed.
- Reporting acceptance — 2 tests passed before the final secondary-report fields were added; rerun during integration.
- Launcher acceptance — 2 tests passed, including occupied shared lock and timeout descendant cleanup.

No real `preflight`, native CUDA smoke, source-parity reference run, pilot, or
E evaluation was performed by this session. The GPU runtime was provisioned by
the supervisor, but physical CUDA was unavailable in this coding worker. Run
the exact commands in [the runbook](runbook.md) on the target GPU and preserve
their receipts before treating the implementation as operational. In
particular, the finalized synthetic lifecycle, adversarial persistence suite,
actual native-source parity subprocess, GPU allocation behavior, and complete
independent audit all still need supervisor acceptance against the final source
identity. No accuracy conclusion follows from the synthetic checks.
