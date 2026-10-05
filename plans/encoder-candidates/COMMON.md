# Instructions for each encoder candidate session

Implement exactly this session. Read root AGENTS.md, docs/README.md,
docs/onboarding.md, docs/investigation.md, the shared contract, the current
session and previous handoffs. Inspect Git status and preserve prior uncommitted
work, including PDFs and both existing workflows. No commits or pushes.

1. Work only in the session's named paths. Necessary earlier-package repairs
   are allowed with explanation and regression checks. Never edit old scientific
   packages, configs/results, any session plan, runner/checker, or automation tests.
2. Use AGFL_PYTHON. No subagents, nested runners, package installation, downloads,
   real EEG training, real probes, or GPU jobs. Sessions 01–09 are implementation
   with tiny CPU synthetic checks. Session 10 reads real evidence only.
3. Missing numerical dependencies block numerical sessions. Missing real data
   does not block tested software that explicitly reports unavailable inputs.
   Missing real cohort evidence blocks session 10. Distinguish current-session
   acceptance failures from downstream input-readiness findings.
4. Tests must exercise behavior, leakage prevention, masking, gradients, reload,
   mismatched provenance and incomplete evidence as applicable. No skipped tests,
   invented measurements, or claiming a static syntax check is numerical evidence.
5. Keep public APIs/data schemas fixed. Imports for plan/preflight stay lazy.
   Do not implement later modules as placeholder successes. Unimplemented actions
   must fail explicitly. Read existing code before reusing its interfaces.
6. Run current and affected earlier tests, inspect the diff and git diff --check.
   Leave original inputs byte-identical. Preserve any failed implementation edits
   and report what remains. No test-based architecture or epoch selection.
7. Return only the supplied JSON response with the exact session ID. completed
   requires a nonempty summary, actual executed checks, and blockers: []. Current
   unfinished work/failed acceptance means blocked. Real-input readiness findings
   remain in preflight/docs and next_session_notes even when software completed.
   Do not clear real failures to make a session pass. Handoff exact APIs and
   constructor/config choices needed by the next weaker agent.

The runner independently verifies artifacts and tests. A receipt certifies
software acceptance only, not classification improvement or a completed study.
