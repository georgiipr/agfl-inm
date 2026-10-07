# Bounded repair 05-R1: execution admission failures

Read CONTRACT.md, ACCEPTANCE.md, session05 handoff, and receipts05/review.json,
05/post-review.json. This repair is authorized after observed scope/integrity
failures; it is not advancement to integration. One fresh gpt-6-luna/high worker,
30 minutes maximum, supervisor observations at most60 seconds apart. The larger
bound reflects the concrete execution repairs below. No real fitting or E scores.

Preserved attempt: `.session-runs/published-checkpoint-covariance/05/failed-source.zip`.
Original initializer: `02/failed-source.zip::published_covariance/__init__.py`.

Allowed existing edits: session05 execution modules (`protocol.py`, `cli.py`,
`study.py`, `audit.py`, `reporting.py`, `native_reference.py`, `__main__.py`),
new execution helpers/tests, configs/published-covariance.json,
scripts/run_published_covariance.sh, docs/published-covariance/runbook.md.
Explicitly restore and narrowly change `published_covariance/__init__.py` to lazy
exports preserving all original public names while allowing stdlib-only plan.
Write new docs/published-covariance/session-05-R1-handoff.md. Do not alter accepted
data/masks/classifier/atcnet_native/completion/training code or previous handoffs,
tests, plans, supervisor machinery or historical outputs.

Required repairs:

1. Restore initializer public API. Required source identities include it and all
   implementation files, launcher, declared plans/contract/audit recipe, complete
   installed package versions and concrete device/runtime flags. Verify the exact
   required nine H5/source hashes, not a faulty subset test or origin-only claims.
   Preserve actual MAT-byte checks. Record all frozen numeric/mask-key settings in
   config, consistent with accepted scientific code. No scientific settings change.
2. Audit must reject incomplete/duplicate/foreign cell indexes even on its first
   invocation, rederive official IDs/labels/masks/probabilities, check saved cell
   metrics too, and bind all existing immutable numeric artifacts by hashes.
   Summaries require identity-bound complete audit evidence and must reject changed
   probability bytes even when metrics/row index are unchanged. Incomplete studies
   may give explicit incomplete reports with no cohort estimate, never a pass.
   Completed task/pilot/seal reuse must remain independently replayed, no overwrite.
   Avoid rereading/rebuilding all891 cells merely to check a summary if an audited
   whole-output numeric manifest provides the correct byte binding.
3. Enforce safe root and nested paths/symlinks in every write; support a documented
   fresh real output name after source/runtime changes without targeting historical
   artifacts. Keep missing/partial/foreign/corrupt outputs fail closed.
4. Real launcher must run each fit and each task audit/reference in separately
   bounded children (3600s fit,600s audit/reference); hold shared lock across the
   operation, preserve time/command/exit logs, stop on failures. Gate remaining24
   fits on a reviewed pilot before any fitting; audit each fit before moving on.
   Avoid one all27/891cell audit/reference child under600s. Provide task modes and
   immutable aggregation of verified receipts as needed; no unbounded retries.
5. Preserve complete source-backed native T/E references and exact full-input arm
   equality. Fit/E audit receipts must bind artifacts and identity; native E scoring
   remains impossible before all27 selected states sealed. Native parent/child GPU
   allocation uses memory growth before initialization and rejects missing CUDA.
6. Human report must contain real numeric tables when complete, all participant
   effects incl negatives, learned-fixed/zero, percondition BA/accuracy/logloss and
   degradation. Report selected epochs, epoch0/saturation counts, T selection scores
   and native inclusion/exclusion identities. Record measured completion latency
   separately from classifier inference and parameter count; do not invent numbers.
   Persist and report frozen-classifier state hashes before/after actual covariance
   fitting. Mark synthetic reports/data explicitly; generate distinct T/E fixture
   trials/IDs to exercise the separation. All disclosures remain gradeB/exploratory.

Acceptance: independent acceptance_execution.py, acceptance_execution_faults.py,
acceptance_launcher.py, acceptance_reporting.py, check_plan.py, package unit suite,
shell syntax, protected inventory; supervisor source review remains mandatory.
Workers may run checks but cannot edit them. Use final-source checks, and stop
editing before claiming a lifecycle pass. Native-checkpoint synthetic CUDA
fit/save/replay is session06 integration work and remains a separate gate.
