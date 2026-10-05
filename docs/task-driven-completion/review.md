# Session 04 integration review

The frozen implementation passed synthetic CPU integration and is eligible for
a separately authorized A01 seed-0 pilot. It is not ready to launch the cohort
without pilot review, and there are no new real completion accuracy measurements.
CUDA and the real completion execution/audit paths remain unverified.

The integration adds `inm.task_driven_completion.audit.audit_task` and the explicit
`python -m inm.task_driven_completion.audit` CLI. The audit compares numeric
backbone state with an independently supplied or historically restored backbone,
loads all five saved selected states, recomputes all 105 predictions from inputs,
and independently calculates class-balanced accuracy, log loss and five degraded
task contrasts. It leaves study artifacts unchanged and never calls calibration
or supervised fitting. Real readers stage only train/validation data externally;
the coding session executed only synthetic audit inputs.

Seven new tests cover fresh/resume/audit process boundaries, changed config
identity rejection, rehashed selected-state and backbone tampering, no fitting by
audit, unequal class counts, unequal participant trial counts, negative effects,
and suppression of incomplete cohort estimates. The two rehashed-tampering tests
demonstrate why the separate audit matters: structurally valid artifacts and the
stored replay assertion alone do not establish numeric replay.

Worker acceptance: **60 tests, no skips**, 23.793 seconds. Supervisor acceptance:
**60 tests, no skips**, 23.772 seconds. Final fresh smoke/resume/audit each exited 0
in separate processes with 105 cells, five strategies and two synthetic completion
fits; audit performed zero fits and reproduced probabilities exactly. A changed
identity was rejected with expected exit 2. Source was frozen before this final
smoke. Logs, identities, command receipts and checksums are in
[readiness.json](readiness.json) and [validation](validation/).

Regression evidence was independently produced by the supervisor. Supervised
Tucker passed 25 tests. The raw completion-Transformer suite had 35 passing tests
and one error because its declaration test assumes an empty historical output;
`results/completion-transformer-v1` was already completed before this work. That
unchanged test passed with its byte-identical declaration in an isolated temporary
fixture, using the unchanged loader and occupancy assertions. Both raw failure
and isolated pass are retained; this is not described as an unqualified raw
36-test pass. No old source, tests, configuration or historical artifacts changed.

The supervisor separately verified all 27 historical input bundles and original
A01 full-input prediction replay for both historical backbones. Copies are clearly
named `supervisor-*` in validation. Those results are historical-input readiness,
not new completion measurements. The worker did not repeat real recording reads.

See [runbook.md](runbook.md) for explicit plan/preflight, synthetic checks, frozen
identity verification, and separate pilot → review → cohort → audit/report steps.
Maintain fixed settings and preserve failed attempts. Reused validation selection,
exploratory unadjusted intervals and lack of independent confirmation remain
scientific limitations even after a future complete real run.
