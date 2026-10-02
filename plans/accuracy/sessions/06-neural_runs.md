# Session 06 — Connect existing pieces into an auditable neural task

Read session 02–05 interfaces, legacy `inm/study.py`, availability masks, and storage helpers.

Implement run_task and run_smoke and wire CLI dispatch. Real run_task prepares one participant/seed then trains requested neural arms and evaluates selected models under declared validation/test masks. Preserve fixed availability banks across arms. Default matrix still declares covariance but until session 07 requesting that arm must produce an explicit unsupported-arm error; allow selecting the neural subset through a documented config option.

Use a manifest with full configuration, all relevant source hashes (including new package), Python/package versions and data identity. Refuse incompatible reuse, never overwrite another study's calibration, and write result/checkpoint/history atomically. A failed arm is visible and does not generate a valid result. Record complete mask coverage, hashes, selected epoch, metrics, parameter count and timings.

Smoke runs entirely on tiny synthetic CPU data with two epochs maximum and a separate synthetic output path. Include enough class members for all partitions; do not weaken real four-class metric validation. Missing real GDFs must not prevent --smoke. Never manufacture real result records from smoke.

Acceptance: isolated CPU smoke exercises load adapter fixture → fit → save/reload → evaluate; corruption/missing checkpoint rejects completion; changed config/source identity rejects reuse; incomplete fit is not reusable; invalid task fails before writes; synthetic and real outputs cannot collide.

Scope: study/CLI wiring and tests; no classical algorithm, report plots, cluster launcher or large training.

Handoff: exact smoke command, output schema, and how session 07 plugs in the classical arm.
