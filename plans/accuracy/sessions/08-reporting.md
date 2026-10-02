# Session 08 — Comparable complete and partial summaries

Read baseline result schema and `inm/reporting.py` aggregation conventions. Implement summarize and --summarize-only dispatch without importing the GPU/training stack unnecessarily.

Write summary.md, per-run and per-participant CSVs, cohort full/degraded tables, full-input degradation, and paired neural differences where coverage/masks agree. Average repeats then seeds within participant then participants equally. Classical model participates only in full-input tables. Missing/corrupt/incompatible cells remain visible; cohort means stay blank until declared tasks complete.

Never aggregate across different split protocols, selection policies, preprocessing, budgets, dataset identities, or synthetic flags. Verify pairing by subject, seed, data/split identity and exact mask hashes. If intervals are implemented, use participant-level paired bootstrap with fixed seed and declared repeats; do not treat mask repetitions as extra participants.

Acceptance: handcrafted result fixtures with analytically known repeat/seed/subject means; unequal trial counts must not change participant weights; missing one seed blanks complete-cohort aggregate; mask/provenance mismatch excludes a pair visibly; full-only classical coverage is valid; synthetic artifacts cannot appear in real reports; corrupt records do not silently count as complete.

Scope: reporting, CLI summary hook, tests. No plotting dependency required.

Handoff: report schema and entry paths for real pilot review and later session gates.
