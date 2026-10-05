# Session 07 Integrate reporting and write the execution runbook

Implement `inm/encoder_audit/{study,reporting}.py`, complete `__main__.py`, and add
`tests/encoder_audit/test_integration.py`, `docs/encoder-audit/runbook.md`, and
`docs/encoder-audit/review.md`. Earlier new-package repairs are allowed with tests.

1. Connect one task, independent synthetic smoke, and summary commands. Persist
   atomic identities, task records, numeric probes, per-sample diagnostics,
   errors, and the exact evidence schema from CONTRACT.md. Implement safe resume
   with checksums; reject output/input overlap, incompatible reuse, and real/synthetic
   mixing. No original study writers may be called.
2. Aggregate repeat → seed → participant, keep conditions/views distinct, and
   report unavailable/mismatched comparisons. Withhold incomplete cohort means.
   Validate task artifacts and source/config/package identities before accepting
   complete evidence. Preserve unfavorable results and convergence failures.
3. Write exact commands for inventory, synthetic smoke, a manual A01/seed-0
   pilot, inspecting its reports, running the remaining 26 tasks sequentially,
   summarizing, and invoking coding session 08. Explain CPU resource use and the
   separate output location. Freeze source after this session before real runs.
4. Run the whole new test suite and existing 65 baseline tests. Run only synthetic
   smoke into a temporary output. Verify evidence-gate compatibility using fixtures,
   but do not create a synthetic file at the real evidence path. The real evidence
   absence is expected here and must not block software completion.

Acceptance: end-to-end synthetic task and smoke, resume equality, tampered
artifact/config/source rejection, partial coverage, unequal subject sample counts,
duplicate tasks, pairing conflicts, and input preservation. Tests verify no test
metrics escape into reports. Handoff runnable commands and any remaining limitations.
