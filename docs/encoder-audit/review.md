# Session 07 integration review

Session 07 connects the verified source readers, alignment audit, clean saved
checkpoint replay, four fixed legacy probes, and paired validation
reconstruction into one task receipt. Outputs are written atomically and include
the fixed task schema, config/source/package identities, original manifest and
artifact checksums, probe numeric NPZ files, train/validation sample
diagnostics, tolerances, timing, explicit Boolean checks, and retained errors.
Existing study writers are not called.

Task reuse checks the audit identity and hashes for every saved numeric output.
It refuses incompatible or incomplete task directories. Report generation
validates all task identity fields, source-file digests, partitions, exact check
keys, and artifact hashes before including a task. Synthetic tasks cannot enter
real evidence. Failed real tasks are retained in the task directory and listed
separately in partial reports; they are never accepted as evidence tasks. The
report schema matches the session-08 gate; a partial report
is permitted, but cannot satisfy the real-cohort gate.

The report separates condition and classification view, keeps repeats inside
their subject/seed task, averages seeds within participants, and weights
participants equally. Missing task coverage suppresses cohort means. A
cross-study mismatch is listed as unavailable rather than discarded as a
within-study failure. Test rows, labels, and metrics are never summarized.
Validation findings remain exploratory because validation also selected the
historical checkpoints.

Synthetic fixtures cover task serialization and resume equality, output tamper
detection, changed config/source identity rejection, incomplete coverage,
duplicate task configuration, unequal trial counts across participants,
pairing conflicts, no test-metric escape, and smoke output isolation. Their
acceptance establishes software behavior, not a research result. The runbook
requires a human-inspected A01/seed-0 pilot before the remaining tasks and asks
that source/configuration stay frozen through the real run.

## Readiness inherited from inventory

The real analysis is not certified ready by this implementation. Inventory
records baseline historical source mismatches in `inm/baselines/diagnostics.py`,
`inm/baselines/eegnet.py`, `inm/baselines/protocol.py`,
`inm/baselines/reporting.py`, `inm/baselines/study.py`, and `inm/training.py`.
In particular, replay dependencies `inm/baselines/eegnet.py` and
`inm/training.py` prevent faithful baseline checkpoint replay. Real task output
must preserve that failure. Do not silently use expanded-checkout code or call
the replay ready. No real task, recording load, neural fit, or GPU job is part
of session 07.

See [the execution runbook](runbook.md) for preflight, isolated smoke, pilot,
sequential cohort commands, summary, and session-08 evidence review.
