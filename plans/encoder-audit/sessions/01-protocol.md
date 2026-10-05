# Session 01 Declare the audit and inventory its inputs

Implement only `inm/encoder_audit/{__init__,__main__,protocol,inventory}.py`,
`configs/encoder-audit.json`, `docs/encoder-audit/inventory.md`, and
`tests/encoder_audit/test_protocol.py`. Read the shared contract, existing
manifest formats, and `inm/baselines/diagnostics.py` before coding.

1. Implement the fixed config, strict validation, config-relative paths, stable
   task enumeration, and lazy CLI. Plan must say 27 tasks, 108 CPU probe fits,
   zero neural refits, train/validation only. Validate path containment in both
   directions, resolving symlinks. Reject output equal to the repository root.
2. Inventory JSON metadata, source maps, recorded checksums, coverage, saved
   checkpoint availability, dependencies, and data paths. Do not deserialize
   tensors or summarize test scores. Missing artifacts get explicit structured
   blockers. The legacy pretraining and downstream heads are not recoverable
   from the encoder-only state: expose this limitation as a capability field.
3. Document expected inputs, source identities actually observed, unavailable
   capabilities, and that no audit measurement or probe fit was performed.
   Do not regenerate or update original experiment reports.

Acceptance: planning works with numerical imports blocked; invalid parameters,
test partitions, mixed input protocols, unknown search settings, missing files,
and input/output overlap fail clearly. Fixtures exercise a source study with an
encoder-only checkpoint and ensure it is not labeled a reloadable classifier.
Run `test_protocol.py` and the legacy planner. Handoff config keys and CLI output.

An inventory that correctly rejects incompatible historical source hashes is a
valid deliverable. Keep those structured inventory blockers and its nonzero exit
status. If all Session 01 implementation and acceptance requirements pass, return
`completed` with `blockers: []` and describe the real-replay limitation in
`next_session_notes`. Do not mark real replay ready or repair historical inputs.
