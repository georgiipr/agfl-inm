# Session 02 Read verified artifacts without exposing test data

Implement `inm/encoder_audit/artifacts.py` and
`tests/encoder_audit/test_artifacts.py`; update only inventory capability wiring
if necessary. Read both studies' actual manifests, calibration/checkpoint
formats, dataset records, persisted splits, and saved history selection fields.

1. Verify original identities and checksums before loading. Establish explicit
   types for source metadata, train/validation IDs, labels, cached features,
   normalization, encoder/factor state, and baseline selected checkpoints.
2. Public views must not include test arrays/labels/metrics. A combined legacy
   container may be opened to slice approved partitions immediately; never
   propagate the complete cache. Derive labels from verified source metadata or
   the loader when needed; never guess labels from row order.
3. Validate unique IDs, index bounds, disjointness, original subject/seed,
   four-class mapping, channel order, shapes, and historical replay source files.
   Return a specific capability limitation for missing original heads.
4. Do not require current whole-tree identity to equal a historical manifest:
   new audit files were absent historically. Compare the historical files needed
   for replay and record both source identities. Unsupported format is an error.

Acceptance: corrupt hashes are rejected before deserialization; swapped task
identities and overlapping splits fail; poisoned test labels/features/metric
values cannot alter returned views. Input-directory files stay byte-identical.
Run `test_protocol.py` and `test_artifacts.py`. Handoff exact dataclass fields and
the authoritative route used to recover aligned train/validation labels.
