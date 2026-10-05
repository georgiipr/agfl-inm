# Session 02 Load paired train and validation data

Bounded paths:

- `inm/encoder_candidates/data.py`
- `tests/encoder_candidates/test_data.py`

Implement PreparedData and prepare_subject using current loading code and the
persisted baseline split metadata. Read inm/baselines/data.py and the original
metadata formats. Do not call its prepare_subject writer or get_split: membership
already exists. Verify stable IDs, labels, cue/channel order, exclusions, recording
checksums, dataset identity and disjoint membership before training-only
normalization. Write new provenance only under the new task directory.

Return only train/validation arrays. Keep the loader injectable for tiny fixtures.
No old checkpoint loading or source-replay bypass is needed. A changed recording,
preprocessing mismatch, wrong subject/seed or reordered membership must fail.

Acceptance: poison held-out signals/labels after fixture loading and prove public
training/validation values/statistics unchanged; poisoned validation must not
change fitted normalization. Test overlap, missing classes, mismatched file
hashes, ordered IDs and immutable input bytes. Include canonical shape/order and
normalization-reference comparisons. Handoff PreparedData fields precisely.
