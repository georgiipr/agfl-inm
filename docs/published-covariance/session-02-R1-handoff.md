# Session 02-R1 handoff: data-boundary repair

**Decision:** the four independent boundary defects are repaired and all three
required data suites pass. This handoff supersedes no prior receipt; the original
session-02 handoff and archived failed source remain unchanged.

`published_covariance/data.py` now validates label, trial-start, sampling-rate
and artifact metadata as finite integral scalars before conversion. It requires
the author's entire 1,750-sample outer trial window and fails on an incomplete
source trial. Native extraction preserves raw NaN/Inf values without arithmetic;
the explicit normalizer selects observed channels before arithmetic, rejects
nonfinite observed values, and still rejects nonfinite fit inputs. Normalizer
save validates lowercase SHA-256 and subject/T trial identities, validates the
numeric arrays, and creates its target exclusively. Load verifies schema,
subject/session/trial-ID consistency, the ordered-ID digest, source hash and
finite numeric arrays with positive scales.

`docs/published-covariance/data.md` now uses the native time-first crop
`X[start:start+1750, :22][375:1500, :].T`, describes fail-closed bounds and
mask-boundary finiteness, and records the completed supervisor alignment
receipt. That receipt covers all 18 A01–A09 T/E recordings (288 trials each,
zero exclusions), with original GDF channel/time agreement at the predeclared
`1e-8` microvolt / `1e-10` relative tolerance, T labels aligned to GDF events,
and E labels aligned to official label-only MAT labels. No classifier scoring,
covariance fit or outcome metric was run here.

Using `.venv-published-covariance/bin/python`, the required suites passed
independently:

- `-m unittest discover -s tests/published_covariance -v`: 9 tests passed.
- `PYTHONPATH=. .venv-published-covariance/bin/python plans/published-checkpoint-covariance/acceptance_data_boundaries.py`: 4 tests passed.
- `PYTHONPATH=. .venv-published-covariance/bin/python plans/published-checkpoint-covariance/acceptance_data.py`: 5 tests passed, including the A01 T comparison against the pinned native loader.

The new local cases cover fractional metadata, the full outer-window bound,
normalizer overwrite and false subject/session provenance, altered ID hashes,
unsupported schema, and invalid saved numeric arrays. The independent boundary
suite verifies hidden nonfinite raw samples are invariant after masking and
observed nonfinite values fail at normalization. Real all-subject alignment is
supervisor-owned evidence; it is not a worker scoring result.
