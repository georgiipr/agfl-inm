# Session 08 Integrate tasks and provenance-safe resume

Bounded paths:

- `inm/encoder_candidates/study.py`
- `inm/encoder_candidates/__main__.py`
- `tests/encoder_candidates/test_study.py`

Connect prepare_subject, five model fits, selected evaluation and four probes
into task execution. Persist the EXACT task/fit artifacts in CONTRACT.md. Make
checks record executed verifications, never constant defaults. Save all split,
recording, config, source and package identities; compare them before any reuse.
Record failure status/errors even if later arms fail. Completed arms may be reused
only after every required hash/identity verifies; do not delete failed outputs
or silently repeat fits. Explain how to restart failed work under a new output.

Wire --task-index, --device and --smoke. Implement an independent tiny synthetic
smoke that actually trains/reloads all five models and fits probes, not a
hand-authored success JSON. At this session --summarize-only may still fail as
unimplemented. Session 09 completes the report. Block smoke output overlap with
inputs/real output and mark every synthetic artifact.

Acceptance: fake fixture loader + real tiny numeric pipeline, immutable inputs,
135/108 cohort budget consistency, same task result on compatible resume,
changed source/config/package rejection, checksum tampering, failed/partial arm
handling, path containment and test-field absence. All actual checks must pass.
