# Reproducible training and candidate result review

This follow-up turns the completed candidate implementation into a reproducible,
reviewed experiment. The previous coding sessions did not train the real cohort:
its output directory was empty. A separate check found uncontrolled CPU dropout
randomness despite fixed model seeds. These sessions repair that issue and test
public CLI execution before any real training is launched.

| Session | Work | Real experiment prerequisite |
|---|---|---|
| [01](sessions/01-rng.md) | Repair and test scoped RNG and repeatable training | None |
| [02](sessions/02-readiness.md) | Test consecutive CLI tasks/resume; write fresh config and readiness evidence | Real A01 input verification only, no training |
| [03](sessions/03-pilot.md) | Review one five-arm pilot; decide operational proceed or stop | Completed real A01/seed0 pilot |
| [04](sessions/04-findings.md) | Compare candidates and make scientific verdicts | Complete real cohort and report |

Read [CONTRACT.md](CONTRACT.md) and the unchanged
[candidate study contract](../encoder-candidates/CONTRACT.md). Model designs,
CE loss, 250-epoch maximum budget, splits, 135 neural fits, 108 CPU probes and
validation-only comparisons remain fixed. This is not another architecture search.

## Execution sequence

From the repository root, inspect without running models or training:

```bash
bash scripts/run_followup_sessions.sh --list
bash scripts/run_followup_sessions.sh --dry-run
bash scripts/run_candidate_experiments.sh --pilot --dry-run
```

Then run the two bounded repair/readiness coding sessions:

```bash
bash scripts/run_followup_sessions.sh
# Default is sessions 01–02 only.
# Or: bash scripts/run_followup_sessions.sh --session 01
# Then: bash scripts/run_followup_sessions.sh --session 02
```

After readiness is verified, explicitly train the pilot, review it, train the
cohort, and review results:

```bash
# REAL TRAINING: five neural fits and four probes for A01/seed0.
bash scripts/run_candidate_experiments.sh --pilot

# A fresh coding session reviews the saved pilot; it does not train.
bash scripts/run_followup_sessions.sh --session 03

# REAL TRAINING: remaining 26 tasks sequentially, then summary.
# Requires a current pilot-review.json with decision proceed.
bash scripts/run_candidate_experiments.sh --cohort

# A fresh coding session analyses all real evidence; it does not train.
bash scripts/run_followup_sessions.sh --session 04
```

The experiment launcher defaults to CPU. To use CUDA, session02 must first
record successful synthetic CUDA reproducibility checks in readiness.json;
then pass `--device cuda` to both experiment commands. An unavailable GPU is
not silently treated as tested. All real fits should use the same verified
device/environment. Device availability checks do not substitute for numerical
repeatability checks. The launcher sets CUBLAS_WORKSPACE_CONFIG before CUDA use.

`--summarize` validates and aggregates existing outputs without training; an
incomplete report exits nonzero. No mode supplied prints usage and performs no
training. The experiment script never invokes Codex and never retries a failure.
A completed compatible task may be verified and skipped by the study; failed,
partial, corrupt or incompatible tasks stop the run and remain preserved.

## Models and logs

The coding runner uses gpt-6-luna with high reasoning by default, following the
existing workflow. Override --model or AGFL_SESSION_MODEL for a model available
to your account; no pricing or availability claim is implied. Each task receives
a fresh conversation, fixed contract, prior verified handoffs and bounded files.
The stdin/schema invocation is unchanged from the documented earlier runner;
see [official noninteractive documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

```bash
bash scripts/run_followup_sessions.sh --session 01 --model gpt-6.1-sol
```

Coding options also include --from/--through, --effort, --python, --codex,
--timeout (1800 seconds per session/check phase), and --state-dir. Both scripts
accept --python and --state-dir; pass the SAME state directory if customized.
The experiment script has no training timeout; runtime depends on hardware.
Python defaults to AGFL_PYTHON or repository .venv; coding falls back to python3.
Use the configured scientific environment, Python3.12+, Bash, Git, flock and
GNU timeout, plus an authenticated Codex CLI for coding sessions.

Logs/receipts use `.session-runs/candidate-followup/`, with real execution logs
in its `experiments/` subdirectory. Both scripts share the existing workspace
lock with older workflows. Resume rechecks completed coding sessions. Invalid
handoffs, skipped/failed tests, plan edits and missing evidence stop execution.
No real training starts merely because coding sessions 01–02 complete.

## New identity and preserved work

Session02 creates `configs/encoder-candidates-reproducible.json` and readiness
artifacts under `docs/candidate-followup/`. Real outputs go to
`results/encoder-candidates-reproducible-v1`. These config/readiness files are
future session deliverables, so training before their creation must fail.
Existing configs, plans, scripts, receipts and results are preserved. The RNG
fix belongs only to the new candidate package; it does not modify legacy studies.

Readiness includes frozen source/config/package identity, executed checks,
verified devices and hashes of saved verification logs. Real execution rechecks
it before training. Pilot review is bound to the exact task.json checksum;
editing the pilot invalidates the cohort decision. Cohort review requires all
27 real tasks with five fits each, matching masks and verified numeric artifacts.
The pilot gate checks just A01/seed0 directly; it never writes a fabricated
complete-cohort report. Synthetic results cannot pass either evidence gate.

Changes after readiness require renewed verification and a fresh output identity.
The scripts deliberately target the reviewed reproducible-v1 config/output;
repairing a failed real study requires a reviewed path/plan update, not removing
failure records or changing --state-dir alone. Changes to these new plans,
runners/checker, experiment launcher or referenced original candidate contract
invalidate new coding receipts. Older workflow fingerprints are unchanged.

The predeclared screen remains +2 percentage points validation BA and positive
differences for at least6/9 participants. Passing is a reason for independent
confirmation, not proof of strong classification. Keep negative results,
participant variation, training gaps and reused-validation limitations visible.

## Verify the orchestration

```bash
.venv/bin/python -m unittest discover -s tests/automation -p 'test_followup_runner.py' -v
bash -n scripts/run_followup_sessions.sh
bash -n scripts/run_candidate_experiments.sh
```

Tests use fake Codex, fake scientific execution and isolated temporary workspaces;
they never perform paid model calls or real EEG fits. They exercise sequencing,
resume, pilot/cohort separation, checksums, evidence gates, readiness, locks and
fail-stop behavior. They do not certify future scientific findings.
