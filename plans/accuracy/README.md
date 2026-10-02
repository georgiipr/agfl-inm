# Accuracy improvement: sequential implementation sessions

These plans turn the agreed recommendations into small tasks for a less capable
coding model. They are implementation instructions, not completed research or a
promise of SOTA accuracy. The initial deliverable is a simpler, reproducible
baseline study. The existing `inm` tensor experiment remains a reference.

## Session sequence

| ID | Task | Main deliverable |
|---|---|---|
| 01 | Audit current results and environment | Dependency-light diagnostics and explicit unknowns |
| 02 | Declare a separate baseline protocol | Validated configuration and dependency-light CLI |
| 03 | Prepare data and splits | Train-only normalization; within-session and T/E splits |
| 04 | Implement standard EEGNet and masked input | Early spatial filtering; explicit raw-input masking |
| 05 | Implement training and checkpoint selection | Full/robust validation selection; reloadable checkpoints |
| 06 | Connect neural experiments | One-task execution, synthetic smoke mode, provenance |
| 07 | Add a classical covariance baseline | Shrinkage covariance + matrix logarithm + logistic regression |
| 08 | Aggregate comparable results | Complete/partial subject tables and paired comparisons |
| 09 | Diagnose the legacy feature/tensor pipeline | Separate encoder, retrained-head, and compression effects |
| 10 | Review integration and write the runbook | Independent checks and concrete pilot commands |
| 11 | Evidence-gated temporal ablation | Small temporal head, only after real baseline pilot results |
| 12 | Evidence-gated tensor follow-up | Small MHA-only frozen-feature comparison and controls |

Sessions 01–10 are the default batch. Sessions 11–12 require actual experiment
evidence and must report `blocked` if it is absent. They are explicitly selected
with `--through 12` after completing and inspecting the baseline pilot.
Every session has a bounded file scope, proposed interfaces, and acceptance
tests. Read [the shared contract](CONTRACT.md) before editing any research code.

## Run the sessions

The runner creates a **fresh Codex conversation for each session** in this
working tree. It includes the shared instructions, one task, and compact prior
handoffs. It never launches parallel models. Run it from a normal terminal:

```bash
# Inspect the sequence without contacting a model.
bash scripts/run_accuracy_sessions.sh --list
bash scripts/run_accuracy_sessions.sh --dry-run

# Run the ten implementation sessions, resuming completed work automatically.
bash scripts/run_accuracy_sessions.sh

# Run just the first two sessions.
bash scripts/run_accuracy_sessions.sh --through 02

# Use an available model or retry a blocked session with a stronger model.
bash scripts/run_accuracy_sessions.sh --from 04 --through 04 --model gpt-6.1-sol

# Override the default project .venv for Python checks when needed.
bash scripts/run_accuracy_sessions.sh --python /path/to/venv/bin/python

# After the real baseline pilot and its reports exist:
bash scripts/run_accuracy_sessions.sh --from 11 --through 12
```

Default model: `gpt-6-luna`; default reasoning effort: `high`. Both are
overridable (`--model`, `--effort`) and availability depends on your account.
Luna is listed for focused coding in the [official model documentation](https://learn.chatgpt.com/docs/models).
The runner uses [`codex exec`](https://learn.chatgpt.com/docs/non-interactive-mode),
`--output-schema`, and `--output-last-message`, checked against local CLI
0.159.2. It uses workspace-write sandboxing and noninteractive approval policy;
commands requiring unavailable permissions return errors. It never enables
unrestricted execution.

Prerequisites: Bash, Python 3.12+, Git, `flock`, GNU `timeout`, and an authenticated
Codex CLI. The interpreter defaults to `AGFL_PYTHON`, then the repository's
`.venv/bin/python`, then `python3`; `--python` overrides this selection.
For numeric sessions, use an environment containing
`requirements.txt`. This workstation lacked those numerical packages at planning
time. The runner does not silently install packages, download EEG, or submit GPU
jobs; it stops with the session's stated blocker. Optional plotting is unnecessary.
Acceptance tests and synthetic smoke use CPU even with CUDA PyTorch installed.
For real GPU execution, use the [baseline environment guide](../../docs/baselines/environment.md)
and run `bash scripts/run_baselines.sh --check-cuda` from a terminal with GPU access.

## Progress, retries, and model changes

Logs are saved under `.session-runs/accuracy/`. Each attempt has its input prompt,
JSONL model events, stderr, structured handoff, independent verification output,
and pre/post Git status. A completed receipt is written only after both the model
reports completion and the external acceptance checks pass. Empty or entirely
skipped tests do not pass. The runner stops immediately on a failure, timeout,
malformed response, blocked status, or failed acceptance checks.

Rerun the same command to continue. Completed predecessors are rechecked before
they are reused; `--from` cannot bypass an unfinished dependency. If a prompt,
contract, schema, or manifest changes, receipts become stale and the runner stops
instead of silently trusting the previous completion. Review changed plans and
use a new `--state-dir .session-runs/accuracy-v2` to rerun idempotently. Plan files
must not be modified by a running session.

Session receipts certify software checks, **not scientific success**. Later
sessions may edit earlier modules; rechecking tests helps catch regressions but
does not replace review. Use a dedicated branch/worktree if desired. Existing
uncommitted orientation files are permitted and recorded; sessions must preserve
them. No automatic commits, pushes, resets, or cleanup of project files occur.

Each model call has a 30-minute timeout (override with `--timeout SECONDS`).
Retries are manual, so a failing session cannot loop and spend indefinitely.
Timeout limits elapsed time, not tokens or money. Interrupt with Ctrl-C; inspect
partial edits and the attempt log before restarting. Do not edit the checkout
concurrently with a running batch.

## After session 10

Run the generated preflight and synthetic smoke command, then one declared
participant/seed pilot on the experiment machine. Collect training curves,
per-class metrics, full and degraded validation scores, and checkpoint reload
equality before scaling to all nine participants and three seeds. A single pilot
establishes operational readiness only; it is not grounds to claim superiority.
Use validation for architecture/rank choices and keep test results out of those
decisions. The new study's runbook will specify the exact commands and outputs.

## Runner verification

Run `python -m unittest discover -s tests/automation -p 'test_*.py' -v` and
`bash -n scripts/run_accuracy_sessions.sh`. The automation tests use an isolated
temporary Git repository and a fake Codex executable. They cover sequential
handoffs, resume, failures, blocked/malformed responses, skipped/empty tests,
timeouts, model overrides, quoting, stale plans, and concurrent-run locking.
They make no paid model calls and do not claim to test account/model availability.
