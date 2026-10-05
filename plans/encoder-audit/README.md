# Encoder investigation in sequential sessions

This plan investigates whether the channel/window-local encoder retains useful
class information, and whether its limitation precedes Tucker reconstruction.
It creates diagnostic tools first; real measurements and their interpretation
follow separately. No reconstruction-error threshold is treated as proof of
classification quality, and no architecture improvement is promised.

The eight tasks are deliberately narrow for weaker coding agents. Each starts
in a fresh conversation, receives the shared contract and prior verified
handoffs, and must pass external acceptance checks before the next task runs.
Read [CONTRACT.md](CONTRACT.md) for the complete scientific and API requirements.

| Session | Deliverable |
|---|---|
| [01](sessions/01-protocol.md) | Fixed audit protocol, CLI plan, and input inventory |
| [02](sessions/02-artifacts.md) | Verified artifact readers exposing training/validation only |
| [03](sessions/03-alignment.md) | Cue, split, normalization, and frozen-feature replay checks |
| [04](sessions/04-checkpoints.md) | Clean evaluation of saved baseline checkpoints |
| [05](sessions/05-probes.md) | Three fixed linear probes and a shuffled-label control |
| [06](sessions/06-reconstruction.md) | Paired reconstruction and classification diagnostics |
| [07](sessions/07-integration.md) | One-task execution, honest aggregation, smoke, and runbook |
| [08](sessions/08-evidence.md) | Review complete real evidence and predeclare at most one next question |

The original pretraining head was not saved with the legacy encoder. These
sessions must expose that limitation rather than invent its predictions. Saved
spatial EEGNet classifiers are replayable. New linear probes measure information
accessible to those probes, not all possible nonlinear class information.

## Run one coding session at a time

From the repository root, inspect the plans without model calls:

```bash
bash scripts/run_encoder_sessions.sh --list
bash scripts/run_encoder_sessions.sh --dry-run --session 01
```

Then execute one bounded session. The default model follows the existing
project workflow, `gpt-6-luna`, with `high` reasoning; override it with a model
available to your account. No model-availability or pricing claim is implied.

```bash
bash scripts/run_encoder_sessions.sh --session 01
bash scripts/run_encoder_sessions.sh --session 02

# Retry an unfinished task using a different model.
bash scripts/run_encoder_sessions.sh --session 02 --model gpt-6.1-sol

# Explicit interpreter, if the default project environment is unsuitable.
bash scripts/run_encoder_sessions.sh --session 03 --python /path/to/venv/bin/python
```

`--session` is shorthand for equal `--from` and `--through`. Do not combine it
with those range options. Completed predecessors must exist and still pass.
For a sequential batch of implementation sessions, run:

```bash
bash scripts/run_encoder_sessions.sh
# Equivalent default range: 01 through 07.
```

This starts only one model at a time. It does not run real probe fits, retrain
neural networks, install packages, download data, or launch the scientific study.
No paid model calls were made to test this automation.

## What happens after session 07

The implemented runbook at `docs/encoder-audit/runbook.md` will provide the exact
real-data commands. First run inventory and synthetic smoke, then a real
A01/seed-0 diagnostic task and inspect it. With that operational check complete,
run the remaining 26 tasks sequentially and summarize into the fresh
`results/encoder-audit-v1` directory. Freeze analysis source across that run.

The real cohort entails 108 small fixed logistic probe fits on CPU, replay of
81 saved neural baseline checkpoints, and train/validation-only diagnostics.
It does not refit the original encoders, Tucker factors, or neural classifiers.
Real runtime depends on input size, replay cost, and the machine; none is claimed
from synthetic tests.

Only after the real report exists:

```bash
bash scripts/run_encoder_sessions.sh --session 08
```

Before calling a model, the runner requires a complete real 9-participant,
3-seed audit with consistent train/validation identities, current source hashes,
27 verified task files, and successful integrity checks. A partial pilot or
synthetic report cannot pass. Session 08 may conclude that no intervention is
justified; it never starts an architecture search or fits a new model.

## Resume and troubleshooting

Logs and receipts live in `.session-runs/encoder-audit/`. Each attempt retains
the exact prompt, model events, stderr, structured handoff, pre/post Git status,
and independent verification output. Re-run the same command after inspecting
a failure. There are no automatic retries or silent continuation on blocked,
malformed, failed, skipped, or empty checks. A completed task is rechecked rather
than fitted by a new coding model again.

The final handoff's `blockers` describes unfinished work in that coding session.
A completed handoff requires `blockers: []`, a summary, and actual checks. An
inventory may correctly find that real replay is blocked by incompatible inputs;
retain that finding in the inventory and `next_session_notes`. Passing software
acceptance does not clear the real replay blocker. A contradictory handoff needs
review and independent acceptance before recovering a receipt; do not simply
delete blockers to force continuation.

Use `--timeout SECONDS` to change the default 1800-second bound for each model
call and acceptance phase. Ctrl-C preserves partial work and logs. Do not edit
the checkout while a batch runs. Both this runner and the original accuracy
runner share the existing `.session-runs/accuracy.lock`, so different state
directories cannot bypass workspace locking.

All plan Markdown/JSON and this runner/checker are fingerprinted. Changes make
receipts stale; review them and use a new `--state-dir .session-runs/encoder-audit-v2`
to rebuild acceptance history. Changed scientific source/config/packages require
a new scientific output identity too; changing the automation state directory
alone does not authorize reusing incompatible research outputs. The session-08
evidence path is declared in `manifest.json` and must match the reviewed run.

Prerequisites: Bash, Git, Python 3.12+, `flock`, GNU `timeout`, and authenticated
Codex CLI. Python defaults to `AGFL_PYTHON`, then `.venv/bin/python`, then
`python3`. `--codex` or `AGFL_CODEX_BIN` can select the executable;
`--model` or `AGFL_SESSION_MODEL` selects the coding model. Numerical sessions
need the existing scientific dependencies. Missing sandbox permissions produce
an error in noninteractive mode, not a permission bypass.

The CLI invocation uses `codex exec`, stdin prompts, structured JSON output,
`--output-last-message`, and a workspace-write sandbox, checked against local
CLI 0.160.0 and the [official noninteractive documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

## Automation verification

```bash
python -m unittest discover -s tests/automation -p 'test_encoder_runner.py' -v
bash -n scripts/run_encoder_sessions.sh
```

Tests use temporary Git workspaces and a fake Codex executable. They exercise
sequencing, model changes, predecessor checks, resume, timeouts, locking,
invalid responses, missing/failed/skipped acceptance, plan integrity, and real
evidence gate failures. These verify orchestration, not scientific conclusions.
The old accuracy runner and its plan files remain unchanged, preserving their
existing receipt fingerprints.
