# Encoder architecture comparison sessions

These ten bounded sessions implement and evaluate three candidate architectures:
a local multiscale power encoder, a filter-bank spatial model, and a compact
EEGNet–Transformer. Fresh local and spatial controls make five neural arms.
Read [CONTRACT.md](CONTRACT.md) for the fixed designs, data boundaries, losses,
training budgets, comparisons and evidence schema. No improvement is promised.

The default runner performs coding sessions 01–09 only, one fresh conversation
at a time. These use synthetic CPU tests. Real experiments are a separate,
manual step after implementation; session 10 requires their complete evidence.
No coding agent may launch real EEG fits or enlarge the fixed experiment.

| Session | Bounded deliverable |
|---|---|
| [01](sessions/01-protocol.md) | Fixed config, identities, lazy plan and preflight |
| [02](sessions/02-data.md) | Verified paired splits and training-only normalization |
| [03](sessions/03-local.md) | Existing local control and new local power encoder |
| [04](sessions/04-spatial.md) | Spatial EEGNet reference and filter-bank candidate |
| [05](sessions/05-transformer.md) | Compact Transformer head on the same EEGNet front-end |
| [06](sessions/06-training.md) | Fixed supervised training, selection and checkpoint reload |
| [07](sessions/07-diagnostics.md) | Frozen-feature probes and raw-input electrode-loss evaluation |
| [08](sessions/08-study.md) | Task orchestration, provenance and synthetic smoke |
| [09](sessions/09-reporting.md) | Paired reports, integration checks and real execution runbook |
| [10](sessions/10-evidence.md) | Review complete real results and make a bounded decision |

## What the experiment will answer

The primary comparison changes the local encoder while retaining the existing
head and 32-feature electrode/window interface. It tests whether explicit power
features improve supervised classification and fixed probes. The two secondary
comparisons test complete spatial classifiers against a freshly trained spatial
EEGNet. They cannot isolate a single component or establish Tucker usefulness.

The fixed cohort has 9 participants × 3 seeds × 5 arms = **135 neural fits**,
plus two probes for each of two local arms = **108 small CPU fits**. Training is
full-input only; matched random electrode losses are validation diagnostics.
All epoch selection and candidate screening use validation. No test scoring,
mixed-training expansion, reconstruction training or hyperparameter search.

The 2-percentage-point/6-of-9-participants screening rule is predeclared in the
contract. It identifies a promising candidate for independent confirmation,
not a significant result or guaranteed application-ready accuracy. Existing
validation splits are reused, so results remain exploratory.

## Run coding sessions

From the repository root:

```bash
# Inspect without starting models or writing execution state.
bash scripts/run_candidate_sessions.sh --list
bash scripts/run_candidate_sessions.sh --dry-run

# One fresh coding session at a time.
bash scripts/run_candidate_sessions.sh --session 01
bash scripts/run_candidate_sessions.sh --session 02

# Or process implementation sessions 01–09 sequentially.
bash scripts/run_candidate_sessions.sh

# Resume an unfinished session using a model available to your account.
bash scripts/run_candidate_sessions.sh --session 03 --model gpt-6.1-sol
```

The default follows this repository's existing workflow: `gpt-6-luna` with
`high` reasoning. Override with `--model`, `AGFL_SESSION_MODEL`, or `--effort`.
This is not a pricing or account-availability claim. Actual execution uses your
Codex account; automation tests use a fake executable and make no model calls.

Other options: --from ID, --through ID, --python PATH, --codex PATH,
--state-dir PATH, --timeout SECONDS (default 1800 per model/check phase).
--session cannot be combined with range options. Python defaults to AGFL_PYTHON,
then repository .venv/bin/python, then python3; Python 3.12+ is required.
Bash, Git, flock, GNU timeout and an authenticated Codex CLI are required.

The runner uses stdin prompts, JSON event logs, a structured final response,
workspace-write sandbox and approval never. This follows local CLI help and
[official noninteractive documentation](https://learn.chatgpt.com/docs/non-interactive-mode).
It does not bypass sandbox failures. Each session receives prior verified API
handoffs, fixed requirements, allowed paths and acceptance checks.

## Resume and failure handling

Logs and completion receipts are in `.session-runs/encoder-candidates/`.
A session advances only after its JSON handoff, required files, independent tests
and unchanged plan fingerprint pass. Completed sessions are rechecked and skipped.
Wrong IDs, blocked reports, nonempty blockers on a completed report, missing/
skipped/failed tests, timeouts and plan edits stop execution. No automatic retry.
Inspect the failed attempt and rerun its command after repair; partial edits are
preserved. Never delete scientific blockers simply to force a completion receipt.

Software blockers and real-input readiness differ: completed code with passing
synthetic tests may still report missing real data in preflight and its handoff.
A completed JSON handoff must have blockers:[]; pending real-input findings remain
in next_session_notes. This distinction does not permit claiming a real study
complete or suppressing failed scientific checks.

The new runner shares `.session-runs/accuracy.lock` with the previous workflows,
preventing simultaneous edits by different runners. Do not edit the checkout
while sessions run. All new plan Markdown/JSON and its runner/checker are
fingerprinted; changes invalidate new receipts and require reviewed recovery or
a new state directory. The existing accuracy and encoder-audit plans, runners
and receipts remain independent and unchanged.

## After implementation

Session 09 creates `docs/encoder-candidates/runbook.md` with exact commands:
preflight, genuine tiny synthetic smoke, then one real A01/seed0 pilot containing
five neural fits and four CPU probes. Inspect it before explicitly starting the
remaining 26 tasks. Freeze Python source, config and packages during the run;
changed identities require a fresh scientific output directory.

The planned package is `inm/encoder_candidates/`, config
`configs/encoder-candidates.json`, output `results/encoder-candidates-v1/`.
These are implementation targets, not existing experiment results. Original
baseline split metadata is read-only; all models are newly trained. Historical
checkpoint replay mismatches do not require editing or bypassing old manifests.
Metadata/data identity mismatches still block real execution. The earlier
encoder audit remains useful but its blocked replay is not a coding prerequisite.

Only after all real tasks and their report pass:

```bash
bash scripts/run_candidate_sessions.sh --session 10
```

The gate checks complete 27-task/five-arm coverage, current source/config,
identities, declared checks, paired mask coverage and checkpoint/history/
prediction/probe hashes. It rejects missing, partial, synthetic or stale evidence
before a model call. Numerical correctness remains the responsibility of the
implementation tests and scientific review. No further training starts in session 10.

## Verify this automation

```bash
.venv/bin/python -m unittest discover -s tests/automation -p 'test_candidate_runner.py' -v
bash -n scripts/run_candidate_sessions.sh
```

Tests use isolated temporary Git workspaces and fake Codex. They cover sequential
handoffs, resume, failures, locking, model changes, budgets/timeouts, report
contradictions, provenance changes and evidence rejection. They do not establish
that candidate models have been implemented or improve EEG classification.
