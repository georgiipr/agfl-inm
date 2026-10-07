# Learned covariance before a published frozen checkpoint

Supervisor handoff prepared 2026-10-06. Repository:
`/home/kalexu97/Projects/AGFL`; branch `research/baseline-accuracy`.
This plans a new experiment; creating this plan launches no workers or training.

## Start a new supervisor session

Send this message when ready to execute:

> Follow `plans/published-checkpoint-covariance/README.md` and its contract.
> Use the sequential-session-watchdog skill, one bounded worker at a time,
> with independent supervisor acceptance between sessions. I authorize auditing
> and downloading official published checkpoints/source/data as needed,
> implementing the isolated adapter and experiment, and creating an isolated
> runtime if needed. After acceptance, run the real operational pilot and the
> complete nine-participant, three-covariance-seed study on CUDA outside the
> sandbox, training only covariance. Keep the classifier frozen. Finish with
> independently audited results and a scientific verdict. Preserve all earlier
> work. Do not run alongside the other session in the same checkout. Prior E
> exposure is unknown and the study is exploratory; do not ask me to assert
> that E was never seen. If no compatible published checkpoint set can be
> established, report the concrete blocker without substituting a newly trained
> classifier or silently changing the experiment.

Sending that message authorizes the stated implementation and real execution.
Tool-level escalation remains separate. No further permission is needed for
routine implementation decisions, acceptance, or already-authorized real runs.
Do not send messages to model authors or repository maintainers without a
separate explicit user instruction.

## Objective and stopping point

Test whether learned channel covariance improves **whole-trial electrode-loss
robustness of a published pretrained motor-imagery classifier**, compared with
fixed training-estimated covariance and zero fill. Preserve the classifier's
full-input predictions. The target is transfer of the completion technique to
a strong existing model, not a new architecture or an automatic SOTA claim.

One checkpoint family, all nine BCI IV-2a participants, three covariance-training
seeds, three arms: **27 covariance fits and 81 arm evaluations**, plus nine
native-reference checks. Published backbone weights are reused, never trained.
Static loss retains 16 or 6 electrodes for the whole trial. Dynamic loss and
additional architectures are deferred; the reasons are in the contract.

## Required reading

- Root `AGENTS.md`, `docs/README.md`, `docs/onboarding.md`, `docs/investigation.md`.
- [This study's contract](CONTRACT.md) and [acceptance requirements](ACCEPTANCE.md).
- [Current matched covariance evidence](../../docs/covariance-confirmation/matched-review.md).
- [Current covariance mathematics](../../docs/covariance-expansion/model-and-protocol.md).
- [Exploratory E handoff](../../docs/covariance-confirmation/NEXT-EXPLORATORY-E.md).
- [Reusable sequential-session-watchdog skill](/home/kalexu97/.codex/skills/sequential-session-watchdog/SKILL.md).

The clean architecture cohort, joint covariance cohort and matched frozen-backbone
comparison are complete. Do not restart them. The other session may still be
implementing or running exploratory E evaluation: inspect its latest receipts,
host processes and `.session-runs/accuracy.lock`. A free experiment lock alone
does not prove that another coding worker has stopped. Do not modify that work,
stop its processes, or run concurrent workers in this checkout. If ownership is
unclear, finish read-only investigation and report the exact coordination need.

## Ordered sessions and execution gates

| Session | Deliverable | Gate before advancing |
|---|---|---|
| [01: checkpoint audit](sessions/01-checkpoints.md) | Pinned nine-subject checkpoint set, provenance, native recipe and framework decision | Supervisor accepts compatibility and interpretation before outcomes |
| [02: data adapter](sessions/02-data.md) | Native input pipeline, T splits, trial/label alignment and static masks | Independent alignment and hidden-value checks |
| [03: classifier adapter](sessions/03-classifier.md) | Exact published classifier loading and differentiable frozen inference | Native-reference forward and gradient parity |
| [04: covariance](sessions/04-covariance.md) | Fixed/learned completion, bounded optimizer and checkpoint selection | Numeric solve/gradient/frozen-state checks |
| [05: execution](sessions/05-execution.md) | Immutable persistence, audits, reporting and bounded launcher | Failure/resume/aggregation tests |
| [06: integration](sessions/06-integration.md) | Reviewed runbook, isolated runtime and synthetic CUDA readiness | Supervisor acceptance, protected hashes and sealed scientific declaration |

Default coding deadline: 20 minutes per fresh worker. Record worker model, start,
deadline, plan/source hashes and protected inventory. Inherit the model unless
the user chooses another; do not silently change models on failure. Check progress
at most 60 seconds apart and send meaningful updates. No parallel workers.

Before session 01, supervisor turns [ACCEPTANCE.md](ACCEPTANCE.md) into independent
checks under this plan directory; implementation workers cannot edit them. After
each session, inspect actual changes, execute the relevant checks independently,
verify protected hashes and save a proceed/stop receipt. Preserve failures and
use separately documented bounded repairs; never mark an incomplete session done.

Coding workers do no real supervised fitting or E scoring. Supervisor may perform
the bounded T-only reference checks specified in sessions 02/03. Real CUDA smoke,
pilot and cohort are separate supervisor operations after accepted implementation.

## Real sequence after readiness

1. Recheck workspace ownership, shared lock, inputs, runtime and frozen source.
2. Seal checkpoint/preprocessing/disclosure/mask/training/analysis identities.
   Finish native T reference checks. No accuracy-based checkpoint selection.
3. Fit and audit A01 covariance seeds 0/1/2 using T only. Accept the pilot on
   correctness and resources, never on accuracy. Do not score E in the pilot.
4. Fit and audit the remaining 24 covariance tasks sequentially. Seal all 27
   selected covariance states before the first E prediction in this study.
5. Evaluate all three arms on E for all 27 tasks with independent task audits.
   Compare native and wrapped full-input predictions for all nine subjects.
6. Independently recompute the complete report and write the scientific verdict.

The new launcher's verified runbook must provide exact commands; these commands
do not exist yet. It must reuse `.session-runs/accuracy.lock`, apply per-child
timeouts and preserve commands/output/exits. The supervisor remains active,
checks progress at most 60 seconds apart, and does not call a detached process
an actively supervised watchdog. No unnecessary replay of old real cohorts.

Finish with links to the audit, runbook, numeric report and verdict, actual counts,
remaining blockers and whether any process/watcher is still running. Stop after
one complete study. Backbone retraining, a second model family, dynamic losses
or an expanded comparator suite require a new declared study.
