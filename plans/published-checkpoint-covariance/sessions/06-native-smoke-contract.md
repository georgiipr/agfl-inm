# Session06 concrete native smoke acceptance

Read session06-integration.md and accepted05-R1/review.json. One fresh bounded
gpt-6-luna/high worker,20minutes; supervisor owns host CUDA operations. No real
fitting or outcome scoring in coding work. Keep accepted scientific core files
(data/masks/classifier/atcnet_native/completion/training) unchanged unless an
observed blocker is reviewed with the supervisor. Narrow integration changes may
touch execution modules, launcher, new synthetic helper/tests and runbook/lock.
Preserve previous handoffs and tests; add session-06-handoff.md.

The existing `smoke --native --device cuda:0` currently does only forward/input
gradient checks. Extend it to accept an exclusive synthetic `--output` directory,
outside real results, and execute the actual native checkpoints, not the toy
classifier. Keep all9 native synthetic forward/nonzeroinputgradient/frozenstate
checks. For A01, run a genuinely short synthetic covariance fit using accepted
train_covariance(...,synthetic=True), real classifier, at least one actual update.
Generated native shape[N,22,1125], T-only generated IDs, disjoint train/validation.
No real MAT/GDF data loading in this smoke. Use memory growth before GPU init,
verify actual GPU tensor placement, finite nonzero covariance/input gradients,
and unchanged full classifier parameter/BatchNorm tensor bytes. No CPU fallback.

Save/reload selected free state and recompute predictions in this public launcher
path. Epoch0 selection is legitimate; demonstrate updates through real history,
not by requiring selected_free differs when epoch0 wins. Produce:

- native-smoke.json: synthetic_fixture:true, device:"cuda:0", subjects:A01..A09,
  frozen_before_sha256, frozen_after_sha256, actual training history (atleast2rows).
- native-smoke.npz: values[N,22,1125], mask bool[N,22], initial_free,selected_free,
  probabilities,replay_probabilities,covariance_gradient,input_gradient.

The supervisor's acceptance_cuda.py invokes the public Bash launcher, inspects
these artifacts, loads the actual frozen A01 model, replays probabilities onGPU
and independently computes conditional completion with NumPy. Inspect that check
before coding. Never edit supervisor acceptance. Smoke outputs are generated
software evidence and must never enter real reports.

Finalize runbook exact commands: preflight, native synthetic smoke, T-only native
references BEFORE realpilot, pilot/audit, correctness-onlypilotreview, remaining24
fits/seal, boundedEtask evaluation/audit, finalaggregate and independent numeric
verify_real_report.py. Fix the incomplete synthetic runbook command sequence.
Freeze actual installed dependencies from .venv-published-covariance; GPU extras
are alreadyinstalled. Record device flags/runtime and space/resources estimates
honestly. Supervisor cleared only task-created duplicatepipdownloads by approval;
~3.1GiB diskfree, realoutputs expected~1.5GiB. Check actualsizes before realwork.

Optional narrow cleanup: remove unreachable old monolithic audit code after its
fail-closed raise. Keep pipeline math/settings/selection fixed. Include covariance
free parameter count253 in report in addition to backbone count if absent.

Run full package tests, common acceptance.py, stdlibplan, Bashsyntax; preserve
actual failures/limits. Supervisor runs physical native GPU check independently
after sourcefreeze. Readiness is not a measured realstudy result. No real fits,
E scores, architecture changes, prioroutputmutations, commits or pushes.
