# CUDA execution v2

The user authorized real CUDA experiments outside the sandbox on 2026-10-05.
The current declaration is `configs/task-driven-completion-cuda-v2.json`, with
output `results/task-driven-completion-cuda-v2`. The frozen study identity is
`74bfeb36445f3b0d365fac8ca03514e9e8e359c2c205b277e0389bf2922eb071`.

The first v1 pilot stopped before calibration or completion fitting because
fused Transformer inference on CUDA failed the CPU/GPU full-input gate. Its
source snapshot, configuration, output and failure logs remain preserved.
Bounded repair 01 disables the MHA fastpath, scopes/restores that backend flag,
and records/enforces it in execution metadata. Science and the original
rtol=1e-4/atol=1e-6 tolerance are unchanged. See `repair-01-handoff.md`.

Independent acceptance passed six GPU test groups (no skips), three protocol
tests, and full-input replay for all 27 historical tasks. The largest CPU/GPU
probability difference was 1.0728836059570312e-6; every comparison passed the
combined absolute/relative tolerance. All 7,806 protected files were unchanged.
`readiness-v2.json` and `.session-runs/task-driven-completion-cuda/repair-01/`
contain receipts and exact logs. Prior CPU regression acceptance passed 60 tests.

Real tasks run sequentially on cuda:0, NVIDIA GeForce RTX 3060 Laptop GPU.
Training-only factor calibration and original historical replay remain CPU.
There are 27 participant/seed tasks, 54 supervised completion fits, five strategies
per task and 105 saved validation prediction cells per task. This is a validation
screen; selection and evaluation reuse validation and do not constitute independent
confirmation. Negative effects must be preserved.

Supervisor runner: `plans/task-driven-completion-cuda/run_experiments_v2.py`.
State/logs: `.session-runs/task-driven-completion-cuda/experiments-v2/`.
`status.json` gives current state; `events.jsonl` records starts, heartbeats and
exits; `resources.jsonl` records process memory and GPU utilization/memory;
`task-XX.review.json` records independent audit acceptance. Fits have a 7,200-second
deadline, audits 1,200 seconds, with process-group termination on timeout. Any
failure stops advancement and preserves artifacts; there are no automatic retries.

The pilot is task 0 (A01/seed 0). After it passes independent replay/audit, a
recorded operational pilot review permits tasks 1–26; accuracy is not an advancement
gate. The cohort's detached watchdog can survive the conversational turn and
generates the complete report only after all tasks pass. It is a local process,
not a reboot-persistent service. Do not relaunch into existing attempt logs.

```bash
.venv/bin/python plans/task-driven-completion-cuda/run_experiments_v2.py pilot
.venv/bin/python plans/task-driven-completion-cuda/run_experiments_v2.py cohort --detach
```

## Launch receipt

The v2 pilot completed on 2026-10-05: fitting took 193.09 seconds and independent
audit 14.52 seconds. All 105 prediction cells replayed exactly; histories were
finite, masks/orders/IDs paired, and the backbone unchanged. Sampled GPU memory
peaked at 525 MiB. See [pilot review](pilot-review-v2.json). These real pilot
measurements are incomplete scientific evidence and do not establish cohort gains.

The remaining 26 tasks were launched with the detached watchdog at 09:46 UTC,
PID 33647. Runtime state is in
[status.json](../../.session-runs/task-driven-completion-cuda/experiments-v2/status.json);
the [cohort log](../../.session-runs/task-driven-completion-cuda/experiments-v2/cohort-watchdog.log)
records progress and terminal status. This is a launch receipt, not a claim that
the cohort has completed. No coding worker remains active.

Workflow references: [repair plan](../../plans/task-driven-completion-cuda/repair-01.md),
[worker handoff](repair-01-handoff.md), and
[sequential-session-watchdog skill](/home/kalexu97/.codex/skills/sequential-session-watchdog/SKILL.md).
