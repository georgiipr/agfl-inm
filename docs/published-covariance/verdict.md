# Published-checkpoint covariance: completed exploratory result

Completed on 2026-10-07. The accepted output is
[`results/published-covariance-v1-cuda-root`](../../results/published-covariance-v1-cuda-root/).
It contains all **27 covariance fits, 81 arm evaluations, 891 prediction cells,
and nine native E references**, covering nine participants and three covariance
seeds. All task audits, the final cohort audit, independent numeric recomputation,
and repeated-summary byte checks passed. No backbone was trained.

Learning the channel covariance improved degraded-input balanced accuracy over
fixed covariance by **9.55 percentage points**, averaged equally over retained-16
and retained-6 conditions and participants. The paired participant bootstrap 95%
interval is **6.32 to 12.55 points** (2,000 draws, PCG64 seed 20261006). The
corresponding learned-minus-zero-fill difference is **31.82 points**.

| Retained electrodes | Zero fill BA | Fixed covariance BA | Learned covariance BA | Learned − fixed |
|---|---:|---:|---:|---:|
| 22, full input | 81.13% | 81.13% | 81.13% | 0.00 pp |
| 16, static loss | 43.70% | 75.46% | 77.55% | +2.09 pp |
| 6, static loss | 33.01% | 45.79% | 62.80% | +17.00 pp |

The added benefit is largest with six retained electrodes. Every participant's
average across the two degraded conditions was positive, ranging from +1.52 to
+16.06 points. This does not mean every condition improved: at 16 retained
electrodes, A02 changed by −0.28 points and A09 by −2.50 points. Those negative
results are preserved. Mask repeats are averaged first, then covariance seeds
within participant, then participants equally.

Full-input predictions are exactly equal across arms and covariance seeds. The
maximum native-versus-wrapped full-input probability difference was also zero
for all nine T references and all nine E references. All 288 trials per session
were retained, with no exclusions. All 27 before/after frozen-classifier hashes
match. The learned covariance has 253 free parameters before the frozen
115,172-parameter classifier. Epoch zero was eligible and selected in 0/27 fits;
1/27 fits reached the declared 100-epoch ceiling.

## Scientific interpretation

This supports transfer of learned covariance completion to the admitted frozen
EEG-ATCNet checkpoint family for whole-trial static electrode loss, particularly
severe loss. It does not establish generalization to another model family,
participant population, dynamic loss pattern, or unseen acquisition protocol.

The result is **exploratory and descriptive**. The external checkpoint set has
[grade-B provenance](checkpoint-audit.md): E-guided checkpoint selection cannot
be conclusively excluded, and historical project exposure to E outcomes is
unknown. The published backbone and author normalization already used all T
trials, including the later covariance-validation subset. Covariance seeds vary
splits, training order and masks; they are not independent backbone-training
seeds. E was excluded from covariance fitting and selection in this completed
run, but these provenance limits preclude a confirmatory independent-test or
SOTA claim. The bootstrap interval does not remove those limits.

## Preserved failure and runtime repair

The initial output, `results/published-covariance-v1`, remains unchanged and
incomplete: 27 fitted states and 13 audited E tasks were preserved. A05 seed 1
aborted with exit 134 before writing its task cells when TensorFlow constructed
an invalid path to CUDA's `libdevice.10.bc`. No complete result was reported from
that attempt.

The [bounded diagnosis](runtime-failure-handoff.md) led to explicit discovery of
the already installed CUDA compiler data and `ptxas` executable. Two fresh
compiled-math/gradient probes and the all-nine synthetic native CUDA fit/replay
check passed before restarting. The original compiler lookup warnings disappeared.
Packages, scientific source, masks, optimizer and selection rules were unchanged.
Because `XLA_FLAGS` changed the recorded runtime identity, all 27 covariance fits
were repeated in the fresh output; no old fit state or E cell was imported.
The partial E exposure from the first attempt remains part of the history. No
E outcome was inspected to choose this operational repair. Total computation
across both attempts therefore included 54 covariance fits; the reported study
contains the fresh 27 only.

The fresh pilot took 12 minutes 12 seconds, the remaining-fit launcher 76 minutes
51 seconds, and evaluation with task audits 18 minutes 20 seconds. Sampled GPU
memory peaked at 2,253 MiB during pilot fitting and 4,296 MiB across the full
restart, including native reference subprocesses (five-second samples, not an
exact instantaneous peak). The final output occupies approximately 1.12 GiB.

## Reviewable evidence and reproduction

- [Complete human report](../../results/published-covariance-v1-cuda-root/report.md)
  and [machine report](../../results/published-covariance-v1-cuda-root/report.json).
- [Final cohort audit](../../results/published-covariance-v1-cuda-root/audit.json)
  and [27-state seal](../../results/published-covariance-v1-cuda-root/fit-seal.json).
- [Independent numeric verification](../../.session-runs/published-checkpoint-covariance/runtime-repair/final-independent-report.stdout)
  and [command receipt](../../.session-runs/published-checkpoint-covariance/runtime-repair/final-independent-report.json).
- [Independent pre-E fit review](../../.session-runs/published-checkpoint-covariance/runtime-repair/restart-cohort-independent.stdout).
- [Explicit-runtime reproduction commands](runtime-recovery.md),
  [general runbook](runbook.md), and [dependency lock](dependencies-lock.txt).
- [Plan](../../plans/published-checkpoint-covariance/README.md),
  [integration handoff](session-06-handoff.md), and
  [sequential-session-watchdog skill](/home/kalexu97/.codex/skills/sequential-session-watchdog/SKILL.md).

Accepted identity SHA256:
`6153df7e7d434f6e6bc6a2403f7671f835936ff60da60f420af6106da29299d7`.
Machine-report SHA256:
`f3db7139915dda71a036dfcc44de404e3a2d109a207ad60438feaf2ab38bd766`.
