# Independent supervisor acceptance

The supervisor owns executable acceptance checks under this plan directory.
Create them before their corresponding worker session using the requirements
below; workers cannot modify them. This document contains specifications, not
a claim that tests or a runner already exist. No fake measurement fixtures.

Use the runtime sealed after the checkpoint audit. Save its exact executable
path in receipts. The mandatory common commands, once implemented, are:

```text
<runtime-python> -m unittest discover -s tests/published_covariance -v
<runtime-python> plans/published-checkpoint-covariance/acceptance.py
python3 -I -S plans/published-checkpoint-covariance/check_plan.py
bash -n scripts/run_published_covariance.sh
```

The stdlib check launches the dependency-light plan without importing scientific
packages and expects 27 covariance fits, zero backbone fits, 81 arm evaluations.
Run relevant subsets per session and the complete suite at integration. Empty
discovery/skipped mandatory checks do not pass. Record exits and inspect failures.

## Session gates

1. **Checkpoint audit:** hash/inspect all nine real assets; verify pinned source,
   model tensor/config compatibility, native recipe and provenance grade. Confirm
   no performance-based run selection. Unknown checkpoint provenance is explicit;
   an unresolved input recipe cannot receive a compatibility pass.
2. **Data:** independent known arrays expose wrong units/channel permutations,
   off-by-one cue indexing and labels shifted by excluded trials. Verify official
   input origin, native T adapter equality, deterministic80/20 membership, no
   new E-fitted transform and whole-pipeline hidden-channel invariance. Distinguish
   recording MAT files from label-only MAT files. Runtime sample metadata must
   account for every included/excluded original trial.
3. **Classifier:** native reference and wrapped inference agree on fixed synthetic
   and T-only inputs. Compare probabilities/logits, not only labels. Parameter
   and BN buffers remain unchanged through train-mode callers. Verify nonzero
   finite input gradients. If porting frameworks, compare input-gradient vectors
   or directional derivatives with native autodiff/finite differences.
4. **Covariance:** analytical conditional-completion example and independent
   NumPy solve; fixed/learned identical at epoch0; covariance gradients agree
   with finite differences; full bypass is bitwise; observed preservation,
   hiddenNaN/Inf invariance and all-missing rejection. A synthetic task demonstrates
   covariance-only updates with unchanged backbone/BN state. Test epoch0 and
   tie handling, validation immutability and exact selected-state restoration.
5. **Execution:** meaningful fault injection for missing/altered weights/config/
   input/checkpoints, partial task, corrupt probabilities, foreign output,
   unsafe paths, occupied lock, timeout child cleanup and incomplete fit seal.
   Successful completed-task resume and repeated summary must preserve bytes.
   Independent unequal-size synthetic participants test hierarchical aggregation
   and learned-minus-fixed paired bootstrap, including negative effects.
6. **Integration:** full suite and actual native-checkpoint CUDA forward/backward
   on synthetic input; short synthetic covariance fit/save/replay through the
   public launcher. All historical protected hashes unchanged. No real E scoring.

## Parity rules fixed before tests

For the same deterministic native runtime/device, require exact saved-probability
replay and full-input equality across arms. Native-to-port or adapter numerical
comparisons use abs1e-5/rel1e-4 probability tolerance and identical predicted
labels on the prescribed inputs; input directional derivatives use abs1e-5/rel1e-3
away from nondifferentiable points. Inspect per-tensor maximum errors.

These are proposed admission thresholds, not measured precision claims. If a
backend cannot meet them, preserve the failure and diagnose the cause. Prefer
native execution or a separately reviewed repair; do not relax tolerances after
seeing experiment outcomes. Native data array comparisons need units-aware
absolute tolerances declared in the adapter audit before checking real arrays.

## Real acceptance

Operational pilot: all three A01 T fits and independent replays pass, frozen
backbone and source identity preserved, runtime/memory reported. Accuracy is
not a gate. No E scoring until all27 T-fit states are sealed.

Completion: 27 covariance fits, nine native full-input references, 81 paired
arm evaluations, every task independently audited, exact full-input equality,
all native trial/exclusion counts explained, and complete aggregate recomputed
independently from numeric predictions. Prior exposure and external checkpoint
selection grade must appear in machine-readable and human-readable reports.

Do not spend GPU time replaying unrelated completed cohorts. Test regression
only for reused interfaces at risk; old protected-source hashes must still match.
