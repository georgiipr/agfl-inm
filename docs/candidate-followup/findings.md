# Completed cohort findings

## Evidence reviewed

The frozen reproducible-v1 cohort is complete: 27 real tasks (A01–A09, seeds
0–2), 135 neural fits, and 108 local probes. The evidence report identifies the
same study/config/source/package identity as the task records. I independently
checked every task digest listed in `report/evidence.json`, each task's five fit
record digests, and all 675 fit-artifact digest entries (checkpoints, histories,
predictions and local probe artifacts). All identities, statuses, paired-arm
records, CPU execution-device fields and task-level Boolean checks passed. No
row was missing, synthetic, failed or silently omitted. The issue table is
empty.

I recalculated clean `full_22` means from fit records, averaging three seeds
within each participant and then the nine participants equally. Paired contrasts
and participant bootstrap intervals were recomputed from those same participant
means (2,000 resamples, seed 20261003); values agree with the saved report.
These are validation results. Validation also selected each fit's epoch, so the
intervals are exploratory and validation is reused; they are not independent
confirmatory intervals. Three contrasts were examined. No test metrics were
read or used here.

## Verdict summary

| Candidate and paired reference | Clean validation BA | Paired BA gain (exploratory 95% interval) | Positive participants | Clean validation log-loss | Parameters | Mean BA gap (train − validation) | Mean log-loss gap (validation − train) |
|---|---:|---:|---:|---:|---:|---:|
| `local_power` vs `local_control` | 49.95% vs 43.80% | +6.15 pp [+2.04, +10.13] pp | 7/9 | 1.227 vs 1.327; loss improvement 0.100 [0.045, 0.163] | 13,860 vs 14,180 | 9.18 pp | +0.208 |
| `spatial_filterbank` vs `spatial_eegnet` | 45.73% vs 59.50% | −13.77 pp [−22.20, −5.60] pp | 1/9 | 1.303 vs 1.057; loss improvement −0.245 [−0.416, −0.082] | 1,268 vs 3,796 | 1.52 pp | +0.059 |
| `spatial_transformer` vs `spatial_eegnet` | 64.87% vs 59.50% | +5.37 pp [−0.42, +12.00] pp | 7/9 | 0.993 vs 1.057; loss improvement 0.065 [−0.099, +0.257] | 11,092 vs 3,796 | 11.51 pp | +0.436 |

“Loss improvement” is reference log-loss minus candidate log-loss, so positive
values favor the candidate. Participant counts use positive BA differences.
The declared screen is a cohort mean BA gain of at least 2 pp plus at least 6/9
positive participants. It is a practical screen for independent confirmation,
not a significance or strong-classification claim.

## Candidate assessments

### Local power — passes the screen; retain for independent confirmation

The mean paired BA gain is +6.15 pp, with 7/9 positive participants, satisfying
the screen. Its exploratory interval is wholly positive but its lower bound is
only just above the 2 pp screening margin. Clean log-loss also improves on
average by 0.100, with 8/9 participants improving and an exploratory interval
of [0.045, 0.163]. Parameter count is slightly lower than the local control
(13,860 vs 14,180). The mean clean BA gap is 9.18 pp (clean train BA 59.13%,
validation BA 49.95%); the validation-minus-train log-loss difference is
+0.208, showing worse held-out calibration/loss despite the BA gain.

The four missing-electrode validation conditions do not preserve the full-input
advantage over local control. Relative BA differences are −1.00 pp (static 16),
−1.49 pp (dynamic 16), −1.33 pp (static 6), and −1.09 pp (dynamic 6); log-loss
is also worse by 0.058, 0.047, 0.059 and 0.029 respectively. This candidate's
benefit is therefore specific to the clean full-input comparison in this study.

The ordered local logistic probe averages 45.43% BA versus 26.85% for its
single shuffled-label control, a 18.57 pp gap. Local control is 42.12% versus
24.13%, a 17.99 pp gap. All probe fits converged. This is evidence that frozen
ordered features support a linear probe under this split, not a p-value, proof
of causal information, or a substitute for the neural comparison. The single
shuffle is only a diagnostic control.

### Spatial filter bank — does not pass; do not advance this configuration

It is 13.77 pp below the paired spatial EEGNet reference on mean BA; only 1/9
participants is positive and the entire exploratory interval is negative.
Log-loss is worse by 0.245 on average; only 2/9 improve. It uses 1,268 parameters
versus 3,796. Its small 1.52 pp train–validation BA gap accompanies low clean
train BA (47.25%) and validation BA (45.73%), consistent with weak fitting
capacity/optimization under this fixed setup rather than evidence of a useful
compact model. Its validation log-loss exceeds train log-loss by 0.059.

At 16 observed electrodes it has lower log-loss than EEGNet by 0.227 (static)
and 0.192 (dynamic), but BA is lower by 6.97 and 7.46 pp. At six electrodes,
BA is lower by 6.63–6.67 pp and log-loss is worse by 0.727–0.804. The mixed
loss diagnostics do not reverse the full-input result or provide a robustness
case.

### Spatial Transformer — passes the numerical screen, but remains inconclusive

Its mean BA gain is +5.37 pp with 7/9 positive participants, satisfying the
predeclared screen. However, its exploratory interval [−0.42, +12.00] pp
includes zero. Mean log-loss improves by 0.065, but its interval also crosses
zero ([-0.099, +0.257]); 7/9 participant means improve. This is suggestive,
not settled evidence. It uses 11,092 parameters, about 2.9 times the EEGNet
reference. Its mean clean BA gap is 11.51 pp (train BA 76.38%, validation BA
64.87%); validation log-loss exceeds train log-loss by 0.436.

Missing-electrode BA is essentially tied with the reference at 16 electrodes
(−0.02 pp static, +0.22 pp dynamic) and trails by 0.62 pp in both six-electrode
conditions. Its log-loss is worse than EEGNet in all four conditions, by
0.567–0.757. The candidate's possible full-input advantage does not extend to
these robustness diagnostics.

## History and secondary diagnostics

Selected checkpoints were read from all fit histories. The selected epochs are
the ones chosen by the frozen full-input validation BA / log-loss / earlier-epoch
rule. Clean train metrics above are separate eval-mode scores computed at those
checkpoints; they are not optimization-time minibatch metrics. At the selected
epoch, the mean optimization-time minibatch accuracy/loss was 57.43%/1.039 for
local power, 35.98%/1.357 for spatial filter bank and 68.08%/0.722 for the
Transformer. These training-mode values are descriptive optimization
summaries, not substitutes for clean train or validation scores.

The saved robustness table averages repeats, then seeds within participant,
then participants equally. It retains all five arms and each of the four
predeclared electrode-loss scenarios. As above, fewer electrodes substantially
reduce BA for every arm. The Transformer keeps BA near its EEGNet reference but
has much worse log-loss, while spatial filter bank's relative log-loss advantage
at 16 electrodes is paired with materially lower BA and disappears at six.

The ordered-versus-shuffled probe gap is similar for the two local encoders;
one shuffled-label fit per task cannot be interpreted as a null distribution or
a significance test. Bootstrap intervals are likewise exploratory because of
validation reuse and multiple contrasts. No candidate establishes strong EEG
classification, generalization beyond these nine participants, or robust
missing-electrode operation.
