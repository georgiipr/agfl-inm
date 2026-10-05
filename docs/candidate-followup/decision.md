# Cohort decision

**Verdict:** retain `local_power` as the sole candidate for a separately designed,
independent confirmation. Do not advance the current `spatial_filterbank`
configuration. Treat the spatial Transformer as inconclusive and do not add it
to the next experiment in this round.

The complete cohort passes the evidence-integrity review. The participant-equal
primary contrast (`local_power` minus `local_control`) gains 6.15 percentage
points of clean validation balanced accuracy, has 7/9 positive participants,
and an exploratory paired 95% interval of [2.04, 10.13] pp. It also improves
mean validation log-loss by 0.100. Thus it meets the frozen +2 pp / 6-of-9
screen. The gains are modestly sized, validation was reused for epoch selection,
and its advantage disappears under the four missing-electrode conditions; the
screen is a reason to confirm, not a claim of established benefit.

The spatial filter bank loses 13.77 pp against its paired EEGNet reference
(1/9 positive; interval [−22.20, −5.60] pp) and has worse full-input log-loss.
The Transformer meets the numerical BA screen (+5.37 pp, 7/9) but its interval
crosses zero, log-loss uncertainty crosses zero, and its missing-electrode
log-loss is consistently worse than EEGNet. Given the one-follow-up limit, the
primary local result is the more direct and better-supported confirmation target.

The follow-up should use a new, independently collected or independently
partitioned cohort and a frozen protocol comparing `local_power` with the same
local control, with no architecture tuning against this validation set. Revisit
the Transformer only in a later, separately justified study. This recommendation
does not authorize any fitting or change to the completed study artifacts.

The report's participant bootstrap is exploratory (2,000 resamples, seed
20261003), the three contrasts create multiplicity, and epoch selection reused
validation. Results are equally weighted across nine participants after seed
averaging; they do not establish performance in other people or strong
classification. No test score was consulted for these verdicts.
