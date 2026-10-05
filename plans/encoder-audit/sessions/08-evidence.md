# Session 08 Interpret the real audit and bound the next experiment

This session requires a complete real 27-task audit at
`results/encoder-audit-v1/report/evidence.json`; the runner validates it before
calling a model. No synthetic evidence or selected favorable participant may
substitute. Real fitting is not part of this session.

Edit only `docs/encoder-audit/findings.md`,
`docs/encoder-audit/next-experiment.md`, and
`tests/encoder_audit/test_findings.py`. You may read verified train/validation
audit artifacts and their source identities. Do not modify scientific code or
any audit outputs after their hashes were recorded.

1. Report alignment/replay findings first. Then contrast clean training and
   validation, per-class errors, participant variation, fixed probe views, and
   completion's paired classification effects. Cite exact evidence artifacts and
   coverage. Describe unrecoverable legacy heads and source limits explicitly.
2. Distinguish optimization weakness, generalization gap, feature bottleneck,
   compression loss, and unresolved alternatives. Probe/NRMSE association does
   not establish causality. Do not compare unmatched pipelines as a tensor effect.
3. Choose at most one future controlled question justified by validation-only
   evidence, or explicitly choose no intervention. Examples: one training-budget
   intervention after optimization evidence, or a matched local-versus-spatial
   encoder ablation after representation evidence. Do not bundle both, expand
   ranks, or start training. Existing legacy behavior remains unchanged.
4. The future experiment document declares hypothesis, invariant controls,
   changed factor, full/robust selection policy chosen in advance, participants,
   seeds, budget, failure rules, new output identity, and a stop decision for no
   gain. Acknowledge reused-validation selection bias and require fresh evaluation
   design before a confirmatory performance claim. Test results cannot choose it.

Acceptance: findings tests use synthetic report fixtures to exercise completeness,
identity references, and explicit unavailable/no-intervention outcomes. They must
not encode a required positive result or assert fixed real performance numbers.
Run all new-package tests. A scientifically inconclusive completed audit can
complete this session; missing or invalid evidence cannot.
