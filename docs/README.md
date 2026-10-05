# Documentation map

Start with the [current research pipeline PDF](concept/current_pipeline.pdf)
or [its LaTeX source](concept/current_pipeline.tex) for the dataset, model,
training loss, and evaluation on the research branch. The
[collaborator onboarding guide](onboarding.md) and
[earlier concept PDF](concept/agfl_concept.pdf) explain the legacy frozen-feature
experiment; they should not be read as the current baseline architecture.

| Document | Use |
|---|---|
| [Reproduce learned covariance](task-driven-completion-cuda/reproduce.md) | Verify the shared numerical evidence without GPU/data, or run a fresh CUDA pilot/cohort with Bash launchers |
| [Learned covariance completion](concept/learned_covariance.pdf) · [LaTeX source](concept/learned_covariance.tex) | Self-contained mathematics, completion and training diagrams, a worked example, and verified validation results for learned channel covariance before a frozen EEGNet–Transformer |
| [Supervised spectral Tucker](concept/supervised_tucker.pdf) · [LaTeX source](concept/supervised_tucker.tex) | Matched frozen versus classification-trained factors, differentiable observed-entry ridge inference, and three-seed evaluation |
| [Tensor temporal encoder screen](concept/tensor_temporal.pdf) · [LaTeX source](concept/tensor_temporal.tex) | Three EEGNet-free tensor architectures, matched controls, collaborator-branch review, and the nine-participant exploratory screen |
| [Spatial EEGNet and Transformer guide](concept/spatial_models.pdf) · [LaTeX source](concept/spatial_models.tex) | Exact pipelines, training and validation results, published architecture comparisons, and limits on novelty claims |
| [Onboarding](onboarding.md) | Reading order, architecture, workflow, and first contributions |
| [Investigation](investigation.md) | Dated evidence, local readiness, and interpretation limits |
| [Baseline runbook](baselines/runbook.md) | Prepare, smoke-test, pilot, resume, and report the separate baseline study |
| [Baseline integration review](baselines/review.md) | Session-10 boundaries, software checks, and empirical gates |
| [First complete baseline results](baselines/results-review.md) | Cohort measurements, reporting/initialization fixes, and corrected replication commands |
| [Corrected baseline results](baselines/corrected-results-review.md) | Completed reproducible cohort and the optional session-11 comparison |
| [Temporal comparison results](baselines/temporal-results-review.md) | Completed paired head comparison and the legacy pilot prerequisite for session 12 |
| [Legacy pilot review](baselines/legacy-pilot-review.md) | Validated encoder/head diagnostics and readiness for session 12 |
| [Completed tensor comparison](baselines/tensor-results-review.md) | Verified 162-fit cohort, validation decision, and next accuracy investigation |
| [Research branch and collaborator review](baselines/branch-review.md) | Branch ownership, adopted Git fix, and useful ideas from George's model redesign |
| [Accuracy implementation sessions](../plans/accuracy/README.md) | Small sequential tasks and a resumable Codex Bash runner |
| [Encoder investigation sessions](../plans/encoder-audit/README.md) | Bounded audits, frozen-feature probes, reconstruction diagnostics, and sequential execution |
| [Encoder architecture sessions](../plans/encoder-candidates/README.md) | Five fixed candidate/control arms, bounded coding tasks and sequential runner |
| [Candidate execution follow-up](../plans/candidate-followup/README.md) | RNG repair, execution readiness, explicit real pilot/cohort and verdict sessions |
| [Project overview](project_overview.md) | Research question and proposal mapping |
| [Model adaptation](model_adaptation.md) | Encoder locality, representation routes, and pooling |
| [Tensor mathematics](tensor_math.md) | Factor fitting and exact masked ridge solves |
| [Availability protocol](availability_protocol.md) | Channel identities and static/dynamic masks |
| [Experiment protocol](experiment.md) | Study matrix, selection, reporting, and artifacts |
| [Concept build instructions](concept/README.md) | Rebuilding and reviewing the PDF |
| [Repository README](../README.md) | Environment and official dataset acquisition |
| [Assistant orientation](../AGENTS.md) | Working context and research invariants |

The existing technical documents describe the implemented experiment. The
[original proposal](../references/Proposal_Tensor_Attention_Variable_Signal_Availability.pdf)
describes a broader design. Their differences are deliberate and must accompany
any presentation of this checkout. No empirical improvement is asserted by the
new orientation documents.
