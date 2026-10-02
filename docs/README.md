# Documentation map

Start with [the collaborator onboarding guide](onboarding.md). For a shareable
explanation with equations and diagrams, read [the concept PDF](concept/agfl_concept.pdf)
or edit [its LaTeX source](concept/agfl_concept.tex).

| Document | Use |
|---|---|
| [Onboarding](onboarding.md) | Reading order, architecture, workflow, and first contributions |
| [Investigation](investigation.md) | Dated evidence, local readiness, and interpretation limits |
| [Baseline runbook](baselines/runbook.md) | Prepare, smoke-test, pilot, resume, and report the separate baseline study |
| [Baseline integration review](baselines/review.md) | Session-10 boundaries, software checks, and empirical gates |
| [First complete baseline results](baselines/results-review.md) | Cohort measurements, reporting/initialization fixes, and corrected replication commands |
| [Corrected baseline results](baselines/corrected-results-review.md) | Completed reproducible cohort and the optional session-11 comparison |
| [Temporal comparison results](baselines/temporal-results-review.md) | Completed paired head comparison and the legacy pilot prerequisite for session 12 |
| [Legacy pilot review](baselines/legacy-pilot-review.md) | Validated encoder/head diagnostics and readiness for session 12 |
| [Completed tensor comparison](baselines/tensor-results-review.md) | Verified 162-fit cohort, validation decision, and next accuracy investigation |
| [Accuracy implementation sessions](../plans/accuracy/README.md) | Small sequential tasks and a resumable Codex Bash runner |
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
