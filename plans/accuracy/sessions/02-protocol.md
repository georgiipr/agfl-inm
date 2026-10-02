# Session 02 — Configuration and dependency-light CLI

Read the shared contract and session 01 handoff. Inspect legacy `inm/protocol.py` for useful conventions, but do not relax its hard-coded study restrictions.

Implement validated new-study config, task and arm enumeration, and CLI plan/preflight commands. Resolve relative dataset/output paths against the config directory. Default dataset is the repository's sibling `ml/`, thus `../../ml` from `configs/`. Output is `../results/baselines-v1`; cross-session uses its own output directory.

Implement in the new configs and protocol module precisely the four arms and two split protocols specified by the read-only `plans/accuracy/CONTRACT.md`. Do not edit that contract, any plan, or the runner/checker. Record implementation decisions in `docs/baselines/protocol.md` and the handoff instead. Include model/training/selection settings, five degraded-mask repeats, and 2,000 subject bootstrap repeats if paired intervals are later implemented. Reject unknown options, invalid ranks/dimensions, inconsistent window sizes, invalid arm/regime combinations, and ambiguous selection policies. Validate numerical ranges without importing numerical packages.

`--plan` prints resolved paths, 27 tasks / 108 fits, protocols and arm coverage. `--preflight` calls diagnostics and returns nonzero for missing execution prerequisites, with actionable messages. Future execution dispatch can explicitly report 'not implemented yet'; it must not pretend to run training. Help/plan must work without torch.

Scope: protocol/CLI, two configs, protocol tests; tiny diagnostic wiring allowed.

Acceptance: both valid configs; malformed settings; config-relative paths from another CWD; plan subprocess works without importing torch; default matrix and classical full-only coverage; old 378-fit plan unchanged.

Handoff: validated config structure and CLI argument names. Freeze public field names for following sessions.
