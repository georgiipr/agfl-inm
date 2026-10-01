# AGFL-inm working instructions

- Do not execute project code, tests, training, inference or diagnostics locally
  unless the user explicitly changes that restriction. Static inspection is allowed.
- Keep changes within this repository; preserve unrelated checkouts.
- No new dependencies or pytest prerequisite. Use the existing cluster environment.
- Train participants individually; keep model and attention choices separate.
- Use explicit masks and train-only statistics/factor fitting. Validation selects
  epochs; test scores must not select models or change the protocol silently.
- Keep all declared participants/results, including unfavorable outcomes.
- Preserve configuration, data/split/calibration and source provenance.
- Use Slurm for the supplied experiment and rsync when transferring artifacts.
- Do not claim accuracy or numerical correctness from static checks.
- Respect the user's delegation limit. Do not delegate unless explicitly requested.
- Keep public documentation independent of workstation paths and cluster accounts.
