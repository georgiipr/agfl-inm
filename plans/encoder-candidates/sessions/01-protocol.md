# Session 01 Declare protocol and dependency-light CLI

Bounded paths:

- `inm/encoder_candidates/__init__.py`
- `inm/encoder_candidates/__main__.py`
- `inm/encoder_candidates/protocol.py`
- `configs/encoder-candidates.json`
- `docs/encoder-candidates/protocol.md`
- `tests/encoder_candidates/test_protocol.py`

Implement the fixed protocol, strict config validation, task and arm enumeration,
study identity and lazy CLI. Read the shared contract and the two existing
protocol modules for patterns; do not import numerical packages at module scope.

Create the exact five-arm config, including the fixed architecture/training/
probe settings. Resolve paths relative to config and reject symlink overlap
in both directions. Preflight inventories required recordings and split metadata,
package discovery and known limits without loading EEG or old checkpoints. The
historical source mismatch does not block fresh training using verified splits.
Future CLI operations must report unimplemented, never success.

Acceptance: plan says 27 tasks / 135 neural fits / 108 CPU probes / zero Tucker
fits with numerical imports blocked. Reject unknown arms/settings, test exposure,
wrong regimes, budget search, duplicate IDs and invalid paths. Only explicitly
synthetic fixtures may shrink cohort/budget. Test provenance changes when source
or config bytes change. Document no fits have run; handoff exact keys and APIs.
