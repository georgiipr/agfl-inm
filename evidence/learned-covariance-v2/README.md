# Learned covariance evidence

Use the [reproduction guide](../../docs/task-driven-completion-cuda/reproduce.md)
to verify saved scores on CPU or prepare a fresh CUDA experiment.

`study-bundle.tar.gz` preserves the completed study, required historical backbones
and split metadata, execution/audit receipts, and failed-attempt provenance.
`manifest.json` records all file hashes and the original scientific source and
package versions. Raw GDF recordings are not included.

The readable `report/` files are byte-identical copies of the archived report.
The numerical verifier reads the archived probabilities rather than trusting a
summary table. Results are from reused validation selection, not independent testing.
