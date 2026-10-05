"""Read-only reuse of the accepted CPU mathematics and calibration.

Only execution moves to CUDA; calibration and numerical state parsing stay CPU.
"""
from inm.task_driven_completion.completion import (
    validate_input, scoped_cpu_rng, fit_initialization, ZeroCompleter,
    TuckerCompleter, CovarianceCompleter, paired_completers,
    reconstruction_loss, completion_state_arrays, restore_completer,
)
