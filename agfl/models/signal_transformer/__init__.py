"""Signal Transformer with EEG electrode tokens."""

from .._shared import ModelSpec
from .config import DEFAULTS
from .eeg import EEGModel

SPEC = ModelSpec('signal_transformer', DEFAULTS, EEGModel)

__all__ = ["SPEC", "EEGModel"]
