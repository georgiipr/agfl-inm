"""EEGNet with independently selected spatial attention."""
from .._shared import ModelSpec
from .config import EEG_DEFAULTS
from .eeg import EEGModel

SPEC = ModelSpec('eegnet', EEG_DEFAULTS, EEGModel)
