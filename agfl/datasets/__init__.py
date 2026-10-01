"""EEG dataset contract and persisted split helpers."""
from .base import SignalDataset
from .splits import get_split, validate_split

__all__ = ["SignalDataset", "get_split", "validate_split"]
