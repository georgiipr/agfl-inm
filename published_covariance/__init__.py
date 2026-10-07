"""Public data and static-mask API, imported lazily for dependency-light planning."""

from importlib import import_module

__all__ = [
    "CHANNEL_IDS", "CLASS_NAMES", "NativeRecording",
    "fit_native_normalization", "load_native_recording", "load_normalization",
    "normalize", "save_normalization", "stratified_split",
]

def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(import_module(".data", __name__), name)

def __dir__():
    return sorted(set(globals()) | set(__all__))
