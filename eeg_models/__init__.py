"""EEG models with built-in MHA and helpers for the INM experiment."""
import sys

# Keep this check parseable by older system interpreters so direct launches
# explain the required environment before importing modules with newer syntax.
if sys.version_info < (3, 12):
    raise ImportError(
        "EEG models require Python 3.12 or newer; running Python {0} at {1}. "
        "Activate the project's Python 3.12+ virtual environment and relaunch."
        .format(sys.version.split()[0], sys.executable)
    )

__version__ = "0.5.0"
