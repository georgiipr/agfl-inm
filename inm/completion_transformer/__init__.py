"""Raw completion replay study with dependency-light protocol exports.

Planning and provenance imports stay in the Python standard library. Numerical
components are loaded only when one of their public names is requested.
"""

from .protocol import (
    ARM_IDS,
    CONDITIONS,
    STRATEGY_IDS,
    inspect_inputs,
    load_config,
    study_identity,
    tasks,
)

__all__ = [
    "ARM_IDS",
    "CONDITIONS",
    "STRATEGY_IDS",
    "inspect_inputs",
    "load_config",
    "study_identity",
    "tasks",
    "CompletionAdapter",
    "CovarianceCompleter",
    "TuckerCompleter",
    "completion_state_arrays",
    "load_completion_state",
    "preserve_observed",
    "verify_historical_task",
    "prepare_task",
    "restore_backbones",
    "replay_full_input",
    "run_task",
    "summarize",
]


def __getattr__(name: str):
    if name == "CompletionAdapter":
        from .adapters import CompletionAdapter
        return CompletionAdapter
    if name in {"CovarianceCompleter", "TuckerCompleter", "preserve_observed",
                "completion_state_arrays", "load_completion_state"}:
        from .completion import (CovarianceCompleter, TuckerCompleter, preserve_observed,
                                 completion_state_arrays, load_completion_state)
        return {
            "CovarianceCompleter": CovarianceCompleter,
            "TuckerCompleter": TuckerCompleter,
            "preserve_observed": preserve_observed,
            "completion_state_arrays": completion_state_arrays,
            "load_completion_state": load_completion_state,
        }[name]
    if name in {"verify_historical_task", "prepare_task", "restore_backbones", "replay_full_input"}:
        from . import replay
        return getattr(replay, name)
    if name == "run_task":
        from .study import run_task
        return run_task
    if name == "summarize":
        from .reporting import summarize
        return summarize
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
