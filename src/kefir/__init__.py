"""Public entry points; corpus tools do not import the training runtime."""
from importlib import import_module

_EXPORTS = {
    "ExecutionSpec": ".training", "RunSpec": ".training", "TrainResult": ".training",
    "train": ".training", "evaluate_checkpoint": ".evaluate",
    "generate_text": ".generate", "load_tokenizer": ".tokenizer",
    "load_weights": ".artifact", "save_weights": ".artifact",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
