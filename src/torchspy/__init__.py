"""TorchSpy - A debug tool for PyTorch tensors.

This package provides utilities for saving and inspecting intermediate tensors
during PyTorch model forward passes, useful for debugging attention mechanisms
and other complex neural network components.

Example:
    >>> from torchspy import TensorSaver, DebugContext
    >>> from torchspy.saver import spy_save
    >>>
    >>> # Setup saver and register target modules
    >>> saver = TensorSaver("./debug_tensors")
    >>> saver.register_modules(model, target_classes=(AttentionLayer,))
    >>>
    >>> # Run inference with debug context
    >>> with DebugContext(saver, prefix="step0"):
    ...     output = model(inputs)
    >>>
    >>> # Inside your module's forward(), call spy_save():
    >>> # spy_save("q", q, self)  # saves as step0.{module_path}.q.call0.pt

"""

__all__ = [
    "BaseDebugger",
    "CallTracer",
    "DebugContext",
    "TensorSaver",
    "get_debug_context",
    "spy_save",
]

from typing import TYPE_CHECKING

# No eager imports here to avoid circular import problems during test collection.
# Use the lazy __getattr__ below so attributes are imported only when accessed.

if TYPE_CHECKING:
    # For static type checkers and Sphinx autodoc
    from .context import DebugContext, get_debug_context  # noqa: F401
    from .saver import TensorSaver, spy_save  # noqa: F401
    from .tracer import CallTracer  # noqa: F401


def __getattr__(name: str):
    if name == "TensorSaver":
        from .saver import TensorSaver

        return TensorSaver

    if name == "spy_save":
        from .saver import spy_save

        return spy_save
    if name == "DebugContext":
        from .context import DebugContext

        return DebugContext
    if name == "get_debug_context":
        from .context import get_debug_context

        return get_debug_context
    if name == "CallTracer":
        from .tracer import CallTracer

        return CallTracer
    if name == "BaseDebugger":
        from ._base import BaseDebugger

        return BaseDebugger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
