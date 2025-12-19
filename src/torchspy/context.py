"""Debug context for scoped tensor debugging.

This module provides the DebugContext class and helper functions for
thread-safe tensor debugging within PyTorch model forward passes.
"""

import contextvars
import logging
from typing import TYPE_CHECKING

from torch import Tensor, nn

if TYPE_CHECKING:
    from torchspy.saver import TensorSaver

logger = logging.getLogger(__name__)


# Context variable for thread-safe debug context
_debug_context: contextvars.ContextVar["DebugContext | None"] = contextvars.ContextVar(
    "debug_context", default=None
)


class DebugContext:
    """Context manager for scoped tensor debugging.

    Use this to add a prefix to saved tensors and enable spy_save() calls
    within module forward methods.

    Attributes:
        saver (TensorSaver): The saver instance to use.
        prefix (str): Prefix added to all tensor names in this context.
        module_path_override (str | None): Override module path for spy_save().

    Example:
        >>> from torchspy import TensorSaver, DebugContext
        >>>
        >>> saver = TensorSaver("./debug_tensors")
        >>> with DebugContext(saver, prefix="batch0_step0"):
        ...     output = model(inputs)
        ...     # Inside forward: spy_save("q", q, self)
        ...     # Saves as: batch0_step0.{module_path}.q.call0.pt

    """

    def __init__(
        self,
        saver: "TensorSaver",
        prefix: str = "",
        module_path_override: str | None = None,
    ) -> None:
        """Initialize the debug context.

        Args:
            saver (TensorSaver): The saver instance.
            prefix (str): Prefix for tensor names. Defaults to "".
            module_path_override (str | None): Override the module path.
                Useful when calling spy_save() from helper functions.

        """
        self.saver = saver
        self.prefix = prefix
        self.module_path_override = module_path_override
        self._token: contextvars.Token | None = None

    # Backward compatibility alias
    @property
    def debugger(self) -> "TensorSaver":
        """Backward compatibility alias for saver."""
        return self.saver

    def __enter__(self) -> "DebugContext":
        """Enter the debug context."""
        self._token = _debug_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the debug context."""
        if self._token is not None:
            _debug_context.reset(self._token)

    def _save(self, name: str, tensor: Tensor, module: nn.Module | None = None) -> None:
        """Save a tensor with context-aware naming.

        Args:
            name (str): The tensor variable name (e.g., "q", "attn_mask").
            tensor (Tensor): The tensor to save.
            module (nn.Module | None): The module saving this tensor.
                Used to look up the module path.

        """
        if self.module_path_override is not None:
            module_path = self.module_path_override
        elif module is not None:
            module_path = self.saver.get_module_path(module)
        else:
            module_path = "manual"

        if module_path == "unknown":
            logger.warning("Module path unknown for tensor '%s'. Skipping save.", name)
            return

        full_name = (
            f"{self.prefix}.{module_path}.{name}"
            if self.prefix
            else f"{module_path}.{name}"
        )
        self.saver.save(full_name, tensor)


def get_debug_context() -> DebugContext | None:
    """Get the current debug context.

    Returns:
        DebugContext | None: The active context, or None if no context is active.

    """
    return _debug_context.get()
