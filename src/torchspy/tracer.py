"""CallTracer for recording module execution order.

This module provides the CallTracer class that uses PyTorch forward hooks
to automatically record the order of module calls during a forward pass.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Type

from torch import nn

from torchspy._base import BaseDebugger

logger = logging.getLogger(__name__)


class CallTracer(BaseDebugger):
    """Traces the execution order of PyTorch modules using forward hooks.

    This class registers forward hooks on target modules to automatically
    record their call order without requiring any modifications to the
    module code.

    Attributes:
        output_dir (Path): Directory where trace files are saved.
        enabled (bool): Whether tracing is active.
        call_trace (list[str]): List of module paths in call order.
        module_paths (dict[int, str]): Maps module id to its path string.
        hooks (list): List of registered hook handles for cleanup.

    Example:
        >>> from torchspy import CallTracer
        >>>
        >>> tracer = CallTracer("./debug_traces")
        >>> tracer.register_hooks(model, target_classes=(nn.Linear, nn.MultiheadAttention))
        >>>
        >>> # Run forward pass - hooks automatically record call order
        >>> output = model(inputs)
        >>>
        >>> # Save the trace
        >>> tracer.save_trace("forward_pass.txt")
        >>> # Or get the trace as a list
        >>> print(tracer.call_trace)

    """

    def __init__(self, output_dir: str | Path, enabled: bool = True) -> None:
        """Initialize the call tracer.

        Args:
            output_dir (str | Path): Directory to save trace files.
            enabled (bool): Whether tracing is enabled. Defaults to True.

        """
        super().__init__(output_dir, enabled)
        self.call_trace: list[str] = []
        self.hooks: list[Any] = []

    def register_hooks(
        self,
        model: nn.Module,
        target_classes: tuple[Type[nn.Module], ...] = (nn.Module,),
        target_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ) -> None:
        """Register forward hooks on target modules.

        This method walks through the model hierarchy and registers forward
        hooks on modules that match the specified criteria.

        Args:
            model (nn.Module): The root model to inspect.
            target_classes (tuple[Type[nn.Module], ...]): Register hooks on
                modules that are instances of these classes. Defaults to (nn.Module,).
            target_names (list[str] | None): Register hooks on modules whose
                path contains any of these substrings.
            exclude_names (list[str] | None): Exclude modules whose path
                contains any of these substrings.

        """
        for name, module in model.named_modules():
            module_path = name or "root"
            if self._should_register_module(
                module, module_path, target_classes, target_names, exclude_names
            ):
                self.module_paths[id(module)] = module_path
                hook = module.register_forward_hook(self._create_hook(module_path))
                self.hooks.append(hook)
                logger.info("Registered hook on: %s", module_path)

    def _create_hook(self, module_path: str) -> Callable:
        """Create a forward hook that records the module path.

        Args:
            module_path (str): The path of the module in the model hierarchy.

        Returns:
            Callable: A hook function that records the module call.

        """

        def hook(module: nn.Module, input: Any, output: Any) -> None:  # noqa: A002
            if self.enabled:
                self.call_trace.append(module_path)

        return hook

    def save_trace(self, filename: str = "call_trace.txt") -> Path:
        """Save the call trace to a text file.

        Args:
            filename (str): Name of the output file. Defaults to "call_trace.txt".

        Returns:
            Path: Path to the saved trace file.

        """
        path = self.output_dir / filename
        with open(path, "w") as f:
            f.write("\n".join(self.call_trace))
        logger.info("Saved call trace to: %s (%d calls)", path, len(self.call_trace))
        return path

    def get_trace(self) -> list[str]:
        """Get the current call trace.

        Returns:
            list[str]: List of module paths in call order.

        """
        return self.call_trace.copy()

    def reset_trace(self) -> None:
        """Clear the recorded call trace."""
        self.call_trace.clear()

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        logger.info("Removed all hooks")

    def __enter__(self) -> "CallTracer":
        """Enter context - enable tracing."""
        self.enabled = True
        self.reset_trace()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - optionally save trace."""
        pass

    def __del__(self) -> None:
        """Clean up hooks on deletion."""
        self.remove_hooks()
