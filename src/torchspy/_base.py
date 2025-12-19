"""Base class for tensor debugging utilities.

This module provides the BaseDebugger class with common functionality
shared between TensorSaver and CallTracer.
"""

import logging
from pathlib import Path
from typing import Type

from torch import nn

logger = logging.getLogger(__name__)


class BaseDebugger:
    """Base class for tensor debugging utilities.

    Provides common functionality for directory management and module
    registration that is shared between TensorSaver and CallTracer.

    Attributes:
        output_dir (Path): Directory where output files are saved.
        enabled (bool): Whether the debugger is active.
        module_paths (dict[int, str]): Maps module id to its path string.

    """

    def __init__(self, output_dir: str | Path, enabled: bool = True) -> None:
        """Initialize the base debugger.

        Args:
            output_dir (str | Path): Directory to save output files.
            enabled (bool): Whether debugging is enabled. Defaults to True.

        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.module_paths: dict[int, str] = {}

    def _should_register_module(
        self,
        module: nn.Module,
        module_path: str,
        target_classes: tuple[Type[nn.Module], ...],
        target_names: list[str] | None,
        exclude_names: list[str] | None,
    ) -> bool:
        """Determine if a module should be registered.

        Args:
            module (nn.Module): The module to check.
            module_path (str): The module's path in the model hierarchy.
            target_classes (tuple[Type[nn.Module], ...]): Register if instance of these.
            target_names (list[str] | None): Register if path contains any of these.
            exclude_names (list[str] | None): Exclude if path contains any of these.

        Returns:
            bool: True if the module should be registered.

        """

        if not isinstance(module, target_classes):
            return False

        if target_names is not None and not any(t in module_path for t in target_names):
            return False

        if exclude_names is not None and any(t in module_path for t in exclude_names):
            return False

        return True

    def get_module_path(self, module: nn.Module) -> str:
        """Get the registered path for a module.

        Args:
            module (nn.Module): The module to look up.

        Returns:
            str: The module's path, or "unknown" if not registered.

        """
        return self.module_paths.get(id(module), "unknown")
