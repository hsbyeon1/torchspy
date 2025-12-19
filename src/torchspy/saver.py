"""TensorSaver for saving intermediate tensors during PyTorch forward passes.

This module provides the TensorSaver class that manages tensor saving,
module path registration, and call counting.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Type

import torch
import torch.linalg as LA
from torch import Tensor, nn

from torchspy._base import BaseDebugger
from torchspy.context import get_debug_context

logger = logging.getLogger(__name__)


class TensorSaver(BaseDebugger):
    """Utility to save intermediate tensors during PyTorch forward passes.

    This class registers module paths and provides infrastructure for saving
    tensors from within module forward methods via the spy_save() function.

    Attributes:
        output_dir (Path): Directory where tensors are saved.
        enabled (bool): Whether debugging is active.
        call_counts (dict[str, int]): Tracks call count per tensor name.
        module_paths (dict[int, str]): Maps module id to its path string.

    Example:
        >>> from torchspy import TensorSaver, DebugContext, spy_save
        >>>
        >>> # Setup saver
        >>> saver = TensorSaver("./debug_tensors")
        >>> saver.register_modules(model, target_classes=(AttentionLayer,))
        >>>
        >>> # Run with debug context
        >>> with DebugContext(saver, prefix="step0"):
        ...     output = model(inputs)
        >>>
        >>> # Inside your module's forward(), call spy_save():
        >>> # spy_save("q", q, self)  # saves as {prefix}.{module_path}.q.call{n}.pt

    """

    def __init__(self, output_dir: str | Path, enabled: bool = True) -> None:
        """Initialize the tensor saver.

        Args:
            output_dir (str | Path): Directory to save tensor files.
            enabled (bool): Whether saving is enabled. Defaults to True.

        """
        super().__init__(output_dir, enabled)
        self.call_counts: dict[str, int] = defaultdict(int)

    def register_modules(
        self,
        model: nn.Module,
        target_classes: tuple[Type[nn.Module], ...] = (nn.Module,),
        target_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ) -> None:
        """Register module paths for target modules.

        This populates the module_paths mapping so that spy_save() can
        look up the module's path in the model hierarchy.

        Args:
            model (nn.Module): The root model to inspect.
            target_classes (tuple[Type[nn.Module], ...]): Register modules
                that are instances of these classes.
            target_names (list[str] | None): Register modules whose path
                contains any of these substrings.
            exclude_names (list[str] | None): Exclude modules whose path
                contains any of these substrings.

        """
        for name, module in model.named_modules():
            module_path = name or "root"
            if self._should_register_module(
                module, module_path, target_classes, target_names, exclude_names
            ):
                self.module_paths[id(module)] = module_path
                logger.info("Registered module path: %s", module_path)

    def save(
        self,
        name: str,
        tensor: Tensor,
        call_idx: int | None = None,
        norm_only: bool = False,
    ) -> None:
        """Save a tensor to disk.

        Args:
            name (str): The tensor name (will be used in filename).
            tensor (Tensor): The tensor to save.
            call_idx (int | None): Call index. If None, auto-increments
                based on name.
            norm_only (bool): If True, save only the L2 norm of the tensor
                (flattened per batch). Defaults to False.

        """
        if not self.enabled:
            return

        if call_idx is None:
            call_idx = self.call_counts[name]
            self.call_counts[name] += 1

        filename = f"{name}.call{call_idx}.pt"
        path = self.output_dir / filename
        if norm_only:
            tensor = tensor.view(tensor.size(0), -1)
            tensor = LA.norm(tensor, dim=-1)
        torch.save(tensor.detach().cpu(), path)
        logger.debug("Saved tensor: %s", path)

    def reset_counts(self) -> None:
        """Reset all call counters.

        Call this between batches if you want per-batch indexing.

        """
        self.call_counts.clear()


def spy_save(name: str, tensor: Tensor, module: nn.Module | None = None) -> None:
    """Save a tensor if a debug context is active.

    This is a convenience function to call from within module forward methods.
    If no debug context is active, this function does nothing (no-op).

    Args:
        name (str): The tensor variable name (e.g., "q", "k", "attn_mask").
        tensor (Tensor): The tensor to save.
        module (nn.Module | None): The module instance (self). Used to determine
            the module path. Pass  from within a module's forward().

    Example:
        >>> class MyModule(nn.Module):
        ...     def forward(self, x):
        ...         q = self.proj_q(x)
        ...         spy_save("q", q, self)
        ...         return q

    """
    ctx = get_debug_context()
    if ctx is not None:
        ctx._save(name, tensor, module)
