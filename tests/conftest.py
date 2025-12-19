"""Shared test fixtures for torchspy tests."""

from pathlib import Path

import pytest
import torch
from torch import nn

from torchspy import TensorSaver, spy_save

# Import inside fixtures to avoid triggering package import-time side effects
# during pytest collection which can cause circular import issues.


class SimpleModule(nn.Module):
    """Simple module for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        spy_save("out", out, self)
        return out


class NestedModule(nn.Module):
    """Nested module for testing call tracing."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


@pytest.fixture
def saver(tmp_path: Path):
    """Create a TensorSaver with a temporary directory."""
    return TensorSaver(tmp_path)


@pytest.fixture
def simple_model() -> SimpleModule:
    """Create a SimpleModule for testing."""
    return SimpleModule()


@pytest.fixture
def nested_model() -> NestedModule:
    """Create a NestedModule for testing."""
    return NestedModule()
