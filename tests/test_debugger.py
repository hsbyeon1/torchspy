"""Tests for torchspy package."""

from pathlib import Path

import pytest
import torch
from torch import nn

from torchspy import (
    CallTracer,
    DebugContext,
    TensorSaver,
    get_debug_context,
    spy_save,
)


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
def debugger(tmp_path: Path) -> TensorSaver:
    """Create a TensorSaver with a temporary directory."""
    return TensorSaver(tmp_path)


@pytest.fixture
def model() -> SimpleModule:
    """Create a SimpleModule for testing."""
    return SimpleModule()


@pytest.fixture
def nested_model() -> NestedModule:
    """Create a NestedModule for testing."""
    return NestedModule()


class TestTensorSaver:
    """Tests for TensorSaver class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that init creates output directory."""
        output_dir = tmp_path / "debug"
        debugger = TensorSaver(output_dir)
        assert output_dir.exists()
        assert debugger.enabled

    def test_register_modules(self, debugger: TensorSaver, model: SimpleModule) -> None:
        """Test module registration."""
        debugger.register_modules(model, target_classes=(nn.Linear,))
        assert len(debugger.module_paths) == 1
        assert "linear" in next(iter(debugger.module_paths.values()))

    def test_register_by_name(self, debugger: TensorSaver, model: SimpleModule) -> None:
        """Test module registration by name pattern."""
        debugger.register_modules(model, target_names=["linear"])
        assert len(debugger.module_paths) == 1

    def test_save_tensor(self, debugger: TensorSaver, tmp_path: Path) -> None:
        """Test tensor saving."""
        tensor = torch.randn(2, 3)
        debugger.save("test", tensor)

        saved_path = tmp_path / "test.call0.pt"
        assert saved_path.exists()

        loaded = torch.load(saved_path, weights_only=True)
        assert torch.allclose(tensor, loaded)

    def test_call_counting(self, debugger: TensorSaver, tmp_path: Path) -> None:
        """Test that call counts increment correctly."""
        tensor = torch.randn(2, 3)

        debugger.save("test", tensor)
        debugger.save("test", tensor)

        assert (tmp_path / "test.call0.pt").exists()
        assert (tmp_path / "test.call1.pt").exists()

    def test_reset_counts(self, debugger: TensorSaver) -> None:
        """Test count reset."""
        tensor = torch.randn(2, 3)

        debugger.save("test", tensor)
        assert debugger.call_counts["test"] == 1

        debugger.reset_counts()
        assert debugger.call_counts["test"] == 0

    def test_disabled(self, tmp_path: Path) -> None:
        """Test that disabled debugger does not save."""
        debugger = TensorSaver(tmp_path, enabled=False)
        tensor = torch.randn(2, 3)
        debugger.save("test", tensor)

        saved_path = tmp_path / "test.call0.pt"
        assert not saved_path.exists()


class TestDebugContext:
    """Tests for DebugContext class."""

    def test_context_sets_and_resets(self, debugger: TensorSaver) -> None:
        """Test that context variable is properly set and reset."""
        assert get_debug_context() is None

        with DebugContext(debugger, prefix="test") as ctx:
            assert get_debug_context() is ctx

        assert get_debug_context() is None

    def test_spy_save_in_context(
        self, debugger: TensorSaver, model: SimpleModule, tmp_path: Path
    ) -> None:
        """Test spy_save within context."""
        debugger.register_modules(model, target_classes=(SimpleModule,))

        x = torch.randn(1, 4)
        with DebugContext(debugger, prefix="step0"):
            model(x)

        # Check that file was saved with correct naming
        files = list(tmp_path.glob("*.pt"))
        assert len(files) == 1
        assert "step0" in files[0].name
        assert "out" in files[0].name

    def test_spy_save_without_context(self) -> None:
        """Test that spy_save is no-op without context."""
        tensor = torch.randn(2, 3)
        # Should not raise
        spy_save("test", tensor)

    def test_prefix_in_filename(self, debugger: TensorSaver, tmp_path: Path) -> None:
        """Test that prefix is included in filename."""
        tensor = torch.randn(2, 3)

        with DebugContext(debugger, prefix="batch0", module_path_override="layer1"):
            ctx = get_debug_context()
            ctx._save("q", tensor)

        files = list(tmp_path.glob("*.pt"))
        assert len(files) == 1
        assert "batch0.layer1.q" in files[0].name


class TestCallTracer:
    """Tests for CallTracer class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that init creates output directory."""
        output_dir = tmp_path / "traces"
        tracer = CallTracer(output_dir)
        assert output_dir.exists()
        assert tracer.enabled

    def test_register_hooks(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test hook registration on target modules."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))
        assert len(tracer.hooks) == 3
        assert len(tracer.module_paths) == 3

    def test_traces_call_order(
        self, tmp_path: Path, nested_model: NestedModule
    ) -> None:
        """Test that call order is correctly recorded."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        x = torch.randn(1, 4)
        nested_model(x)

        assert len(tracer.call_trace) == 3
        assert tracer.call_trace[0] == "layer1"
        assert tracer.call_trace[1] == "layer2"
        assert tracer.call_trace[2] == "layer3"

    def test_save_trace(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test saving trace to file."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        x = torch.randn(1, 4)
        nested_model(x)

        path = tracer.save_trace("test_trace.txt")
        assert path.exists()

        content = path.read_text()
        lines = content.strip().split("\n")
        assert lines == ["layer1", "layer2", "layer3"]

    def test_reset_trace(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test resetting the trace."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        x = torch.randn(1, 4)
        nested_model(x)
        assert len(tracer.call_trace) == 3

        tracer.reset_trace()
        assert len(tracer.call_trace) == 0

    def test_remove_hooks(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test removing hooks."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))
        assert len(tracer.hooks) == 3

        tracer.remove_hooks()
        assert len(tracer.hooks) == 0

        # Verify hooks are actually removed - no more tracing
        tracer.reset_trace()
        x = torch.randn(1, 4)
        nested_model(x)
        assert len(tracer.call_trace) == 0

    def test_disabled_tracer(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test that disabled tracer does not record."""
        tracer = CallTracer(tmp_path, enabled=False)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        x = torch.randn(1, 4)
        nested_model(x)

        assert len(tracer.call_trace) == 0

    def test_context_manager(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test using tracer as context manager."""
        tracer = CallTracer(tmp_path, enabled=False)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        x = torch.randn(1, 4)

        with tracer:
            nested_model(x)

        assert len(tracer.call_trace) == 3

    def test_register_by_name(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test hook registration by name pattern."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_names=["layer1", "layer3"])

        x = torch.randn(1, 4)
        nested_model(x)

        assert len(tracer.call_trace) == 2
        assert "layer1" in tracer.call_trace
        assert "layer3" in tracer.call_trace

    def test_exclude_names(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test excluding modules by name pattern."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(
            nested_model, target_classes=(nn.Linear,), exclude_names=["layer2"]
        )

        x = torch.randn(1, 4)
        nested_model(x)

        assert len(tracer.call_trace) == 2
        assert "layer2" not in tracer.call_trace
