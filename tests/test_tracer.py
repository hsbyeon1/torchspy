"""Tests for CallTracer class."""

from pathlib import Path

import torch
from torch import nn

from torchspy import CallTracer
from .conftest import NestedModule


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

    def test_multiple_forward_passes(
        self, tmp_path: Path, nested_model: NestedModule
    ) -> None:
        """Test tracing accumulates over multiple forward passes."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        x = torch.randn(1, 4)
        nested_model(x)
        nested_model(x)

        assert len(tracer.call_trace) == 6  # 3 layers * 2 passes

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

    def test_save_trace_default_filename(
        self, tmp_path: Path, nested_model: NestedModule
    ) -> None:
        """Test saving trace with default filename."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        x = torch.randn(1, 4)
        nested_model(x)

        path = tracer.save_trace()
        assert path.name == "call_trace.txt"

    def test_get_trace(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test getting trace as a copy."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        x = torch.randn(1, 4)
        nested_model(x)

        trace = tracer.get_trace()
        assert trace == ["layer1", "layer2", "layer3"]

        # Verify it's a copy
        trace.append("extra")
        assert "extra" not in tracer.call_trace

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

    def test_context_manager_resets_trace(
        self, tmp_path: Path, nested_model: NestedModule
    ) -> None:
        """Test that context manager resets trace on entry."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        x = torch.randn(1, 4)
        nested_model(x)
        assert len(tracer.call_trace) == 3

        with tracer:
            assert len(tracer.call_trace) == 0  # Reset on entry
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

    def test_get_module_path(self, tmp_path: Path, nested_model: NestedModule) -> None:
        """Test getting module path."""
        tracer = CallTracer(tmp_path)
        tracer.register_hooks(nested_model, target_classes=(nn.Linear,))

        path = tracer.get_module_path(nested_model.layer1)
        assert path == "layer1"

    def test_get_module_path_unknown(self, tmp_path: Path) -> None:
        """Test getting path for unregistered module."""
        tracer = CallTracer(tmp_path)
        module = nn.Linear(4, 4)
        path = tracer.get_module_path(module)
        assert path == "unknown"
