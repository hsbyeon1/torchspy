"""Tests for DebugContext class and spy_save function."""

from pathlib import Path

import torch

from torchspy import DebugContext, TensorSaver, get_debug_context, spy_save
from .conftest import SimpleModule


class TestDebugContext:
    """Tests for DebugContext class."""

    def test_context_sets_and_resets(self, saver: TensorSaver) -> None:
        """Test that context variable is properly set and reset."""
        assert get_debug_context() is None

        with DebugContext(saver, prefix="test") as ctx:
            assert get_debug_context() is ctx

        assert get_debug_context() is None

    def test_nested_contexts(self, saver: TensorSaver) -> None:
        """Test nested debug contexts."""
        with DebugContext(saver, prefix="outer") as outer_ctx:
            assert get_debug_context() is outer_ctx

            with DebugContext(saver, prefix="inner") as inner_ctx:
                assert get_debug_context() is inner_ctx

            assert get_debug_context() is outer_ctx

        assert get_debug_context() is None

    def test_spy_save_in_context(
        self, saver: TensorSaver, simple_model: SimpleModule, tmp_path: Path
    ) -> None:
        """Test spy_save within context."""
        saver.register_modules(simple_model, target_classes=(SimpleModule,))

        x = torch.randn(1, 4)
        with DebugContext(saver, prefix="step0"):
            simple_model(x)

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

    def test_prefix_in_filename(self, saver: TensorSaver, tmp_path: Path) -> None:
        """Test that prefix is included in filename."""
        tensor = torch.randn(2, 3)

        with DebugContext(saver, prefix="batch0", module_path_override="layer1"):
            ctx = get_debug_context()
            ctx._save("q", tensor)

        files = list(tmp_path.glob("*.pt"))
        assert len(files) == 1
        assert "batch0.layer1.q" in files[0].name

    def test_no_prefix(self, saver: TensorSaver, tmp_path: Path) -> None:
        """Test saving without prefix."""
        tensor = torch.randn(2, 3)

        with DebugContext(saver, module_path_override="layer1"):
            ctx = get_debug_context()
            ctx._save("q", tensor)

        files = list(tmp_path.glob("*.pt"))
        assert len(files) == 1
        assert "layer1.q" in files[0].name
        assert ".." not in files[0].name  # No double dots from empty prefix

    def test_module_path_override(self, saver: TensorSaver, tmp_path: Path) -> None:
        """Test module path override."""
        tensor = torch.randn(2, 3)

        with DebugContext(saver, module_path_override="custom.path"):
            ctx = get_debug_context()
            ctx._save("tensor", tensor)

        files = list(tmp_path.glob("*.pt"))
        assert len(files) == 1
        assert "custom.path" in files[0].name

    def test_manual_module_path(self, saver: TensorSaver, tmp_path: Path) -> None:
        """Test saving without module reference uses 'manual' path."""
        tensor = torch.randn(2, 3)

        with DebugContext(saver):
            ctx = get_debug_context()
            ctx._save("tensor", tensor)

        files = list(tmp_path.glob("*.pt"))
        assert len(files) == 1
        assert "manual" in files[0].name

    def test_backward_compatibility_debugger_property(self, saver: TensorSaver) -> None:
        """Test that debugger property works for backward compatibility."""
        ctx = DebugContext(saver)
        assert ctx.debugger is saver


class TestDebugSave:
    """Tests for spy_save function."""

    def test_spy_save_with_module(
        self, saver: TensorSaver, simple_model: SimpleModule, tmp_path: Path
    ) -> None:
        """Test spy_save with module reference."""
        saver.register_modules(simple_model, target_classes=(SimpleModule,))
        tensor = torch.randn(2, 3)

        with DebugContext(saver, prefix="test"):
            spy_save("tensor", tensor, simple_model)

        files = list(tmp_path.glob("*.pt"))
        assert len(files) == 1
        assert "root" in files[0].name  # SimpleModule is root

    def test_spy_save_without_module(self, saver: TensorSaver, tmp_path: Path) -> None:
        """Test spy_save without module reference."""
        tensor = torch.randn(2, 3)

        with DebugContext(saver, prefix="test"):
            spy_save("tensor", tensor)

        files = list(tmp_path.glob("*.pt"))
        assert len(files) == 1
        assert "manual" in files[0].name


class TestGetDebugContext:
    """Tests for get_debug_context function."""

    def test_returns_none_outside_context(self) -> None:
        """Test that get_debug_context returns None outside context."""
        assert get_debug_context() is None

    def test_returns_context_inside_context(self, saver: TensorSaver) -> None:
        """Test that get_debug_context returns context inside context."""
        with DebugContext(saver) as ctx:
            assert get_debug_context() is ctx
