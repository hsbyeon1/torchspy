"""Tests for TensorSaver class."""

from pathlib import Path

import torch
from torch import nn

from torchspy import TensorSaver
from .conftest import SimpleModule


class TestTensorSaver:
    """Tests for TensorSaver class."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """Test that init creates output directory."""
        output_dir = tmp_path / "debug"
        saver = TensorSaver(output_dir)
        assert output_dir.exists()
        assert saver.enabled

    def test_register_modules(
        self, saver: TensorSaver, simple_model: SimpleModule
    ) -> None:
        """Test module registration."""
        saver.register_modules(simple_model, target_classes=(nn.Linear,))
        assert len(saver.module_paths) == 1
        assert "linear" in next(iter(saver.module_paths.values()))

    def test_register_by_name(
        self, saver: TensorSaver, simple_model: SimpleModule
    ) -> None:
        """Test module registration by name pattern."""
        saver.register_modules(simple_model, target_names=["linear"])
        assert len(saver.module_paths) == 1

    def test_register_exclude_names(
        self, saver: TensorSaver, simple_model: SimpleModule
    ) -> None:
        """Test module exclusion by name pattern."""
        saver.register_modules(
            simple_model,
            target_classes=(nn.Linear,),
            exclude_names=["linear"],
        )
        assert len(saver.module_paths) == 0

    def test_save_tensor(self, saver: TensorSaver, tmp_path: Path) -> None:
        """Test tensor saving."""
        tensor = torch.randn(2, 3)
        saver.save("test", tensor)

        saved_path = tmp_path / "test.call0.pt"
        assert saved_path.exists()

        loaded = torch.load(saved_path, weights_only=True)
        assert torch.allclose(tensor, loaded)

    def test_call_counting(self, saver: TensorSaver, tmp_path: Path) -> None:
        """Test that call counts increment correctly."""
        tensor = torch.randn(2, 3)

        saver.save("test", tensor)
        saver.save("test", tensor)

        assert (tmp_path / "test.call0.pt").exists()
        assert (tmp_path / "test.call1.pt").exists()

    def test_explicit_call_idx(self, saver: TensorSaver, tmp_path: Path) -> None:
        """Test saving with explicit call index."""
        tensor = torch.randn(2, 3)
        saver.save("test", tensor, call_idx=42)

        assert (tmp_path / "test.call42.pt").exists()
        assert saver.call_counts["test"] == 0  # Should not auto-increment

    def test_reset_counts(self, saver: TensorSaver) -> None:
        """Test count reset."""
        tensor = torch.randn(2, 3)

        saver.save("test", tensor)
        assert saver.call_counts["test"] == 1

        saver.reset_counts()
        assert saver.call_counts["test"] == 0

    def test_disabled(self, tmp_path: Path) -> None:
        """Test that disabled saver does not save."""
        saver = TensorSaver(tmp_path, enabled=False)
        tensor = torch.randn(2, 3)
        saver.save("test", tensor)

        saved_path = tmp_path / "test.call0.pt"
        assert not saved_path.exists()

    def test_get_module_path(
        self, saver: TensorSaver, simple_model: SimpleModule
    ) -> None:
        """Test getting module path."""
        saver.register_modules(simple_model, target_classes=(nn.Linear,))
        path = saver.get_module_path(simple_model.linear)
        assert path == "linear"

    def test_get_module_path_unknown(self, saver: TensorSaver) -> None:
        """Test getting path for unregistered module."""
        module = nn.Linear(4, 4)
        path = saver.get_module_path(module)
        assert path == "unknown"

    def test_norm_only(self, saver: TensorSaver, tmp_path: Path) -> None:
        """Test saving only the norm of tensor."""
        tensor = torch.randn(2, 3, 4)
        saver.save("test", tensor, norm_only=True)

        saved_path = tmp_path / "test.call0.pt"
        loaded = torch.load(saved_path, weights_only=True)
        assert loaded.shape == (2,)  # Batch size only
