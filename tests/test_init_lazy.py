import importlib

import pytest


def test_lazy_attributes_are_available():
    # Import the package and ensure attributes trigger lazy loading
    torchspy = importlib.import_module("torchspy")
    # Access known lazy attributes
    assert hasattr(torchspy, "TensorSaver")
    assert callable(torchspy.TensorSaver)
    assert hasattr(torchspy, "spy_save")
    # Non-existent attribute should raise AttributeError
    with pytest.raises(AttributeError):
        _ = torchspy.NON_EXISTENT_ATTRIBUTE
