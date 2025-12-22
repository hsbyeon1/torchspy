from types import SimpleNamespace

from torchspy.context import DebugContext, get_debug_context


class DummySaver:
    def __init__(self):
        self.saved = []

    def get_module_path(self, module):
        return getattr(module, "_mp", "unknown")

    def save(self, name, tensor):
        self.saved.append((name, tensor))


def test_debug_context_enter_exit_and_get(tmp_path):
    saver = DummySaver()
    assert get_debug_context() is None
    ctx = DebugContext(saver, prefix="pfx")
    with ctx:
        assert get_debug_context() is ctx
    assert get_debug_context() is None


def test_debug_context_save_skips_unknown_and_warns(caplog):
    saver = DummySaver()
    ctx = DebugContext(saver, prefix="pfx")
    m = SimpleNamespace(_mp="unknown")
    with ctx:
        # call _save with a module whose path resolves to 'unknown'
        ctx._save("x", "tensor", module=m)
    # saver.save should not have been called
    assert saver.saved == []
    # warning logged
    assert any("Module path unknown" in r.message for r in caplog.records)


def test_debug_context_save_with_override(tmp_path):
    saver = DummySaver()
    ctx = DebugContext(saver, prefix="pfx", module_path_override="OVERRIDE")
    with ctx:
        ctx._save("name", "tensor", module=None)
    assert saver.saved[0][0].startswith("pfx.OVERRIDE.name")
