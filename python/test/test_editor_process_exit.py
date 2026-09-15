from __future__ import annotations

import pytest
import gc
import signal
from types import SimpleNamespace

from Infernux.engine.engine import Engine
from Infernux.plugins import PluginManager


class _PluginManagerProbe:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error
        self.engine = None

    def shutdown(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


def _engine_without_native_runtime(*, process_owned: bool) -> Engine:
    engine = Engine.__new__(Engine)
    engine._process_owned_exit = process_owned
    return engine


def test_standalone_process_exit_does_not_unload_process_owned_plugins(monkeypatch):
    manager = _PluginManagerProbe()
    monkeypatch.setattr(PluginManager, "_instance", manager)
    engine = _engine_without_native_runtime(process_owned=True)

    engine._shutdown_plugins_for_exit()

    assert manager.calls == 0


def test_embedded_engine_exit_preserves_plugin_unload_contract(monkeypatch):
    manager = _PluginManagerProbe()
    monkeypatch.setattr(PluginManager, "_instance", manager)
    engine = _engine_without_native_runtime(process_owned=False)
    manager.engine = engine

    engine._shutdown_plugins_for_exit()

    assert manager.calls == 1


def test_embedded_plugin_unload_failure_is_not_suppressed(monkeypatch):
    manager = _PluginManagerProbe(error=RuntimeError("unload failed"))
    monkeypatch.setattr(PluginManager, "_instance", manager)
    engine = _engine_without_native_runtime(process_owned=False)
    manager.engine = engine

    with pytest.raises(RuntimeError, match="unload failed"):
        engine._shutdown_plugins_for_exit()


@pytest.mark.parametrize("owner", [None, object()])
def test_temporary_cook_host_does_not_unload_callers_plugins(monkeypatch, owner):
    manager = _PluginManagerProbe()
    manager.engine = owner
    monkeypatch.setattr(PluginManager, "_instance", manager)
    engine = _engine_without_native_runtime(process_owned=False)

    engine._shutdown_plugins_for_exit()

    assert manager.calls == 0
    assert PluginManager.instance() is manager


@pytest.mark.parametrize("error", [None, KeyboardInterrupt(), RuntimeError("frame failed")])
@pytest.mark.parametrize("gc_enabled", [True, False])
def test_native_loop_always_exits_and_restores_gc(error, gc_enabled):
    calls = []
    engine = Engine.__new__(Engine)

    def run():
        assert not gc.isenabled()
        if error is not None:
            raise error

    engine._engine = SimpleNamespace(run=run)
    engine.exit = lambda: calls.append("exit")
    previous_gc = gc.isenabled()
    (gc.enable if gc_enabled else gc.disable)()
    try:
        if error is None:
            engine._run_native_loop()
        else:
            with pytest.raises(type(error)):
                engine._run_native_loop()
        assert calls == ["exit"]
        assert gc.isenabled() is gc_enabled
    finally:
        (gc.enable if previous_gc else gc.disable)()


def test_exit_after_native_teardown_is_a_noop():
    engine = Engine.__new__(Engine)
    engine._engine = None
    engine.exit()


def test_sigint_finishes_current_frame_before_cleanup():
    calls = []
    engine = Engine.__new__(Engine)

    def run():
        signal.raise_signal(signal.SIGINT)
        calls.append("frame-finished")

    engine._engine = SimpleNamespace(run=run, exit=lambda: calls.append("request-exit"))
    engine.exit = lambda: calls.append("cleanup")
    previous_handler = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        engine._run_native_loop()
    assert calls == ["request-exit", "frame-finished", "cleanup"]
    assert signal.getsignal(signal.SIGINT) is previous_handler


@pytest.mark.parametrize("phase", ["startup", "loop", "normal"])
def test_editor_entry_point_cleans_up_before_releasing_project_lock(monkeypatch, phase):
    import Infernux.engine as entry
    from Infernux.engine import bootstrap as bootstrap_module
    from Infernux.engine import library_sync

    calls = []

    def start():
        if phase == "startup":
            raise KeyboardInterrupt()

    def run():
        if phase == "loop":
            raise KeyboardInterrupt()

    engine = SimpleNamespace(
        set_window_icon=lambda *_: None,
        set_window_title=lambda *_: None,
        show=lambda: None,
        run=run,
        exit=lambda: calls.append("cleanup"),
    )
    monkeypatch.setattr(bootstrap_module, "EditorBootstrap", lambda *_: SimpleNamespace(engine=engine, run=start))
    monkeypatch.setattr(bootstrap_module, "_signal_progress", lambda *_: None)
    monkeypatch.setattr(library_sync, "sync_resources", lambda *_: None)
    monkeypatch.setattr(entry._resources, "activate_library", lambda *_: None)
    monkeypatch.setattr(entry, "_acquire_project_lock", lambda *_: ("lock", "token"))
    monkeypatch.setattr(entry, "_remove_project_lock", lambda *_: calls.append("unlock"))
    monkeypatch.setattr(entry, "_signal_engine_loaded", lambda: None)
    # A successful standalone launch must return after real cleanup, not
    # bypass interpreter teardown (and pending output) with os._exit.
    monkeypatch.setattr(entry.os, "_exit", lambda *_: pytest.fail("forced process exit"))
    if phase == "normal":
        entry.release_engine("project")
    else:
        with pytest.raises(KeyboardInterrupt):
            entry.release_engine("project")
    assert calls == ["cleanup", "unlock"]

