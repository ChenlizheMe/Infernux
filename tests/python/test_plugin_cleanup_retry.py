"""An unsuccessful unload retains ownership until every resource is released."""
import socket
import sys
from pathlib import Path

import pytest

from infernux.plugins import PluginManager
from infernux.plugins.preload import PreloadManager, PreloadState
from infernux.engine.project_context import is_project_component_script
from test_inxpackage_plugins import _export, _project, _source


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupted_cleanup_keeps_callback_and_does_not_repeat_completed_hook(tmp_path, interrupt):
    manager = PreloadManager(str(_project(tmp_path / "project")), runtime=True)
    calls = []
    interrupted = True

    class Service:
        def unload(self):
            calls.append("unload")

    def cleanup():
        calls.append("cleanup")
        if interrupted:
            raise interrupt()

    state = PreloadState("owner", "script", "type", "Service", "source", loaded=True, instance=Service())
    state.cleanup_callbacks.extend([lambda: calls.append("older"), cleanup, lambda: calls.append("newer")])
    manager.states[state.identity] = state
    with pytest.raises(interrupt):
        manager.unload_all()
    interrupted = False
    assert manager.unload_all() == ()
    assert calls == ["unload", "newer", "cleanup", "cleanup", "older"]
    assert manager.states == {}


@pytest.mark.parametrize("runtime", [False, True])
def test_failed_callbacks_keep_lifo_order_until_explicit_success(tmp_path, runtime):
    manager = PreloadManager(str(_project(tmp_path / "project")), runtime=runtime)
    state = PreloadState("owner", "script", "type", "Service", "source", loaded=True)
    calls = []
    blocked = True

    def cleanup(name):
        calls.append(name)
        if blocked and name in {"first", "last"}:
            raise RuntimeError(name + " still owns its resource")

    for name in ("first", "middle", "last"):
        state.cleanup_callbacks.append(lambda name=name: cleanup(name))
    manager.states[state.identity] = state
    for _ in range(2):
        assert manager.unload_all() == (state,)
        assert state.loaded and state.restart_required
        assert len(state.cleanup_callbacks) == 2
    blocked = False
    assert manager.unload_all() == ()
    assert calls == ["last", "middle", "first", "last", "first", "last", "first"]


@pytest.mark.parametrize("operation", ["disable", "uninstall", "reload"])
@pytest.mark.parametrize("partial", [False, True])
def test_package_operation_cannot_forget_a_live_cleanup_resource(tmp_path, monkeypatch, operation, partial):
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "package-cache"))
    source = _source(tmp_path / "source", "vendor/cleanup-retry")
    (source / "runtime").mkdir()
    (source / "runtime/library").mkdir()
    (source / "runtime/library/helper.py").write_text("VALUE = 41\n", encoding="utf-8")
    (source / "runtime/service.py").write_text(
        "import socket\n"
        "from pathlib import Path\n"
        "from infernux.lifecycle import InxPreload\n"
        "class Service(InxPreload):\n"
        "    def preload(self, context):\n"
        "        global resource_owner\n"
        "        resource_owner = self\n"
        "        self.root = Path(context.project_root)\n"
        "        context.own_python_library('runtime/library')\n"
        "        self.calls = []\n"
        "        self.resource = socket.socket()\n"
        "        self.resource.bind(('127.0.0.1', 0))\n"
        "        self.resource.listen()\n"
        "        context.add_cleanup(self.cleanup)\n"
        "        if (self.root / 'fail-start').exists():\n"
        "            raise RuntimeError('startup failed after resource acquisition')\n"
        "    def unload(self):\n"
        "        self.calls.append('unload')\n"
        "        if self.calls.count('unload') != 1:\n"
        "            raise RuntimeError('unload already completed')\n"
        "    def cleanup(self):\n"
        "        self.calls.append('cleanup')\n"
        "        if not (self.root / 'allow-stop').exists():\n"
        "            raise RuntimeError('socket still running')\n"
        "        self.resource.close()\n",
        encoding="utf-8",
    )
    package = _export(source, tmp_path / "cleanup.inxpkg")
    project = _project(tmp_path / "project")
    if partial:
        (project / "fail-start").touch()
    manager = PluginManager(str(project))
    reference = "vendor/cleanup-retry"
    manager.install_package(str(package), install_dependencies=False)
    state, = manager.preloads.states.values()
    instance = sys.modules[state.module_name].resource_owner
    address = instance.resource.getsockname()
    script = project / "Packages/vendor/cleanup-retry/runtime/service.py"
    registry = manager.registry.path
    registry_bytes = Path(registry).read_bytes()

    def act():
        if operation == "disable":
            return manager.set_enabled(reference, False)
        if operation == "uninstall":
            return manager.uninstall(reference)
        return manager.preloads.reload_package(reference)

    try:
        for _ in range(2):
            if operation == "reload":
                assert act() == (state,)
            else:
                with pytest.raises(RuntimeError, match="socket still running"):
                    act()
            assert state is manager.preloads.states[state.identity]
            assert state.loaded is not partial
            assert state.restart_required
            assert len(state.cleanup_callbacks) == 1
            assert script.is_file() and state.module_name in sys.modules
            assert not is_project_component_script(str(script.parent / "library/helper.py"), str(project))
            assert Path(registry).read_bytes() == registry_bytes
            with socket.socket() as contender:
                with pytest.raises(OSError):
                    contender.bind(address)
        (project / "allow-stop").touch()
        (project / "fail-start").unlink(missing_ok=True)
        act()
        assert instance.calls == (["cleanup"] * 4 if partial else ["unload", "cleanup", "cleanup", "cleanup"])
        assert instance.resource.fileno() == -1
        assert all(candidate is not state for candidate in manager.preloads.states.values())
        if operation == "reload":
            replacement, = manager.preloads.states.values()
            assert replacement.loaded and replacement.instance is not instance
        elif operation == "disable":
            assert not manager.registry.installed_record(reference)["enabled"]
        else:
            assert manager.registry.installed_record(reference) is None
            assert not script.exists()
    finally:
        (project / "allow-stop").touch()
        instance.resource.close()
        # Retire the deliberately single-shot test hook even on the old buggy
        # implementation, so one baseline failure cannot leak package modules.
        instance.unload = lambda: None
        manager.shutdown()
