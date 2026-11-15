"""Current-source reload against real native components and public callbacks."""

import importlib
import sys
import textwrap

import pytest

from infernux.components.missing_script import MissingScript
from infernux.components.registry import unregister_component_script
from infernux.engine.project_context import get_project_root, set_project_root
from infernux.engine.reload.coordinator import ReloadCoordinator
from infernux.engine.reload.components import ComponentAdapter
from infernux.runtime_services import _dispatch_updates, _update_snapshot


@pytest.fixture
def reload_project(tmp_path, scene):
    root = tmp_path / "Project"
    assets = root / "Assets"
    assets.mkdir(parents=True)
    previous = get_project_root()
    set_project_root(str(root))
    coordinator = ReloadCoordinator(str(root), adapter=ComponentAdapter(lambda: (scene,)))

    def write(name, source):
        path = assets / (name + ".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
        return str(path.resolve())

    yield scene, coordinator, write
    coordinator.close()
    for path in assets.glob("*.py"):
        unregister_component_script(str(path))
        sys.modules.pop(path.stem, None)
    set_project_root(previous)


def test_package_initializer_is_executed_and_reloaded_with_child(reload_project):
    _scene, coordinator, write = reload_project
    initializer = write("ReloadPackage/__init__", "from .child import value\nanswer = value + 1\n")
    child = write("ReloadPackage/child", "value = 3\n")
    try:
        coordinator.ensure_ready((child,))
        package = importlib.import_module("ReloadPackage")
        assert package.answer == 4
        assert package.child is sys.modules["ReloadPackage.child"]
        write("ReloadPackage/child", "value = 8\n")
        coordinator.ensure_ready((child,))
        assert sys.modules["ReloadPackage"].answer == 9
        assert sys.modules["ReloadPackage"] is not package
        assert coordinator.states[initializer].status == "active"
    finally:
        sys.modules.pop("ReloadPackage", None)
        sys.modules.pop("ReloadPackage.child", None)


def test_preload_owned_public_library_failure_stops_service_and_repairs(reload_project):
    from infernux.plugins.preload import PreloadManager, _ReloadPreloads
    from infernux.engine.path_utils import path_key

    _scene, coordinator, write = reload_project
    helper = write("PublicReloadLibrary", "value = 4\n")
    lifecycle = write("ReloadLifecycle", """
        from pathlib import Path
        from infernux.lifecycle import InxPreload
        class Service(InxPreload):
            def preload(self, context):
                import PublicReloadLibrary
                self.value = PublicReloadLibrary.value
                self.running = True
                context.add_cleanup(lambda: setattr(self, 'running', False))
    """
    )
    manager = PreloadManager(coordinator.graph.project_root)
    adapter = _ReloadPreloads(manager)
    coordinator.adapter.add(adapter)
    try:
        coordinator.ensure_ready((lifecycle,))
        previous = next(iter(manager.states.values())).instance
        assert previous.value == 4 and previous.running
        write("PublicReloadLibrary", "value = (\n")
        coordinator.request((helper,))
        assert not previous.running
        coordinator.compile_pending()
        coordinator.drain()
        assert coordinator.states[lifecycle].status == "failed"
        assert path_key(lifecycle) in manager.failures
        with pytest.raises(ImportError, match="unavailable"):
            importlib.import_module("PublicReloadLibrary")
        write("PublicReloadLibrary", "value = 9\n")
        coordinator.ensure_ready((helper, lifecycle))
        current = next(iter(manager.states.values())).instance
        assert current is not previous and current.value == 9
        assert current.running
    finally:
        coordinator.adapter.remove(adapter)
        sys.modules.pop("PublicReloadLibrary", None)
        sys.modules.pop("ReloadLifecycle", None)


HELPER = """
from infernux.runtime_services import Signal
changed = Signal()
step = 1
"""

PROBE = """
import infernux as inx
from ReloadHelper import changed, step
from infernux.runtime_services import register_update

class ReloadProbe(inx.InxComponent):
    count: int = inx.serialized_field(default=7)

    def awake(self):
        changed.add_listener(self.on_changed)
        register_update(self.tick, phase="service_update")

    def on_changed(self):
        self.count += step

    def tick(self, dt):
        self.count += step
"""


def _start(reload_project):
    scene, coordinator, write = reload_project
    helper = write("ReloadHelper", HELPER)
    script = write("ReloadProbe", PROBE)
    coordinator.ensure_ready((helper, script))
    obj = scene.create_game_object("重载探针")
    component = importlib.import_module("ReloadProbe").ReloadProbe()
    component._script_path = script
    obj.add_py_component(component)
    component._call_awake()
    return coordinator, write, helper, script, obj, component


def test_helper_failure_retires_old_instance_signal_and_updates_then_reconstructs(reload_project):
    coordinator, write, helper, script, obj, original = _start(reload_project)
    old_signal = sys.modules["ReloadHelper"].changed
    old_signal.emit()
    assert original.count == 8
    identity = original.component_id

    write("ReloadHelper", "def invalid(:\n")
    coordinator.request((helper,))
    assert isinstance(obj.get_py_components()[0], MissingScript)
    assert old_signal.emit() == ()
    _dispatch_updates(_update_snapshot()["service_update"], 0.1, 0.1, gameplay_phase=False)
    assert obj.get_py_components()[0]._reload_state["fields"]["count"] == 8
    assert not _update_snapshot()["service_update"]
    coordinator.compile_pending()
    coordinator.drain()
    assert coordinator.states[script].status == "failed"
    with pytest.raises(ImportError, match="unavailable"):
        importlib.import_module("ReloadHelper")

    write("ReloadHelper", HELPER.replace("step = 1", "step = 3"))
    coordinator.ensure_ready((helper, script))
    current = obj.get_py_components()[0]
    assert not isinstance(current, MissingScript)
    assert current is not original and current.component_id == identity
    assert current.count == 8
    sys.modules["ReloadHelper"].changed.emit()
    assert current.count == 11
    old_signal.emit()
    assert current.count == 11
    _dispatch_updates(_update_snapshot()["service_update"], 0.1, 0.1, gameplay_phase=False)
    assert current.count == 14


def test_failed_current_initialization_cleans_new_registrations_and_repairs(reload_project):
    coordinator, write, helper, script, obj, original = _start(reload_project)
    original.count = 23
    broken = PROBE.replace('register_update(self.tick, phase="service_update")',
                           'register_update(self.tick, phase="service_update")\n        raise RuntimeError("bad awake")')
    write("ReloadProbe", broken)
    with pytest.raises(RuntimeError, match="not ready"):
        coordinator.ensure_ready((script,))
    assert isinstance(obj.get_py_components()[0], MissingScript)
    assert not _update_snapshot()["service_update"]
    write("ReloadProbe", PROBE)
    coordinator.ensure_ready((script,))
    current = obj.get_py_components()[0]
    assert current.count == 23
    sys.modules["ReloadHelper"].changed.emit()
    assert current.count == 24


def test_bad_source_does_not_stop_an_unrelated_module(reload_project):
    _scene, coordinator, write = reload_project
    broken = write("BrokenReload", "def bad(:")
    healthy = write("HealthyReload", "value = 31\n")
    coordinator.request((broken, healthy))
    coordinator.compile_pending()
    coordinator.drain()
    assert coordinator.states[broken].status == "failed"
    assert coordinator.states[healthy].status == "active"
    assert sys.modules["HealthyReload"].value == 31


def test_latest_snapshot_and_circular_imports_use_current_bytes(reload_project):
    _scene, coordinator, write = reload_project
    first = write("CycleReloadA", "import CycleReloadB\nvalue = 1\n")
    second = write("CycleReloadB", "import CycleReloadA\nvalue = 2\n")
    coordinator.request((first, second))
    coordinator.compile_pending()
    write("CycleReloadA", "import CycleReloadB\nvalue = 9\n")
    coordinator.request((first,))
    coordinator.compile_pending()
    coordinator.drain()
    assert sys.modules["CycleReloadA"].value == 9
    assert sys.modules["CycleReloadB"].CycleReloadA is sys.modules["CycleReloadA"]


def test_preflight_reads_saved_helper_even_without_a_watcher_event(reload_project):
    coordinator, write, helper, script, obj, _original = _start(reload_project)
    write("ReloadHelper", HELPER.replace("step = 1", "step = 5"))
    coordinator.ensure_ready((script,))
    current = obj.get_py_components()[0]
    sys.modules["ReloadHelper"].changed.emit()
    assert current.count == 12


def test_deleted_dependency_stops_callbacks_and_repair_restores_declared_data(reload_project):
    from pathlib import Path
    coordinator, write, helper, script, obj, original = _start(reload_project)
    original.count = 42
    identity = original.component_id
    signal = sys.modules["ReloadHelper"].changed
    Path(helper).unlink()
    coordinator.request((), removed=(helper,))
    coordinator.compile_pending()
    coordinator.drain()
    assert isinstance(obj.get_py_components()[0], MissingScript)
    assert signal.emit() == ()
    assert not _update_snapshot()["service_update"]
    write("ReloadHelper", HELPER)
    coordinator.ensure_ready((helper, script))
    current = obj.get_py_components()[0]
    assert current.component_id == identity and current.count == 42
    assert current is not original


def test_deferred_old_callback_is_discarded_and_destroy_cleans_only_its_instance(reload_project):
    from infernux.engine.runtime_event_queue import enqueue, drain
    coordinator, write, helper, script, obj, original = _start(reload_project)
    other_object = obj.scene.create_game_object("另一个探针")
    other = type(original)()
    other._script_path = script
    other_object.add_py_component(other)
    other._call_awake()
    assert len(_update_snapshot()["service_update"]) == 2
    assert obj.remove_py_component(original)
    assert len(_update_snapshot()["service_update"]) == 1
    pending = []
    with other._reload_owner.enter():
        enqueue(lambda: pending.append("retired"))
    write("ReloadHelper", HELPER.replace("step = 1", "step = 4"))
    coordinator.ensure_ready((helper, script))
    drain()
    assert not pending
    current = other_object.get_py_components()[0]
    sys.modules["ReloadHelper"].changed.emit()
    assert current.count == 11
