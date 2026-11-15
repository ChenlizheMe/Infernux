"""Adversarial reload scenarios through the real native scene and public APIs."""
import importlib
import os
import py_compile
import sys
import textwrap
import time
from pathlib import Path

import pytest

from test_reload_coordinator import reload_project as reload_project
from infernux.components.missing_script import MissingScript
from infernux.components.registry import component_types_for_script_path
from infernux.runtime_services import (
    _dispatch_updates, _update_snapshot, get_runtime_service, remove_runtime_service,
)


@pytest.mark.parametrize("entry", ["update", "listener", "receiver", "deferred"])
def test_work_created_during_callback_inherits_source_lifetime(reload_project, entry):
    _scene, coordinator, write = reload_project
    path = write("RuntimeDerivedOwner", textwrap.dedent('''
        from infernux.execution import ExecutionScope
        from infernux.runtime_services import register_update, install_runtime_service, Signal
        from infernux.engine.runtime_event_queue import enqueue
        signal = Signal()
        scopes = []
        class Service:
            def marker(self): return "current-service"
        def callback(*args):
            scopes.append(ExecutionScope())
            install_runtime_service("derived-owner-test", Service())
    ''') + {
        "update": '\nregistration = register_update(callback, phase="service_update")\n',
        "listener": '\nsignal.add_listener(callback)\n',
        "receiver": '\nsignal._add_receiver(callback)\n',
        "deferred": '\nenqueue(callback)\n',
    }[entry])
    coordinator.ensure_ready((path,))
    module = sys.modules["RuntimeDerivedOwner"]
    try:
        if entry == "update":
            _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        elif entry == "deferred":
            from infernux.engine.runtime_event_queue import drain
            drain()
        else:
            module.signal.emit()
        assert get_runtime_service("derived-owner-test").marker() == "current-service"
        assert module.scopes[0].state == "active"
        write("RuntimeDerivedOwner", "def invalid(:\n")
        coordinator.request((path,))
        assert get_runtime_service("derived-owner-test") is None
        assert module.scopes[0].state == "closed"
        assert module.signal.emit() == ()
        assert not module.signal._receivers
        assert not _update_snapshot()["service_update"]
    finally:
        remove_runtime_service("derived-owner-test")


@pytest.mark.parametrize("already_active", [False, True])
@pytest.mark.parametrize("import_expression", [
    "importlib.import_module(name, __package__)",
    "import_module(name, __package__)",
    "__import__(__package__ + name, fromlist=['value'])",
])
def test_runtime_dynamic_import_retires_consumer_and_recovers_from_current_source(
    reload_project, already_active, import_expression,
):
    _scene, coordinator, write = reload_project
    helper = write("RuntimeDynamicPkg/helper", "value = 1\n")
    consumer = write("RuntimeDynamicPkg/consumer", '''
        import importlib
        from importlib import import_module
        from infernux.runtime_services import register_update
        name = "." + "helper"
        cached = None
        readings = []
        def tick(dt):
            global cached
            if cached is None:
                cached = IMPORT_EXPRESSION
            readings.append(cached.value)
        registration = register_update(tick, phase="service_update")
    '''.replace("IMPORT_EXPRESSION", import_expression))
    try:
        if already_active:
            coordinator.ensure_ready((helper,))
        coordinator.ensure_ready((consumer,))
        previous = sys.modules["RuntimeDynamicPkg.consumer"]
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert previous.readings == [1]
        assert coordinator.graph.module_for_path(helper) is not None
        assert helper in coordinator.graph.dependency_paths((consumer,))
        write("RuntimeDynamicPkg/helper", "def invalid(:\n")
        affected = coordinator.request((helper,))
        assert consumer in affected
        coordinator.compile_pending()
        coordinator.drain()
        assert coordinator.states[consumer].status == "failed"
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert previous.readings == [1]
        assert not _update_snapshot()["service_update"]
        write("RuntimeDynamicPkg/helper", "value = 9\n")
        coordinator.ensure_ready((helper,))
        current = sys.modules["RuntimeDynamicPkg.consumer"]
        assert current is not previous
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert current.readings == [9] and previous.readings == [1]
    finally:
        for name in ("RuntimeDynamicPkg.consumer", "RuntimeDynamicPkg.helper", "RuntimeDynamicPkg"):
            sys.modules.pop(name, None)


def test_first_runtime_import_ignores_same_size_same_timestamp_pyc(reload_project):
    _scene, coordinator, write = reload_project
    helper = Path(write("RuntimeFreshPkg/helper", "value = 1\n"))
    stamp = helper.stat().st_mtime_ns
    py_compile.compile(str(helper), doraise=True)
    helper.write_text("value = 8\n", encoding="utf-8")
    os.utime(helper, ns=(stamp, stamp))
    consumer = write("RuntimeFreshPkg/consumer", '''
        import importlib
        from infernux.runtime_services import register_update
        values = []
        def tick(dt):
            values.append(importlib.import_module('.helper', __package__).value)
        registration = register_update(tick, phase="service_update")
    ''')
    try:
        coordinator.ensure_ready((consumer,))
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert sys.modules["RuntimeFreshPkg.consumer"].values == [8]
    finally:
        for name in ("RuntimeFreshPkg.consumer", "RuntimeFreshPkg.helper", "RuntimeFreshPkg"):
            sys.modules.pop(name, None)


def test_service_lookup_records_cached_consumer_dependency(reload_project):
    _scene, coordinator, write = reload_project
    provider = write("RuntimeServiceProvider", '''
        from infernux.runtime_services import install_runtime_service
        class Service:
            def marker(self): return 1
        install_runtime_service("cached-provider-test", Service())
    ''')
    consumer = write("RuntimeServiceConsumer", '''
        from infernux.runtime_services import get_runtime_service, register_update
        cached = None
        calls = []
        def tick(dt):
            global cached
            if cached is None:
                cached = get_runtime_service("cached-provider-test")
            calls.append(cached.marker())
        registration = register_update(tick, phase="service_update")
    ''')
    try:
        coordinator.ensure_ready((provider, consumer))
        previous = sys.modules["RuntimeServiceConsumer"]
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert previous.calls == [1]
        write("RuntimeServiceProvider", "def invalid(:\n")
        assert consumer in coordinator.request((provider,))
        assert get_runtime_service("cached-provider-test") is None
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert previous.calls == [1]
    finally:
        remove_runtime_service("cached-provider-test")


def test_retired_raw_callback_cannot_register_itself_again(reload_project):
    _scene, coordinator, write = reload_project
    path = write("RuntimeRetiredRegistration", "def callback(dt): return None\n")
    coordinator.ensure_ready((path,))
    previous = sys.modules["RuntimeRetiredRegistration"].callback
    write("RuntimeRetiredRegistration", "def invalid(:\n")
    coordinator.request((path,))
    from infernux.runtime_services import register_update
    with pytest.raises(ReferenceError, match="retired"):
        register_update(previous, phase="service_update")
    assert not _update_snapshot()["service_update"]


def test_failed_initialization_cannot_be_instantiated_from_component_registry(reload_project):
    scene, coordinator, write = reload_project
    good = '''
        from infernux.components import InxComponent
        class RuntimeCandidate(InxComponent):
            value: int = 42
            def awake(self): pass
    '''
    path = write("RuntimeCandidate", good)
    coordinator.ensure_ready((path,))
    obj = scene.create_game_object("Candidate registry boundary")
    original = sys.modules["RuntimeCandidate"].RuntimeCandidate()
    original._script_path = path
    obj.add_py_component(original)
    original._call_awake()
    write("RuntimeCandidate", good.replace("pass", "raise RuntimeError('current awake failed')"))
    with pytest.raises(RuntimeError, match="not ready"):
        coordinator.ensure_ready((path,))
    assert isinstance(obj.get_py_components()[0], MissingScript)
    assert component_types_for_script_path(path) == ()
    from infernux.components.script_loader import load_and_create_component
    assert load_and_create_component(path, script_guid=original._script_guid) is None
    assert component_types_for_script_path(path) == ()
    assert "RuntimeCandidate" not in sys.modules


def test_twenty_consecutive_edits_retire_coroutines_and_preserve_declared_fields(reload_project):
    scene, coordinator, write = reload_project
    source = '''
        from infernux.components import InxComponent
        from infernux.runtime_services import register_update
        class RuntimeRepeated(InxComponent):
            count: int = 7
            def awake(self):
                self.start_coroutine(self.flow())
                register_update(self.tick, phase="service_update")
            def flow(self):
                while True:
                    yield None
            def tick(self, dt): self.count += INCREMENT
    '''
    path = write("RuntimeRepeated", source.replace("INCREMENT", "1"))
    coordinator.ensure_ready((path,))
    obj = scene.create_game_object("Continuous edits")
    original = sys.modules["RuntimeRepeated"].RuntimeRepeated()
    original._script_path = path
    obj.add_py_component(original)
    original._call_awake()
    identity = original.component_id
    count = original.count
    for increment in range(1, 21):
        previous = obj.get_py_components()[0]
        old_coroutines = previous._coroutine_scheduler
        write("RuntimeRepeated", "def invalid(:\n")
        coordinator.request((path,))
        coordinator.compile_pending()
        coordinator.drain()
        assert isinstance(obj.get_py_components()[0], MissingScript)
        assert old_coroutines.count == 0
        assert not _update_snapshot()["service_update"]
        write("RuntimeRepeated", source.replace("INCREMENT", str(increment)))
        coordinator.ensure_ready((path,))
        current = obj.get_py_components()[0]
        assert current is not previous and current.component_id == identity
        assert current.count == count
        assert current._coroutine_scheduler.count == 1
        assert len(_update_snapshot()["service_update"]) == 1
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        count += increment
        assert current.count == count


def test_explicit_engine_scope_uses_same_loader_and_restart_scope_never_executes(reload_project, tmp_path):
    _scene, coordinator, _write = reload_project
    source = tmp_path / "actual_engine_feature.py"
    source.write_text("value = 1\n", encoding="utf-8")
    core = tmp_path / "actual_engine_core.py"
    core.write_text("value = 3\n", encoding="utf-8")
    coordinator.register_source_scope(str(source), "ReviewEngineFeature")
    coordinator.ensure_ready((str(source),))
    previous = sys.modules["ReviewEngineFeature"]
    source.write_text("value = 2\n", encoding="utf-8")
    coordinator.ensure_ready((str(source),))
    assert sys.modules["ReviewEngineFeature"].value == 2
    assert sys.modules["ReviewEngineFeature"] is not previous
    spec = importlib.util.spec_from_file_location("ReviewEngineCore", core)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    spec.loader.exec_module(module)
    owner = coordinator.register_source_scope(str(core), module.__name__, restart_only=True, loaded_module=module)
    try:
        core.write_text("raise RuntimeError('must never execute during reload')\n", encoding="utf-8")
        coordinator.request((str(core),))
        coordinator.compile_pending()
        coordinator.drain()
        assert owner.closed
        assert coordinator.states[str(core)].status == "restart_required"
        assert sys.modules[module.__name__] is module and module.value == 3
        assert not coordinator.pending_count
    finally:
        sys.modules.pop("ReviewEngineFeature", None)
        sys.modules.pop("ReviewEngineCore", None)


def test_plugin_disable_enable_uninstall_uses_real_resources_manager(tmp_path, engine, monkeypatch):
    from infernux.engine.project_context import get_project_root, set_project_root
    from infernux.engine.resources_manager import ResourcesManager
    from infernux.plugins import PluginManager
    from test_inxpackage_plugins import _source, _project, _export

    source = _source(tmp_path / "source", "review/resource-retirement")
    runtime = source / "runtime"
    runtime.mkdir()
    (runtime / "service.py").write_text(
        "from infernux.runtime_services import install_runtime_service\n"
        "class Service:\n    def marker(self): return 'installed'\n"
        "install_runtime_service('real-retirement-test', Service())\n",
        encoding="utf-8",
    )
    package = _export(source, tmp_path / "retirement.inxpkg")
    project = _project(tmp_path / "project")
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    previous = get_project_root()
    set_project_root(str(project))
    installer = PluginManager(str(project), installation_only=True)
    resources = None
    manager = None
    try:
        installer.install_package(str(package), install_dependencies=False)
        # The actual ResourceChangeHandler shares the real native database.
        # No replacement ResourcesManager.instance() or coordinator stand-in.
        resources = ResourcesManager(str(project), engine)
        manager = PluginManager(str(project), runtime=True)
        assert manager.reload("review/resource-retirement").loaded
        first = get_runtime_service("real-retirement-test")
        assert first.marker() == "installed"
        manager.set_enabled("review/resource-retirement", False)
        assert get_runtime_service("real-retirement-test") is None
        assert manager.set_enabled("review/resource-retirement", True).loaded
        assert get_runtime_service("real-retirement-test") is not first
        manager.uninstall("review/resource-retirement")
        assert get_runtime_service("real-retirement-test") is None
        assert manager.registry.installed_record("review/resource-retirement") is None
    finally:
        if manager is not None:
            manager.shutdown()
        if resources is not None:
            resources.cleanup()
        installer.shutdown()
        remove_runtime_service("real-retirement-test")
        set_project_root(previous)


def test_owned_child_process_library_save_stops_old_process_and_repair_starts_current(
    reload_project, tmp_path, monkeypatch,
):
    from infernux.plugins import PluginManager
    from infernux.plugins.preload import PreloadManager, _ReloadPreloads
    from test_inxpackage_plugins import _source, _export

    _scene, coordinator, _write = reload_project
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("RELOAD_REVIEW_PARENT_PID", str(os.getpid()))
    source = _source(tmp_path / "source", "review/child-library")
    library = source / "editor/public_service"
    library.mkdir(parents=True)
    worker_source = '''
import os
from pathlib import Path
import time
if os.getpid() == int(os.environ['RELOAD_REVIEW_PARENT_PID']):
    raise RuntimeError('Child-only library was executed inside the Editor')
Path(__file__).with_suffix('.ready').write_text(str(os.getpid()) + ':VERSION', encoding='utf-8')
while True:
    time.sleep(.02)
'''
    (library / "worker.py").write_text(worker_source.replace("VERSION", "v1"), encoding="utf-8")
    (source / "editor/startup.py").write_text('''
import os
import subprocess
import sys
from pathlib import Path
from infernux.lifecycle import InxPreload
class ChildService(InxPreload):
    def preload(self, context):
        library = context.own_python_library('editor/public_service')
        self.process = subprocess.Popen([sys.executable, str(Path(library) / 'worker.py')],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        context.add_cleanup(self.stop)
    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
''', encoding="utf-8")
    package = _export(source, tmp_path / "child.inxpkg")
    installer = PluginManager(coordinator.graph.project_root, installation_only=True)
    installer.install_package(str(package), install_dependencies=False)
    preloads = PreloadManager(coordinator.graph.project_root)
    adapter = _ReloadPreloads(preloads)
    coordinator.adapter.add(adapter)
    lifecycle = str(Path(coordinator.graph.project_root) / "Packages/review/child-library/editor/startup.py")
    worker = Path(coordinator.graph.project_root) / "Packages/review/child-library/editor/public_service/worker.py"
    ready = worker.with_suffix(".ready")

    def wait_ready(process, version):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            assert process.poll() is None
            if ready.is_file() and ready.read_text(encoding="utf-8") == f"{process.pid}:{version}":
                return
            time.sleep(.01)
        pytest.fail("The owned child process did not load its current library")

    processes = []
    try:
        coordinator.ensure_ready((lifecycle,))
        first = next(iter(preloads.states.values())).instance.process
        processes.append(first)
        wait_ready(first, "v1")
        worker.write_text("def invalid(:\n", encoding="utf-8")
        assert lifecycle in coordinator.request((str(worker),))
        assert first.poll() is not None
        coordinator.compile_pending()
        coordinator.drain()
        assert coordinator.states[lifecycle].status == "failed"
        assert coordinator.states[str(worker)].status == "failed"
        assert not preloads.states
        for _ in range(3):
            assert coordinator.drain() == 0
        worker.write_text(worker_source.replace("VERSION", "v2"), encoding="utf-8")
        coordinator.ensure_ready((str(worker),))
        current = next(iter(preloads.states.values())).instance.process
        processes.append(current)
        assert current.pid != first.pid
        wait_ready(current, "v2")
        assert all(getattr(module, "__file__", "") != str(worker) for module in sys.modules.values())
    finally:
        preloads.unload_all()
        coordinator.adapter.remove(adapter)
        installer.shutdown()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        for name, module in tuple(sys.modules.items()):
            if str(getattr(module, "__file__", "")).startswith(coordinator.graph.project_root):
                sys.modules.pop(name, None)


def test_panel_enable_paint_close_and_reload_share_owned_lifetime(reload_project, engine):
    from infernux.renderstack.render_stack_pipeline import RenderStackPipeline

    _scene, coordinator, write = reload_project
    path = write("RuntimeOwnedPanel", '''
        from infernux.engine.ui.editor_panel import EditorPanel
        from infernux.runtime_services import install_runtime_service
        class OwnedPanel(EditorPanel):
            def __init__(self):
                super().__init__('Owned lifecycle', 'review.owned_lifecycle')
                self.renders = 0
                self.enables = 0
                self.disables = 0
                self.paint_service = object()
            def _initial_size(self): return 64., 64.
            def _pre_render(self, ctx):
                ctx.set_next_window_pos(0., 0., 0, 0., 0.)
            def on_enable(self):
                self.enables += 1
                install_runtime_service('owned-panel-enable', self)
            def on_render_content(self, ctx):
                self.renders += 1
                install_runtime_service('owned-panel-paint', self.paint_service)
                ctx.label('Current paint')
            def on_disable(self):
                self.disables += 1
                install_runtime_service('owned-panel-cleanup', self)
    ''')
    coordinator.ensure_ready((path,))
    panel = sys.modules["RuntimeOwnedPanel"].OwnedPanel()
    pipeline = RenderStackPipeline()
    engine.register_gui_renderable(panel.window_id, panel)
    engine.set_render_pipeline(pipeline)
    stage = 0
    observations = {}
    deadline = time.monotonic() + 15

    def update(_delta):
        nonlocal stage
        if time.monotonic() > deadline:
            engine.exit()
            return
        if stage == 0 and panel.renders >= 3:
            observations["first_owner"] = panel._reload_owner
            panel._is_open = False
            panel._finalize_close_lifecycle()
            observations["closed_services"] = (
                get_runtime_service('owned-panel-enable'), get_runtime_service('owned-panel-paint'),
                get_runtime_service('owned-panel-cleanup'),
            )
            panel._is_open = True
            stage = 1
        elif stage == 1 and panel.enables == 2 and panel.renders >= 6:
            observations["second_owner"] = panel._reload_owner
            observations["renders_before_failure"] = panel.renders
            write("RuntimeOwnedPanel", "def invalid(:\n")
            coordinator.request((path,))
            coordinator.compile_pending()
            coordinator.drain()
            observations["failure_frame"] = engine.renderer_ui_performance_snapshot["panel_times"][panel.window_id]["sample_count"]
            stage = 2
        elif stage == 2:
            samples = engine.renderer_ui_performance_snapshot["panel_times"][panel.window_id]["sample_count"]
            if samples >= observations["failure_frame"] + 5:
                stage = 3
                engine.exit()

    try:
        engine.set_pre_scene_update_callback(update)
        engine.run()
        assert stage == 3
        assert observations["first_owner"].closed
        assert observations["second_owner"] is not observations["first_owner"]
        assert observations["second_owner"].closed
        assert observations["closed_services"] == (None, None, None)
        assert panel.renders == observations["renders_before_failure"]
        assert get_runtime_service('owned-panel-enable') is None
        assert get_runtime_service('owned-panel-paint') is None
        assert get_runtime_service('owned-panel-cleanup') is None
        assert panel.disables == 2
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable(panel.window_id)
        engine.set_render_pipeline(None)
        pipeline.dispose()
        panel._finalize_close_lifecycle()
        for name in ('owned-panel-enable', 'owned-panel-paint', 'owned-panel-cleanup'):
            remove_runtime_service(name)


@pytest.mark.skipif(sys.platform != "win32", reason="Exercises a real Windows file sharing violation")
def test_locked_package_initializer_save_stops_old_callbacks_and_repairs(reload_project):
    import ctypes
    from ctypes import wintypes

    _scene, coordinator, write = reload_project
    initializer = write("RuntimeLockedPkg/__init__", "marker = 1\n")
    child_source = textwrap.dedent('''
        from infernux.runtime_services import register_update
        calls = []
        def tick(dt): calls.append(1)
        registration = register_update(tick, phase="service_update")
    ''')
    child = write("RuntimeLockedPkg/child", child_source)
    coordinator.ensure_ready((child,))
    previous = sys.modules["RuntimeLockedPkg.child"]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handle = create(initializer, 0x80000000, 6, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    try:
        write("RuntimeLockedPkg/child", child_source + "\nrevision = 2\n")
        coordinator.request((child,))
        assert not _update_snapshot()["service_update"]
        coordinator.compile_pending()
        coordinator.drain()
        assert coordinator.states[initializer].status == "failed"
        assert coordinator.states[child].status == "failed"
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert previous.calls == []
    finally:
        close(handle)
    try:
        coordinator.ensure_ready((child,))
        current = sys.modules["RuntimeLockedPkg.child"]
        assert current is not previous and current.revision == 2
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert current.calls == [1] and previous.calls == []
    finally:
        for name in ("RuntimeLockedPkg.child", "RuntimeLockedPkg"):
            sys.modules.pop(name, None)


def test_colliding_module_identity_stops_affected_old_source_before_reporting_failure(reload_project):
    _scene, coordinator, write = reload_project
    original = write("RuntimeCollision/__init__", '''
        from infernux.runtime_services import register_update
        calls = []
        def tick(dt): calls.append(1)
        registration = register_update(tick, phase="service_update")
    ''')
    coordinator.ensure_ready((original,))
    module = sys.modules["RuntimeCollision"]
    collision = write("RuntimeCollision", "value = 2\n")
    try:
        coordinator.request((collision,))
        assert coordinator.states[collision].status == "failed"
        assert coordinator.states[original].status == "failed"
        assert "collision" in coordinator.states[collision].error
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert module.calls == []
        assert not _update_snapshot()["service_update"]
    finally:
        sys.modules.pop("RuntimeCollision", None)


def test_player_bytecode_preload_owns_public_operations_and_retires_on_unload(reload_project, monkeypatch):
    from infernux.host import OperationRegistry, OperationKind
    from infernux.host.operations import OperationError
    from infernux.plugins.preload import PreloadManager

    _scene, coordinator, write = reload_project
    source = Path(write("RuntimeCompiledOwner", '''
        from infernux.lifecycle import InxPreload
        from infernux.host import OperationRegistry, OperationKind
        from infernux.host.operation_support import operation
        from infernux.runtime_services import install_runtime_service
        class CompiledOwner(InxPreload):
            def preload(self, context):
                install_runtime_service('compiled-owner-test', self)
                OperationRegistry.instance().register(operation(
                    'review.compiled.read', OperationKind.QUERY, 'Read current bytecode',
                    lambda: 42, capability='review.read', thread='any'))
    '''))
    bytecode = source.with_suffix(".pyc")
    py_compile.compile(str(source), cfile=str(bytecode), doraise=True)
    source.unlink()
    registry = OperationRegistry()
    monkeypatch.setattr(OperationRegistry, "_instance", registry)
    manager = PreloadManager(coordinator.graph.project_root, runtime=True)
    try:
        states = manager._load_path(str(bytecode), {"CompiledOwner"})
        assert len(states) == 1 and states[0].loaded and states[0].reload_owner.active
        assert registry.get("review.compiled.read") is not None
        assert get_runtime_service('compiled-owner-test') is states[0].instance
        manager.unload_all()
        with pytest.raises(OperationError) as error:
            registry.get("review.compiled.read")
        assert error.value.code == "operation.not_found"
        assert get_runtime_service('compiled-owner-test') is None
        assert not manager.states
        assert "RuntimeCompiledOwner" not in sys.modules
    finally:
        manager.close()
        remove_runtime_service('compiled-owner-test')


def test_pipeline_discovery_never_lists_a_retired_module_or_restores_its_namespace(reload_project):
    from infernux.renderstack.discovery import discover_pipelines, invalidate_discovery_cache
    _scene, coordinator, write = reload_project
    source = '''
        from infernux.renderstack.render_pipeline import RenderPipeline
        class CatalogCurrent(RenderPipeline):
            name = 'Review Owned Catalog'
            def define_topology(self, graph): pass
    '''
    path = write("RuntimePipelineCatalog", source)
    try:
        coordinator.ensure_ready((path,))
        previous = discover_pipelines()['Review Owned Catalog']
        assert previous is sys.modules['RuntimePipelineCatalog'].CatalogCurrent
        write("RuntimePipelineCatalog", "def invalid(:\n")
        coordinator.request((path,))
        coordinator.compile_pending()
        coordinator.drain()
        assert 'Review Owned Catalog' not in discover_pipelines()
        assert 'RuntimePipelineCatalog' not in sys.modules
        assert coordinator.pending_count == 0
        write("RuntimePipelineCatalog", source)
        coordinator.ensure_ready((path,))
        assert discover_pipelines()['Review Owned Catalog'] is not previous
    finally:
        invalidate_discovery_cache(path)


def test_command_handlers_predicates_and_retirement_own_derived_services(reload_project):
    from infernux.engine.interaction.commands import EditorCommandRegistry
    _scene, coordinator, write = reload_project
    previous_registry = EditorCommandRegistry._instance
    registry = EditorCommandRegistry()
    path = write("RuntimeOwnedCommand", '''
        from infernux.engine.interaction.commands import EditorCommand, EditorCommandRegistry
        from infernux.runtime_services import install_runtime_service
        values = {name: object() for name in ('can', 'checked', 'execute', 'reason')}
        allowed = True
        def can_execute(ctx):
            install_runtime_service('command-can', values['can'])
            return allowed
        def checked(ctx):
            install_runtime_service('command-checked', values['checked'])
            return True
        def execute(ctx):
            install_runtime_service('command-execute', values['execute'])
            return 42
        def reason(ctx):
            install_runtime_service('command-reason', values['reason'])
            return 'disabled by current state'
        command = EditorCommand('review.owned.command', execute, can_execute=can_execute,
            is_checked=checked, disabled_reason=reason, creates_user_action=False)
        EditorCommandRegistry.instance().register(command)
    ''')
    try:
        coordinator.ensure_ready((path,))
        assert registry.can_execute('review.owned.command')
        assert registry.is_checked('review.owned.command')
        assert registry.execute('review.owned.command').value == 42
        sys.modules['RuntimeOwnedCommand'].allowed = False
        assert registry.disabled_reason('review.owned.command') == 'disabled by current state'
        assert all(get_runtime_service('command-' + name) is not None
                   for name in ('can', 'checked', 'execute', 'reason'))
        write("RuntimeOwnedCommand", "def invalid(:\n")
        coordinator.request((path,))
        assert registry.get('review.owned.command') is None
        assert all(get_runtime_service('command-' + name) is None
                   for name in ('can', 'checked', 'execute', 'reason'))
    finally:
        registry.clear()
        EditorCommandRegistry._instance = previous_registry
        for name in ('can', 'checked', 'execute', 'reason'):
            remove_runtime_service('command-' + name)


def test_handle_provider_draw_gesture_and_retirement_own_derived_services(reload_project):
    from infernux.engine.interaction.handles import EditorHandleRegistry
    _scene, coordinator, write = reload_project
    previous_registry = EditorHandleRegistry._instance
    registry = EditorHandleRegistry()
    path = write("RuntimeOwnedHandle", '''
        from infernux.engine.interaction.handles import register_handle_provider
        from infernux.runtime_services import install_runtime_service
        values = {name: object() for name in ('enabled', 'draw', 'begin', 'change', 'commit', 'cancel', 'retire')}
        def own(name): install_runtime_service('handle-' + name, values[name])
        def enabled():
            own('enabled')
            return True
        def changed(value): own('change')
        def begin(value): own('begin')
        def commit(value): own('commit')
        def cancel(value): own('cancel')
        def retire(): own('retire')
        def draw(context):
            own('draw')
            context.position('point', (0, 0, 0), changed, on_begin=begin, on_commit=commit, on_cancel=cancel)
        registration = register_handle_provider('review.owned.handle', draw, enabled=enabled, on_retire=retire)
    ''')
    try:
        coordinator.ensure_ready((path,))
        registry.collect(object())
        assert registry.begin_capture('review.owned.handle:point', (0, 0, 5), (0, 0, -1))
        registry.update_capture((1, 0, 5), (0, 0, -1))
        registry.finish_capture()
        assert all(get_runtime_service('handle-' + name) is not None
                   for name in ('enabled', 'draw', 'begin', 'change', 'commit'))
        assert registry.begin_capture('review.owned.handle:point', (0, 0, 5), (0, 0, -1))
        registry.update_capture((1, 0, 5), (0, 0, -1))
        write("RuntimeOwnedHandle", "def invalid(:\n")
        coordinator.request((path,))
        assert not registry.providers and not registry.frame_handles and not registry.active_capture_id
        assert all(get_runtime_service('handle-' + name) is None
                   for name in ('enabled', 'draw', 'begin', 'change', 'commit', 'cancel', 'retire'))
    finally:
        registry.clear()
        EditorHandleRegistry._instance = previous_registry
        for name in ('enabled', 'draw', 'begin', 'change', 'commit', 'cancel', 'retire'):
            remove_runtime_service('handle-' + name)


def test_preload_unload_all_does_not_retire_unrelated_gameplay_in_shared_coordinator(reload_project):
    from infernux.plugins.preload import PreloadManager, _ReloadPreloads
    _scene, coordinator, write = reload_project
    gameplay = write("RuntimeUnrelatedGame", '''
        from infernux.runtime_services import register_update
        calls = []
        def tick(dt): calls.append(1)
        registration = register_update(tick, phase='service_update')
    ''')
    preload = write("RuntimeRelatedPreload", '''
        from infernux.lifecycle import InxPreload
        class RelatedPreload(InxPreload):
            def preload(self, context): pass
    ''')
    manager = PreloadManager(coordinator.graph.project_root)
    adapter = _ReloadPreloads(manager)
    coordinator.adapter.add(adapter)
    manager._reload = coordinator
    manager._reload_adapter = adapter
    try:
        coordinator.ensure_ready((gameplay, preload))
        module = sys.modules['RuntimeUnrelatedGame']
        manager.unload_all()
        assert coordinator.states[gameplay].status == 'active'
        assert sys.modules['RuntimeUnrelatedGame'] is module
        _dispatch_updates(_update_snapshot()['service_update'], .01, .01, gameplay_phase=False)
        assert module.calls == [1]
        assert not manager.states
    finally:
        manager._reload = None
        manager._reload_adapter = None
        coordinator.adapter.remove(adapter)


def test_deleted_source_removes_actual_public_import_alias(reload_project):
    _scene, coordinator, write = reload_project
    path = write('RuntimePublicAliasSource', 'value = 42\n')
    coordinator.graph.set_module_name(path, 'ReviewPublicAlias')
    coordinator.ensure_ready((path,))
    assert sys.modules['ReviewPublicAlias'].value == 42
    Path(path).unlink()
    coordinator.request((), removed=(path,))
    coordinator.compile_pending()
    coordinator.drain()
    assert 'ReviewPublicAlias' not in sys.modules
    with pytest.raises(ImportError, match='unavailable'):
        importlib.import_module('ReviewPublicAlias')
