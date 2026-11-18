"""Editor queues and theme listeners share source retirement and attribution."""
import sys
from pathlib import Path

import textwrap

import pytest

from infernux.components.registry import unregister_component_script
from infernux.engine.project_context import get_project_root, set_project_root
from infernux.engine.reload.components import ComponentAdapter
from infernux.engine.reload.coordinator import ReloadCoordinator
from infernux.engine.deferred_task import DeferredTaskRunner
from infernux.engine.ui import theme
from infernux.runtime_services import get_runtime_service, install_runtime_service, remove_runtime_service


@pytest.fixture
def reload_project(tmp_path, scene, engine, monkeypatch):
    from infernux.core.assets import AssetManager
    from infernux.lib import AssetRegistry

    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(AssetManager, "_registry", AssetRegistry.instance())
    monkeypatch.setattr(AssetManager, "_native_engine", classmethod(lambda _cls: engine))
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


def test_source_loader_does_not_become_a_runtime_consumer_of_every_project_module(reload_project):
    _scene, coordinator, write = reload_project
    loader = sys.modules["infernux.engine.reload.modules"]
    core_path = str(Path(loader.__file__).resolve())
    coordinator.register_source_scope(core_path, loader.__name__, restart_only=True, loaded_module=loader)
    path = write("EditorLoaderDependency", "value = 1\n")
    coordinator.ensure_ready((path,))
    write("EditorLoaderDependency", "value = 2\n")
    affected = coordinator.request((path,))
    assert core_path not in affected
    assert coordinator.states[core_path].status == "active"


def test_non_component_callbacks_capture_one_source_scope_that_retires(reload_project):
    from infernux.runtime_services import _dispatch_updates, _update_snapshot

    _scene, coordinator, write = reload_project
    path = write("EditorManagedSourceScope", '''
        from infernux.runtime_services import register_update
        from infernux.execution import capture_execution_scope
        scopes = []
        def tick(dt): scopes.append(capture_execution_scope())
        registration = register_update(tick, phase="service_update")
    ''')
    coordinator.ensure_ready((path,))
    module = sys.modules["EditorManagedSourceScope"]
    _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
    _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
    assert module.scopes[0] is not None and module.scopes[0] is module.scopes[1]
    assert module.scopes[0].state == "active"
    module.__infernux_reload_owner__.stop_admission()
    assert module.scopes[0].state == "paused"
    write("EditorManagedSourceScope", "def bad(:\n")
    coordinator.request((path,))
    assert module.scopes[0].state == "closed"


def test_public_programmatic_component_keeps_source_and_identity_for_retirement(reload_project):
    from infernux.components.missing_script import MissingScript

    scene, coordinator, write = reload_project
    path = write("EditorProgrammaticComponent", '''
        from infernux.components import InxComponent
        class EditorProgrammaticComponent(InxComponent): pass
    ''')
    coordinator.ensure_ready((path,))
    module = sys.modules["EditorProgrammaticComponent"]
    obj = scene.create_game_object("Programmatic component")
    component = obj.add_component(module.EditorProgrammaticComponent)
    assert component._script_path == path
    assert component._script_guid and type(component)._get_type_guid()
    identity = component.component_id
    write("EditorProgrammaticComponent", "def bad(:\n")
    coordinator.request((path,))
    missing = obj.get_py_components()[0]
    assert isinstance(missing, MissingScript)
    assert missing.component_id == identity and missing._script_guid == component._script_guid


@pytest.mark.skip(reason="050 Play prepares runtime Worlds; native SceneManager.play is not the entry point")
@pytest.mark.parametrize("failing_phase", ["awake", "start", "on_enable"])
def test_native_first_initialization_failure_closes_component_and_fails_current_source(reload_project, failing_phase):
    from infernux.components.missing_script import MissingScript
    from infernux.lib import SceneManager
    from infernux.runtime_services import _dispatch_updates, _update_snapshot

    scene, coordinator, write = reload_project
    path = write("EditorFirstInitialization", f'''
        from infernux.components import InxComponent
        from infernux.runtime_services import register_update
        calls = []
        class EditorFirstInitialization(InxComponent):
            def {failing_phase}(self):
                register_update(self.tick, phase="service_update")
                raise RuntimeError("current {failing_phase} failed")
            def tick(self, dt): calls.append("old callback")
            def update(self, dt): calls.append("old update")
        class EditorFirstInitializationPeer(InxComponent):
            def update(self, dt): calls.append("same source peer update")
    ''')
    coordinator.ensure_ready((path,))
    module = sys.modules["EditorFirstInitialization"]
    obj = scene.create_game_object("Failed native initialization")
    component = obj.add_component(module.EditorFirstInitialization)
    peer = obj.add_component(module.EditorFirstInitializationPeer)
    manager = SceneManager.instance()
    try:
        manager.play()
        assert not component.enabled and not component._has_started
        if failing_phase == "awake":
            assert not component._awake_called
        assert str(component._initialization_error) == f"current {failing_phase} failed"
        assert not peer._reload_owner.active
        _dispatch_updates(_update_snapshot()["service_update"], .01, .01, gameplay_phase=False)
        assert module.calls == []
        coordinator.drain()
        assert coordinator.states[path].status == "failed"
        assert isinstance(obj.get_py_components()[0], MissingScript)
        assert f"current {failing_phase} failed" in coordinator.states[path].error
    finally:
        manager.stop()


def test_deferred_source_retirement_cancels_remaining_steps_and_completion(reload_project):
    _scene, coordinator, write = reload_project
    path = write("EditorDeferredRetirement", '''
        calls = []
        def first(): calls.append("first")
        def second(): calls.append("second")
        def done(ok): calls.append(("done", ok))
    ''')
    coordinator.ensure_ready((path,))
    module = sys.modules["EditorDeferredRetirement"]
    runner = DeferredTaskRunner()
    assert runner.submit("owned", [("one", .2, module.first), ("two", .8, module.second)], module.done)
    runner.tick()
    assert module.calls == ["first"] and runner.is_busy
    write("EditorDeferredRetirement", "def bad(:\n")
    coordinator.request((path,))
    assert not runner.is_busy
    runner.tick()
    assert module.calls == ["first"]


def test_deferred_submit_owner_cancels_core_callbacks(reload_project):
    _scene, coordinator, write = reload_project
    path = write("EditorDeferredSubmitter", "value = 1\n")
    coordinator.ensure_ready((path,))
    owner = sys.modules["EditorDeferredSubmitter"].__infernux_reload_owner__
    calls = []
    runner = DeferredTaskRunner()
    with owner.enter():
        assert runner.submit("owned", [("one", .5, lambda: calls.append("step"))], calls.append)
    write("EditorDeferredSubmitter", "def bad(:\n")
    coordinator.request((path,))
    runner.tick()
    assert not runner.is_busy and calls == []


def test_deferred_completion_source_retirement_cancels_whole_task(reload_project):
    _scene, coordinator, write = reload_project
    step = write("EditorDeferredStep", "calls = []\ndef step(): calls.append('step')\n")
    completion = write("EditorDeferredCompletion", "calls = []\ndef done(ok): calls.append(ok)\n")
    coordinator.ensure_ready((step, completion))
    first, last = sys.modules["EditorDeferredStep"], sys.modules["EditorDeferredCompletion"]
    runner = DeferredTaskRunner()
    assert runner.submit("owned", [("one", .5, first.step)], last.done)
    write("EditorDeferredCompletion", "def bad(:\n")
    coordinator.request((completion,))
    runner.tick()
    assert not runner.is_busy and first.calls == [] and last.calls == []


def test_completed_and_cancelled_deferred_work_does_not_accumulate_source_cleanup(reload_project):
    _scene, coordinator, write = reload_project
    path = write("EditorDeferredBoundedLifetime", "def step(): pass\ndef done(ok): pass\n")
    coordinator.ensure_ready((path,))
    module = sys.modules["EditorDeferredBoundedLifetime"]
    owner = module.__infernux_reload_owner__
    baseline = len(owner._cleanup), len(owner._children)
    runner = DeferredTaskRunner()
    for _ in range(64):
        assert runner.submit("bounded", [("one", .5, module.step)], module.done)
        assert len(owner._children) == baseline[1] + 1
        runner.tick()
        assert not runner.is_busy
        assert (len(owner._cleanup), len(owner._children)) == baseline
        assert runner.submit("cancelled", [("one", .5, module.step)], module.done)
        runner.cancel()
        assert (len(owner._cleanup), len(owner._children)) == baseline


def test_deferred_callbacks_own_derived_services_and_keep_false_result(reload_project):
    _scene, coordinator, write = reload_project
    path = write("EditorDeferredServices", '''
        from infernux.runtime_services import install_runtime_service
        results = []
        def step():
            install_runtime_service("editor-deferred-step", object())
            return False
        def done(ok):
            results.append(ok)
            install_runtime_service("editor-deferred-complete", object())
    ''')
    coordinator.ensure_ready((path,))
    module = sys.modules["EditorDeferredServices"]
    runner = DeferredTaskRunner()
    try:
        assert runner.submit("owned", [("one", .5, module.step)], module.done)
        runner.tick()
        assert module.results == [False]
        assert get_runtime_service("editor-deferred-step") is not None
        assert get_runtime_service("editor-deferred-complete") is not None
        write("EditorDeferredServices", "def bad(:\n")
        coordinator.request((path,))
        assert get_runtime_service("editor-deferred-step") is None
        assert get_runtime_service("editor-deferred-complete") is None
    finally:
        remove_runtime_service("editor-deferred-step")
        remove_runtime_service("editor-deferred-complete")


def test_theme_listener_owns_work_created_when_core_changes_theme(reload_project):
    _scene, coordinator, write = reload_project
    path = write("EditorThemeServiceOwner", '''
        from infernux.engine.ui.theme import on_theme_changed
        from infernux.runtime_services import install_runtime_service
        def changed(): install_runtime_service("editor-theme-derived", object())
        remove = on_theme_changed(changed)
    ''')
    previous = theme.active_editor_theme()
    coordinator.ensure_ready((path,))
    try:
        assert theme.set_editor_theme("amber" if previous != "amber" else "graphite")
        assert get_runtime_service("editor-theme-derived") is not None
        write("EditorThemeServiceOwner", "def bad(:\n")
        coordinator.request((path,))
        assert get_runtime_service("editor-theme-derived") is None
    finally:
        remove_runtime_service("editor-theme-derived")
        theme.set_editor_theme(previous)


def test_theme_snapshot_skips_listener_retired_by_previous_listener(reload_project):
    _scene, coordinator, write = reload_project
    first = write("EditorThemeFirst", '''
        from infernux.runtime_services import get_runtime_service
        def changed(): get_runtime_service("editor-theme-retire")()
    ''')
    second = write("EditorThemeSecond", "calls = []\ndef changed(): calls.append('stale')\n")
    coordinator.ensure_ready((first, second))
    a, b = sys.modules["EditorThemeFirst"], sys.modules["EditorThemeSecond"]
    remove_a = theme.on_theme_changed(a.changed)
    remove_b = theme.on_theme_changed(b.changed)
    previous = theme.active_editor_theme()
    install_runtime_service("editor-theme-retire", lambda: coordinator.request((), removed=(second,)))
    try:
        assert theme.set_editor_theme("amber" if previous != "amber" else "graphite")
        assert b.calls == []
    finally:
        remove_a()
        remove_b()
        remove_runtime_service("editor-theme-retire")
        theme.set_editor_theme(previous)
