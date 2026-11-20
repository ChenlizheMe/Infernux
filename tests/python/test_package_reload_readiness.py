"""Installed package readiness is checked before any current module executes."""
import sys
from pathlib import Path

import pytest

from infernux.engine.project_context import get_project_root, set_project_root
from infernux.engine.resources_manager import ResourcesManager
from infernux.plugins import PluginManager
from infernux.runtime_services import get_runtime_service, remove_runtime_service, _dispatch_updates, _update_snapshot
from test_inxpackage_plugins import _source, _project, _export


@pytest.fixture
def readiness_project(tmp_path, engine, monkeypatch):
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    root = _project(tmp_path / "project")
    previous = get_project_root()
    set_project_root(str(root))
    installer = PluginManager(str(root), installation_only=True)

    def install(reference, source=None):
        folder = _source(tmp_path / reference.replace('/', '_'), reference)
        runtime = folder / "runtime"
        runtime.mkdir()
        name = 'payload.py' if source is not None else 'payload.txt'
        (runtime / name).write_text(source or 'ordinary package data', encoding='utf-8')
        package = _export(folder, folder.with_suffix('.inxpkg'))
        installer.install_package(str(package), install_dependencies=False)
        return root / 'Packages' / reference / 'runtime' / name

    resources = ResourcesManager(str(root), engine)
    manager = PluginManager(str(root), runtime=True)
    def change(reference, **values):
        document = manager.registry.load()
        record = next(item for item in document['installed'] if item['reference'] == reference)
        record.update(values)
        manager.registry.save(document)
    try:
        yield root, resources, manager, install, change
    finally:
        manager.shutdown()
        resources.cleanup()
        installer.shutdown()
        for name, module in tuple(sys.modules.items()):
            if str(getattr(module, '__file__', '')).startswith(str(root)):
                sys.modules.pop(name, None)
        for name in ('readiness-current', 'readiness-dependency'):
            remove_runtime_service(name)
        set_project_root(previous)


SOURCE = '''from infernux.runtime_services import install_runtime_service, register_update
class CurrentService:
    def marker(self): return 'current code executed'
install_runtime_service('readiness-current', CurrentService())
calls = []
def tick(dt): calls.append(1)
registration = register_update(tick, phase='service_update')
'''


@pytest.mark.parametrize('failure', ['engine', 'missing', 'disabled', 'transitive', 'cycle'])
@pytest.mark.parametrize('reload_method', ['reload', 'reload_all'])
def test_invalid_installed_package_cannot_run_and_unchanged_source_recovers(readiness_project, failure, reload_method):
    _root, resources, manager, install, change = readiness_project
    path = install('review/current', SOURCE)
    install('review/dependency')
    install('review/leaf')
    unchanged = path.read_bytes()
    if failure == 'engine':
        change('review/current', engine='>=999')
    elif failure == 'missing':
        change('review/current', dependencies=['review/missing'])
    elif failure == 'disabled':
        change('review/current', dependencies=['review/dependency'])
        change('review/dependency', enabled=False)
    elif failure == 'transitive':
        change('review/current', dependencies=['review/dependency'])
        change('review/dependency', dependencies=['review/leaf'])
        change('review/leaf', engine='>=999')
    else:
        change('review/current', dependencies=['review/dependency'])
        change('review/dependency', dependencies=['review/current'])
    manager.reload_all()
    state = manager.states['review/current']
    assert not state.loaded and state.error
    assert get_runtime_service('readiness-current') is None
    # The ordinary ResourcesManager entry point must enforce the same decision.
    with pytest.raises(RuntimeError, match='not ready'):
        resources._ensure_event_handler().reload.ensure_ready((str(path),))
    assert get_runtime_service('readiness-current') is None
    change('review/current', engine='>=0.4.8', dependencies=[])
    if reload_method == 'reload':
        state = manager.reload('review/current')
    else:
        manager.reload_all()
        state = manager.states['review/current']
    assert path.read_bytes() == unchanged
    assert state.loaded and not state.error
    assert get_runtime_service('readiness-current').marker() == 'current code executed'


def test_resource_only_package_reports_engine_failure_and_recovers(readiness_project):
    _root, _resources, manager, install, change = readiness_project
    install('review/resource-only')
    change('review/resource-only', engine='>=999')
    manager.reload_all()
    state = manager.states['review/resource-only']
    assert not state.loaded and '999' in state.error and 'current engine' in state.error
    change('review/resource-only', engine='>=0.4.8')
    assert manager.reload('review/resource-only').loaded


def test_resource_script_change_retires_plugin_service_and_repairs(readiness_project):
    _root, resources, manager, install, _change = readiness_project
    path = install('review/current', SOURCE)
    manager._attach_resource_events()
    manager.reload_all()
    previous = get_runtime_service('readiness-current')
    assert previous is not None

    path.write_text('def invalid(:\n', encoding='utf-8')
    resources.submit_script_change(str(path), origin='watchdog')
    assert get_runtime_service('readiness-current') is None
    assert not manager.states['review/current'].loaded

    path.write_text(SOURCE.replace('current code executed', 'repaired source'), encoding='utf-8')
    resources.submit_script_change(str(path), origin='watchdog')
    repaired = get_runtime_service('readiness-current')
    assert repaired is not previous and repaired.marker() == 'repaired source'
    assert manager.states['review/current'].loaded


def test_declared_package_dependency_retires_consumer_without_python_import(readiness_project):
    _root, resources, manager, install, change = readiness_project
    dependency = install('review/dependency', SOURCE.replace('readiness-current', 'readiness-dependency'))
    consumer = install('review/current', SOURCE)
    change('review/current', dependencies=['review/dependency'])
    manager.reload_all()
    assert manager.states['review/current'].loaded
    coordinator = resources._ensure_event_handler().reload
    module = sys.modules[coordinator.graph.module_for_path(str(consumer)).id.module_name]
    _dispatch_updates(_update_snapshot()['service_update'], .01, .01, gameplay_phase=False)
    assert module.calls == [1]
    dependency.write_text('def invalid(:\n', encoding='utf-8')
    assert str(consumer) in coordinator.request((str(dependency),))
    coordinator.compile_pending()
    coordinator.drain()
    _dispatch_updates(_update_snapshot()['service_update'], .01, .01, gameplay_phase=False)
    assert module.calls == [1]
    assert get_runtime_service('readiness-current') is None
    assert not manager.states['review/current'].loaded
    assert manager.states['review/current'].error
