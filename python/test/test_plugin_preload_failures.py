"""A real installed package never reports success after preload discovery fails."""
import json
from pathlib import Path

import pytest

from infernux.engine import player_package_native
from infernux.plugins import InxPackage, PluginManager
from infernux.plugins.preload import _ensure_script_guid
from infernux.version import ENGINE_VERSION


@pytest.fixture
def package_project(tmp_path, monkeypatch):
    assert not player_package_native.using_test_backend()
    root = tmp_path / 'Project'
    (root / 'Assets').mkdir(parents=True)
    (root / 'ProjectSettings').mkdir()
    monkeypatch.setenv('INFERNUX_PACKAGE_CACHE_ROOT', str(tmp_path / 'cache'))
    manager = PluginManager(str(root))

    def install(reference, sources):
        source = tmp_path / ('source-' + reference.replace('/', '-'))
        source.mkdir()
        (source / 'inx_package.json').write_text(json.dumps({
            'reference': reference, 'version': '1.0.0', 'engine': '==' + ENGINE_VERSION,
        }), encoding='utf-8')
        for name, content in sources.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        archive = InxPackage.export_source(str(source), str(source.with_suffix('.inxpkg')))
        return manager.install_package(archive.package_path, install_dependencies=False)

    try:
        yield root, manager, install
    finally:
        manager.shutdown()


GOOD_SOURCE = '''from infernux.lifecycle import InxPreload
class Startup(InxPreload):
    def preload(self, context):
        self.ready = True
'''
FAILURES = {
    'import': "raise RuntimeError('preload-import-rejected')\n" + GOOD_SOURCE,
    'syntax': GOOD_SOURCE + '\ndef invalid(:\n',
    'body': GOOD_SOURCE.replace('self.ready = True', "raise RuntimeError('preload-body-rejected')"),
    'constructor': GOOD_SOURCE + "    def __init__(self):\n        raise RuntimeError('preload-constructor-rejected')\n",
}


@pytest.mark.parametrize('failure', list(FAILURES))
def test_installed_package_failure_is_reported_and_repair_clears_diagnostics(package_project, failure):
    root, manager, install = package_project
    state = install('test/preload-failure', {'runtime/startup.py': FAILURES[failure]})
    assert not state.loaded
    assert state.error
    assert ('SyntaxError' if failure == 'syntax' else 'preload-' + failure + '-rejected') in state.error
    for _ in range(2):
        manager.reload_all()
        state = manager.states['test/preload-failure']
        assert not state.loaded and state.error
    source = root / 'Packages/test/preload-failure/runtime/startup.py'
    source.write_text(GOOD_SOURCE, encoding='utf-8')
    state = manager.reload('test/preload-failure')
    assert state.loaded and not state.error
    assert len(state.lifecycle) == 1 and state.lifecycle[0]['loaded']
    assert not manager.preloads.failures


def test_unchanged_syntax_failure_survives_declaration_catalog_cache(package_project):
    _root, manager, install = package_project
    install('test/invalid-syntax', {'runtime/startup.py': FAILURES['syntax']})
    assert manager.preloads.failures
    for _ in range(2):
        manager.reload_all()
        assert manager.preloads.failures
        assert any('SyntaxError' in error for error in manager.preloads.failures.values())


def test_package_failure_does_not_poison_healthy_or_resource_only_packages(package_project):
    _root, manager, install = package_project
    install('test/broken', {'runtime/startup.py': FAILURES['import']})
    install('test/healthy', {'runtime/startup.py': GOOD_SOURCE})
    install('test/resources', {'runtime/data.txt': 'ready'})
    for refresh in (lambda: None, manager.reload_all):
        refresh()
        broken, healthy, resources = (manager.states['test/' + name] for name in ('broken', 'healthy', 'resources'))
        assert not broken.loaded and 'preload-import-rejected' in broken.error
        assert healthy.loaded and not healthy.error and len(healthy.lifecycle) == 1
        assert resources.loaded and not resources.error and not resources.lifecycle


def test_catch_up_publishes_failure_even_when_no_lifecycle_was_created(package_project):
    root, manager, install = package_project
    state = install('test/catchup', {'runtime/data.txt': 'ready'})
    assert state.loaded
    source = root / 'Packages/test/catchup/runtime/startup.py'
    source.write_text(FAILURES['import'], encoding='utf-8')
    _ensure_script_guid(str(source), str(root))
    assert manager.catch_up_preloads() == ()
    state = manager.states['test/catchup']
    assert not state.loaded and 'preload-import-rejected' in state.error
    source.write_text(GOOD_SOURCE, encoding='utf-8')
    state = manager.reload('test/catchup')
    assert state.loaded and not state.error and len(state.lifecycle) == 1


def test_failed_reload_preserves_live_service_but_reports_rejected_source(package_project):
    root, manager, install = package_project
    state = install('test/live-service', {'runtime/startup.py': GOOD_SOURCE})
    identity = state.lifecycle[0]['identity']
    original = manager.preloads.states[identity].instance
    source = root / 'Packages/test/live-service/runtime/startup.py'
    source.write_text(FAILURES['syntax'], encoding='utf-8')
    for _ in range(2):
        state = manager.reload('test/live-service')
        assert not state.loaded and 'SyntaxError' in state.error
        assert manager.preloads.states[identity].instance is original
        assert state.lifecycle[0]['loaded']
    source.write_text(GOOD_SOURCE, encoding='utf-8')
    state = manager.reload('test/live-service')
    assert state.loaded and not state.error
    assert manager.preloads.states[identity].instance is not original
