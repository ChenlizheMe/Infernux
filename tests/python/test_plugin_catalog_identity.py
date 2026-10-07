"""Registry acquisition preserves the selected identity before publishing anything."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

import pytest

from infernux.plugins import InxPackage
from infernux.version import ENGINE_VERSION
from test_plugin_preload_failures import package_project


@contextmanager
def local_archive_server(archive):
    requests = []
    payload = archive.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/plugin.inxpkg', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def archive_at(tmp_path, reference, version, marker, *, requirements=''):
    source = tmp_path / ('source-' + reference.replace('/', '-') + '-' + version)
    (source / 'runtime').mkdir(parents=True)
    (source / 'inx_package.json').write_text(json.dumps({
        'reference': reference, 'version': version, 'engine': '==' + ENGINE_VERSION,
    }), encoding='utf-8')
    (source / 'runtime/entry.py').write_text(
        'from pathlib import Path\nfrom infernux.lifecycle import InxPreload\n'
        f'Path({str(marker)!r}).write_text("executed")\n'
        'class Startup(InxPreload):\n    pass\n', encoding='utf-8')
    if requirements:
        (source / 'requirements.txt').write_text(requirements, encoding='utf-8')
    target = source.with_suffix('.inxpkg')
    return Path(InxPackage.export_source(str(source), str(target)).package_path), source


def authored_files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for folder in ('ProjectSettings', 'Packages')
            for p in (root / folder).rglob('*') if p.is_file()}


def register_source(manager, reference, archive, source_folder, mode, url):
    source = {'type': 'url', 'location': url}
    if mode in ('local', 'builtin'):
        source = {'type': 'local', 'location': str(source_folder if mode == 'builtin' else archive)}
        if mode == 'builtin':
            source['builtin'] = True
    manager.registry.add_package(reference, version='1.0.0', engine='==' + ENGINE_VERSION, source=source)
    if mode == 'cached':
        path = Path(manager._package_cache().path(reference, '1.0.0'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read_bytes())  # Deliberately mislabel a complete native archive.


@pytest.mark.parametrize('mode', ['url', 'local', 'builtin', 'cached'])
@pytest.mark.parametrize('operation', ['install', 'download'])
@pytest.mark.parametrize('mismatch', ['reference', 'version'])
def test_registry_identity_rejected_before_publication(package_project, tmp_path, mode, operation, mismatch):
    root, manager, _ = package_project
    reference = 'test/catalog-pin'
    marker = tmp_path / 'must-not-execute'
    archive, source = archive_at(tmp_path,
        'test/different-package' if mismatch == 'reference' else reference,
        '2.0.0' if mismatch == 'version' else '1.0.0', marker)
    with local_archive_server(archive) as (url, requests):
        register_source(manager, reference, archive, source, mode, url)
        before = authored_files(root)
        cache_before = {p.relative_to(manager._package_cache().root): p.read_bytes()
                        for p in Path(manager._package_cache().root).rglob('*.inxpkg')}
        with pytest.raises(RuntimeError, match=mismatch + ' mismatch'):
            if operation == 'install':
                manager.install_reference(reference, install_dependencies=False)
            else:
                manager.download_reference(reference)
        assert authored_files(root) == before
        assert not manager.registry.installed()
        assert not marker.exists()
        assert {p.relative_to(manager._package_cache().root): p.read_bytes()
                for p in Path(manager._package_cache().root).rglob('*.inxpkg')} == cache_before
        assert requests == (['/plugin.inxpkg'] if mode == 'url' else [])


@pytest.mark.parametrize('mode', ['url', 'local', 'builtin', 'cached'])
def test_matching_pin_installs_and_then_reloads_without_acquisition(package_project, tmp_path, mode):
    root, manager, _ = package_project
    reference = 'test/catalog-pin'
    marker = tmp_path / 'did-execute'
    archive, source = archive_at(tmp_path, reference, '1.0.0', marker)
    with local_archive_server(archive) as (url, requests):
        register_source(manager, reference, archive, source, mode, url)
        state = manager.install_reference(reference, install_dependencies=False)
        assert state.loaded and marker.read_text() == 'executed'
        record = manager.registry.installed_record(reference)
        assert record['version'] == '1.0.0'
        # A refreshed catalog is discovery, not an upgrade request.
        manager.registry.add_package(reference, version='2.0.0', source={'type': 'url', 'location': url})
        state = manager.install_reference(reference, install_dependencies=False)
        assert state.loaded and manager.registry.installed_record(reference)['version'] == '1.0.0'
        assert requests == (['/plugin.inxpkg'] if mode == 'url' else [])


@pytest.mark.parametrize('mismatch', ['reference', 'version'])
def test_registry_dependency_rejection_does_not_install_parent(package_project, tmp_path, mismatch):
    root, manager, _ = package_project
    reference = 'test/catalog-dependency'
    marker = tmp_path / 'dependency-executed'
    archive, source = archive_at(tmp_path,
        'test/unrequested-dependency' if mismatch == 'reference' else reference,
        '2.0.0' if mismatch == 'version' else '1.0.0', marker)
    parent_marker = tmp_path / 'parent-executed'
    parent, _ = archive_at(tmp_path, 'test/catalog-parent', '1.0.0', parent_marker, requirements='inx:' + reference)
    with local_archive_server(archive) as (url, requests):
        register_source(manager, reference, archive, source, 'url', url)
        before = authored_files(root)
        with pytest.raises(RuntimeError, match=mismatch + ' mismatch'):
            manager.install_package(str(parent))
        assert not manager.registry.installed()
        assert authored_files(root) == before
        assert not marker.exists() and not parent_marker.exists()
        assert requests == ['/plugin.inxpkg']


@pytest.mark.parametrize('operation', ['install_package', 'install_source'])
def test_explicit_unqualified_import_uses_archive_identity(package_project, tmp_path, operation):
    _root, manager, _ = package_project
    marker = tmp_path / 'explicit-import'
    archive, _ = archive_at(tmp_path, 'test/explicit-import', '2.0.0', marker)
    state = getattr(manager, operation)(str(archive), install_dependencies=False)
    assert state.reference == 'test/explicit-import' and state.loaded
    assert manager.registry.installed_record(state.reference)['version'] == '2.0.0'
    assert marker.read_text() == 'executed'


@pytest.mark.parametrize('mismatch', ['reference', 'version', 'matching'])
def test_default_library_install_respects_catalog_identity(package_project, tmp_path, mismatch):
    from infernux.plugins.official import install_default_libraries, sync_official_registry

    root, manager, _ = package_project
    reference = 'test/catalog-default'
    marker = tmp_path / 'default-executed'
    archive, _ = archive_at(tmp_path,
        'test/unrequested-default' if mismatch == 'reference' else reference,
        '2.0.0' if mismatch == 'version' else '1.0.0', marker)
    resources = tmp_path / 'resources'
    catalog = resources / 'official_packages'
    catalog.mkdir(parents=True)
    with local_archive_server(archive) as (url, requests):
        (catalog / 'official-registry.json').write_text(json.dumps({
            '$schema': 'infernux.official_plugin_registry', 'packages': [{
                'reference': reference, 'version': '1.0.0', 'engine': '==' + ENGINE_VERSION,
                'artifact': 'default.inxpkg', 'source': {'type': 'url', 'location': url},
            }],
        }), encoding='utf-8')
        (catalog / 'default-libraries.json').write_text(json.dumps({
            '$schema': 'infernux.default_libraries', 'libraries': [reference],
        }), encoding='utf-8')
        sync_official_registry(str(root), resources_root=str(resources))
        before = authored_files(root)
        if mismatch == 'matching':
            states = install_default_libraries(str(root), resources_root=str(resources), manager=manager)
            assert len(states) == 1 and states[0].reference == reference and states[0].loaded
            assert marker.read_text() == 'executed'
        else:
            with pytest.raises(RuntimeError, match=mismatch + ' mismatch'):
                install_default_libraries(str(root), resources_root=str(resources), manager=manager)
            assert not manager.registry.installed() and not marker.exists()
            assert authored_files(root) == before
        assert requests == ['/plugin.inxpkg']


def test_matching_catalog_downloads_once_then_imports_offline(package_project, tmp_path):
    _root, manager, _ = package_project
    reference = 'test/catalog-download'
    marker = tmp_path / 'download-import'
    archive, source = archive_at(tmp_path, reference, '1.0.0', marker)
    with local_archive_server(archive) as (url, requests):
        register_source(manager, reference, archive, source, 'url', url)
        first = manager.download_reference(reference)
        assert first['reference'] == reference and not first['cached']
        assert not marker.exists() and not manager.registry.installed()
    # The server is closed: both operations must resolve the pinned archive locally.
    second = manager.download_reference(reference)
    assert second['cached'] and second['path'] == first['path']
    state = manager.install_reference(reference, install_dependencies=False)
    assert state.loaded and marker.read_text() == 'executed'
    assert requests == ['/plugin.inxpkg']
