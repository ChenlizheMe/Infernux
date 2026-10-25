"""Registry acquisition preserves the selected identity before publishing anything."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread

import pytest

from infernux.plugins import InxPackage
from infernux.version import ENGINE_VERSION
from test_plugin_preload_failures import package_project as package_project


@contextmanager
def local_archive_server(archive):
    requests = []
    payload = archive.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/plugin.inxpkg", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def archive_at(tmp_path, reference, version, marker, *, requirements=""):
    source = tmp_path / ("source-" + reference.replace("/", "-") + "-" + version)
    (source / "runtime").mkdir(parents=True)
    (source / "inx_package.json").write_text(
        json.dumps(
            {
                "reference": reference,
                "version": version,
                "engine": "==" + ENGINE_VERSION,
            }
        ),
        encoding="utf-8",
    )
    (source / "runtime/entry.py").write_text(
        "from pathlib import Path\nfrom infernux.lifecycle import InxPreload\n"
        f'Path({str(marker)!r}).write_text("executed")\n'
        "class Startup(InxPreload):\n    pass\n",
        encoding="utf-8",
    )
    if requirements:
        (source / "requirements.txt").write_text(requirements, encoding="utf-8")
    target = source.with_suffix(".inxpkg")
    return Path(InxPackage.export_source(str(source), str(target)).package_path), source


def authored_files(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for folder in ("ProjectSettings", "Packages")
        for p in (root / folder).rglob("*")
        if p.is_file()
    }


def register_source(manager, reference, archive, source_folder, mode, url):
    source = {"type": "url", "location": url}
    if mode in ("local", "builtin"):
        source = {"type": "local", "location": str(source_folder if mode == "builtin" else archive)}
        if mode == "builtin":
            source["builtin"] = True
    manager.registry.add_package(reference, version="1.0.0", engine="==" + ENGINE_VERSION, source=source)
    if mode == "cached":
        path = Path(manager._package_cache().path(reference, "1.0.0"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read_bytes())  # Deliberately mislabel a complete native archive.


@pytest.mark.parametrize("mode", ["url", "local", "builtin", "cached"])
def test_matching_pin_installs_and_then_reloads_without_acquisition(package_project, tmp_path, mode):
    root, manager, _ = package_project
    reference = "test/catalog-pin"
    marker = tmp_path / "did-execute"
    archive, source = archive_at(tmp_path, reference, "1.0.0", marker)
    with local_archive_server(archive) as (url, requests):
        register_source(manager, reference, archive, source, mode, url)
        state = manager.install_reference(reference, install_dependencies=False)
        assert state.loaded and marker.read_text() == "executed"
        record = manager.registry.installed_record(reference)
        assert record["version"] == "1.0.0"
        # A refreshed catalog is discovery, not an upgrade request.
        manager.registry.add_package(reference, version="2.0.0", source={"type": "url", "location": url})
        state = manager.install_reference(reference, install_dependencies=False)
        assert state.loaded and manager.registry.installed_record(reference)["version"] == "1.0.0"
        assert requests == (["/plugin.inxpkg"] if mode == "url" else [])


@pytest.mark.parametrize("modified", [False, True])
def test_builtin_release_updates_with_an_empty_original_cache(package_project, tmp_path, monkeypatch, modified):
    from infernux.plugins.official import install_bundled_packages

    root, manager, _ = package_project
    reference = "test/catalog-builtin"
    old_marker, new_marker = tmp_path / "old", tmp_path / "new"
    archive, source = archive_at(tmp_path, reference, "1.0.0", old_marker)
    manager.install_package(str(archive), source={"type": "local", "location": str(archive), "builtin": True},
                            install_dependencies=False)
    before = manager.registry.installed_record(reference)
    baseline = Path(manager._installed_archive_path(before))
    baseline.unlink()
    resources = tmp_path / "resources"
    (resources / "official_packages").mkdir(parents=True)
    manifest = source / "inx_package.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["version"] = "2.0.0"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    entry = source / "runtime/entry.py"
    entry.write_text(entry.read_text(encoding="utf-8").replace(repr(str(old_marker)), repr(str(new_marker))),
                     encoding="utf-8")
    InxPackage.export_source(str(source), str(resources / "test.catalog-builtin.inxpkg"))
    authored = root / "Packages/test/catalog-builtin/runtime/entry.py"
    if modified:
        authored.write_text(authored.read_text(encoding="utf-8") + "LOCAL_EDIT = True\n", encoding="utf-8")
    monkeypatch.setattr("infernux.resources.get_package_resources_path", lambda: str(resources))
    with local_archive_server(archive) as (url, requests):
        published = url.replace("/plugin.inxpkg", "/2.0.0/plugin.inxpkg")
        catalog = {"$schema": "infernux.official_plugin_registry", "packages": [
            {"reference": reference, "version": "2.0.0", "source": {"type": "url", "location": published}},
        ]}
        (resources / "official_packages/official-registry.json").write_text(json.dumps(catalog), encoding="utf-8")
        if modified:
            from infernux.plugins import PackageUpdateConflict

            with pytest.raises(PackageUpdateConflict, match="runtime/entry.py"):
                install_bundled_packages(str(root), resources_root=str(resources), manager=manager)
            assert manager.registry.installed_record(reference) == before
            assert authored.read_text(encoding="utf-8").endswith("LOCAL_EDIT = True\n")
            assert not new_marker.exists()
        else:
            install_bundled_packages(str(root), resources_root=str(resources), manager=manager)
            record = manager.registry.installed_record(reference)
            assert record["version"] == "2.0.0" and new_marker.read_text() == "executed"
            assert record["control"]["guid"] == before["control"]["guid"]
            assert record["source"]["location"] == published and record["source"]["builtin"]
            assert install_bundled_packages(str(root), resources_root=str(resources), manager=manager) == ()
        assert requests == ["/1.0.0/plugin.inxpkg"] and baseline.is_file()


@pytest.mark.parametrize("mismatch", ["reference", "version"])
def test_registry_dependency_rejection_does_not_install_parent(package_project, tmp_path, mismatch):
    root, manager, _ = package_project
    reference = "test/catalog-dependency"
    marker = tmp_path / "dependency-executed"
    archive, source = archive_at(
        tmp_path,
        "test/unrequested-dependency" if mismatch == "reference" else reference,
        "2.0.0" if mismatch == "version" else "1.0.0",
        marker,
    )
    parent_marker = tmp_path / "parent-executed"
    parent, _ = archive_at(tmp_path, "test/catalog-parent", "1.0.0", parent_marker, requirements="inx:" + reference)
    with local_archive_server(archive) as (url, requests):
        register_source(manager, reference, archive, source, "url", url)
        before = authored_files(root)
        with pytest.raises(RuntimeError, match=mismatch + " mismatch"):
            manager.install_package(str(parent))
        assert not manager.registry.installed()
        assert authored_files(root) == before
        assert not marker.exists() and not parent_marker.exists()
        assert requests == ["/plugin.inxpkg"]


@pytest.mark.parametrize("mismatch", ["reference", "version", "matching"])
def test_default_library_install_respects_catalog_identity(package_project, tmp_path, mismatch):
    from infernux.plugins.official import install_default_libraries, sync_official_registry

    root, manager, _ = package_project
    reference = "test/catalog-default"
    marker = tmp_path / "default-executed"
    archive, _ = archive_at(
        tmp_path,
        "test/unrequested-default" if mismatch == "reference" else reference,
        "2.0.0" if mismatch == "version" else "1.0.0",
        marker,
    )
    resources = tmp_path / "resources"
    catalog = resources / "official_packages"
    catalog.mkdir(parents=True)
    with local_archive_server(archive) as (url, requests):
        (catalog / "official-registry.json").write_text(
            json.dumps(
                {
                    "$schema": "infernux.official_plugin_registry",
                    "packages": [
                        {
                            "reference": reference,
                            "version": "1.0.0",
                            "engine": "==" + ENGINE_VERSION,
                            "artifact": "default.inxpkg",
                            "source": {"type": "url", "location": url},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (catalog / "default-libraries.json").write_text(
            json.dumps(
                {
                    "$schema": "infernux.default_libraries",
                    "libraries": [reference],
                }
            ),
            encoding="utf-8",
        )
        sync_official_registry(str(root), resources_root=str(resources))
        before = authored_files(root)
        if mismatch == "matching":
            states = install_default_libraries(str(root), resources_root=str(resources), manager=manager)
            assert len(states) == 1 and states[0].reference == reference and states[0].loaded
            assert marker.read_text() == "executed"
        else:
            with pytest.raises(RuntimeError, match=mismatch + " mismatch"):
                install_default_libraries(str(root), resources_root=str(resources), manager=manager)
            assert not manager.registry.installed() and not marker.exists()
            assert authored_files(root) == before
        assert requests == ["/plugin.inxpkg"]


def test_matching_catalog_downloads_once_then_imports_offline(package_project, tmp_path):
    _root, manager, _ = package_project
    reference = "test/catalog-download"
    marker = tmp_path / "download-import"
    archive, source = archive_at(tmp_path, reference, "1.0.0", marker)
    with local_archive_server(archive) as (url, requests):
        register_source(manager, reference, archive, source, "url", url)
        first = manager.download_reference(reference)
        assert first["reference"] == reference and not first["cached"]
        assert not marker.exists() and not manager.registry.installed()
    # The server is closed: both operations must resolve the pinned archive locally.
    second = manager.download_reference(reference)
    assert second["cached"] and second["path"] == first["path"]
    state = manager.install_reference(reference, install_dependencies=False)
    assert state.loaded and marker.read_text() == "executed"
    assert requests == ["/plugin.inxpkg"]


def test_android_install_and_update_do_not_require_a_host_toolchain(package_project, tmp_path, monkeypatch):
    root, manager, _ = package_project
    monkeypatch.setenv("INFERNUX_SHARED_DATA_ROOT", str(tmp_path / "custom-hub/Shared"))
    monkeypatch.delenv("INFERNUX_ANDROID_SUPPORT_ROOT", raising=False)
    reference = "infernux/platform-android"
    marker = tmp_path / "android-import"
    for version in ("0.2.3", "0.2.9"):
        archive, _ = archive_at(tmp_path, reference, version, marker)
        state = manager.install_package(str(archive), update=version == "0.2.9")
        assert state.loaded and not state.error
        assert manager.registry.installed_record(reference)["version"] == version
    assert marker.read_text() == "executed"
    assert not (tmp_path / "custom-hub").exists()


def test_standalone_plugin_process_uses_the_hubs_recorded_shared_root(tmp_path, monkeypatch):
    from infernux.engine.user_data import get_infernux_shared_data_root
    from infernux.plugins.cache import package_cache_root
    from infernux.plugins.platform_support import android_support_root, ANDROID_SUPPORT_VERSION, _host_id

    user = tmp_path / "user"
    shared = tmp_path / "Custom Hub/InfernuxHubData/Shared"
    user.mkdir()
    (user / "hub-shared-root.json").write_text(json.dumps({"shared_data_root": str(shared)}), encoding="utf-8")
    monkeypatch.setenv("INFERNUX_DATA_ROOT", str(user))
    for key in ("INFERNUX_SHARED_DATA_ROOT", "INFERNUX_ANDROID_SUPPORT_ROOT", "INFERNUX_PACKAGE_CACHE_ROOT"):
        monkeypatch.delenv(key, raising=False)
    assert Path(get_infernux_shared_data_root()) == shared
    assert Path(package_cache_root()) == shared / "Library/Plugins"
    assert android_support_root() == shared / "PlatformKits/android" / ANDROID_SUPPORT_VERSION / _host_id()
    override = tmp_path / "explicit-shared"
    monkeypatch.setenv("INFERNUX_SHARED_DATA_ROOT", str(override))
    assert Path(get_infernux_shared_data_root()) == override


def test_offline_catalog_keeps_newest_discovery_without_upgrading_installed_packages(package_project, tmp_path):
    from infernux.plugins.cache import package_cache_root
    from infernux.plugins.official import sync_official_registry

    _root, manager, install = package_project
    reference = "test/catalog-newest"
    install(reference, {"runtime/note.txt": "installed"})
    registry = manager.registry.load()
    for entry in registry["packages"]:
        if entry["reference"] == reference:
            entry["source"]["official"] = True
    manager.registry.save(registry)
    catalog = tmp_path / "resources/official_packages"
    catalog.mkdir(parents=True)
    cached = Path(package_cache_root()) / "official-registry.json"
    cached.parent.mkdir(parents=True, exist_ok=True)

    def document(version):
        return json.dumps({"$schema": "infernux.official_plugin_registry", "packages": [{
            "reference": reference, "version": version, "engine": ">=" + ENGINE_VERSION,
            "artifact": "test.inxpkg", "source": {"type": "url", "location": f"https://example.invalid/{version}"},
        }]})

    (catalog / "official-registry.json").write_text(document("2.0.0"), encoding="utf-8")
    for cached_version, expected in (("0.1.0", "2.0.0"), ("3.0.0", "3.0.0")):
        cached.write_text(document(cached_version), encoding="utf-8")
        entries = sync_official_registry(manager.project_root, resources_root=str(catalog.parent))
        assert entries[0]["version"] == expected
        assert manager.registry.installed_record(reference)["version"] == "1.0.0"
