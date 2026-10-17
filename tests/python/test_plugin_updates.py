from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from infernux.plugins import InxPackage, PackageConflictError, PackageUpdateConflict, PluginManager
from test_inxpackage_plugins import _fake_inxpack as _fake_inxpack, _project, _source


@pytest.fixture
def installed(tmp_path, monkeypatch):
    project = _project(tmp_path / "project")
    source = _source(tmp_path / "source", "vendor/plugin")
    runtime = source / "runtime"
    runtime.mkdir()
    (runtime / "note.txt").write_text("first", encoding="utf-8")
    (runtime / "retired.txt").write_text("retired", encoding="utf-8")
    (runtime / "retired.py").write_text("value = 1\n", encoding="utf-8")
    manager = PluginManager(str(project), runtime=True)
    old = tmp_path / "first.inxpkg"
    InxPackage.export_source(str(source), str(old))
    manager.install_package(str(old), install_dependencies=False)
    return manager, source, project / "Packages/vendor/plugin"


def next_package(source, version="2.0.0"):
    manifest = source / "inx_package.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    metadata["version"] = version
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    (source / "runtime/note.txt").write_text("second", encoding="utf-8")
    (source / "runtime/retired.txt").unlink(missing_ok=True)
    (source / "runtime/retired.py").unlink(missing_ok=True)
    (source / "runtime/new.txt").write_text("new", encoding="utf-8")
    package = source.parent / f"next-{version}.inxpkg"
    InxPackage.export_source(str(source), str(package))
    return str(package)


def test_registry_readers_wait_for_complete_environment_publication(installed, monkeypatch):
    from infernux.plugins import registry as registry_module

    manager, source, root = installed
    writer = manager.registry
    reader = registry_module.PluginRegistry(manager.project_root)
    document = writer.load()
    document["installed"][0]["version"] = "2.0.0"
    document["python_installs"] = [{"output": "new environment"}]
    writing_environment = threading.Event()
    release_writer = threading.Event()
    reader_started = threading.Event()
    reader_completed = threading.Event()
    original_write = registry_module.write_document_text

    def write(path, text):
        if path == writer.environment_path:
            writing_environment.set()
            assert release_writer.wait(5)
        return original_write(path, text)

    def read():
        reader_started.set()
        result = reader.load()
        reader_completed.set()
        return result

    monkeypatch.setattr(registry_module, "write_document_text", write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        published = pool.submit(writer.save, document)
        try:
            assert writing_environment.wait(5)
            observed = pool.submit(read)
            assert reader_started.wait(5)
            assert not reader_completed.wait(0.2)
        finally:
            release_writer.set()
        published.result(timeout=5)
        result = observed.result(timeout=5)
    assert result["installed"][0]["version"] == "2.0.0"
    assert result["python_installs"] == [{"output": "new environment"}]


@pytest.mark.parametrize("modified", [False, True])
def test_source_update_is_explicit_and_preserves_guid_and_local_edits(installed, modified):
    manager, source, root = installed
    before = manager.registry.installed_record("vendor/plugin")
    package = next_package(source)
    if modified:
        (root / "runtime/note.txt").write_text("local change", encoding="utf-8")
    with pytest.raises(RuntimeError, match="explicit package update"):
        manager.install_source(package, install_dependencies=False)
    if modified:
        with pytest.raises(PackageUpdateConflict, match="local edits"):
            manager.install_source(package, update=True, install_dependencies=False)
        assert manager.registry.installed_record("vendor/plugin") == before
        assert (root / "runtime/note.txt").read_text() == "local change"
    else:
        manager.install_source(package, update=True, install_dependencies=False)
        after = manager.registry.installed_record("vendor/plugin")
        assert after["version"] == "2.0.0"
        assert after["control"]["guid"] == before["control"]["guid"]
        assert (root / "runtime/note.txt").read_text() == "second"


def test_successive_updates_preserve_disabled_identity_author_files_and_guid_locations(installed):
    import py_compile

    manager, source, root = installed
    before = manager.registry.installed_record("vendor/plugin")
    original_guid = next(item["guid"] for item in before["files"] if item["logical_path"] == "runtime/note.txt")
    manager.set_enabled("vendor/plugin", False)
    extra = root / "runtime/author.py"
    extra.write_text("value = 1\n", encoding="utf-8")
    author_bytecode = Path(py_compile.compile(str(extra), doraise=True))
    owned_bytecode = Path(py_compile.compile(str(root / "runtime/retired.py"), doraise=True))
    meta = root / "runtime/note.txt.meta"
    document = json.loads(meta.read_bytes())
    document["importer_settings"] = {"custom": True}
    meta.write_text(json.dumps(document), encoding="utf-8")
    package = next_package(source)
    before_download = manager.registry.installed_record("vendor/plugin")
    with ThreadPoolExecutor(max_workers=1) as executor:
        acquired = executor.submit(manager.download_source, package).result(timeout=30)
    assert manager.registry.installed_record("vendor/plugin") == before_download
    assert (root / "runtime/note.txt").read_text() == "first"
    state = manager.install_package(
        acquired["path"], source=acquired["source"], update=True, install_dependencies=False
    )
    after = manager.registry.installed_record("vendor/plugin")
    assert state.enabled is False
    assert after["version"] == "2.0.0"
    assert after["control"]["guid"] == before["control"]["guid"]
    assert next(item["guid"] for item in after["files"] if item["logical_path"] == "runtime/note.txt") == original_guid
    assert (root / "runtime/note.txt").read_text() == "second"
    assert (root / "runtime/new.txt").read_text() == "new"
    assert json.loads(meta.read_bytes())["importer_settings"] == {"custom": True}
    assert not (root / "runtime/retired.txt").exists()
    assert not (root / "runtime/retired.txt.meta").exists()
    assert not owned_bytecode.exists()
    assert extra.is_file() and author_bytecode.is_file()
    assert extra.read_text(encoding="utf-8") == "value = 1\n"

    note = root / "runtime/note.txt"
    moved = root / "runtime/moved.txt"
    note.rename(moved)
    meta.rename(Path(str(moved) + ".meta"))
    publisher_guid = next(item["guid"] for item in after["files"] if item["logical_path"] == "runtime/new.txt")
    (source / "runtime/new.txt").rename(source / "runtime/renamed.txt")
    (source / "runtime/renamed.txt.meta").write_bytes((root / "runtime/new.txt.meta").read_bytes())
    (source / "runtime/note.txt").write_text("third", encoding="utf-8")
    manifest = source / "inx_package.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["version"] = "3.0.0"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    third = source.parent / "third.inxpkg"
    InxPackage.export_source(str(source), str(third))
    manager.install_package(str(third), update=True, install_dependencies=False)
    current = manager.registry.installed_record("vendor/plugin")
    assert current["version"] == "3.0.0" and current["enabled"] is False
    assert moved.read_text() == "third"
    assert not note.exists()
    assert not (root / "runtime/new.txt").exists()
    assert (root / "runtime/renamed.txt").read_text() == "new"
    assert (
        next(item["guid"] for item in current["files"] if item["logical_path"] == "runtime/renamed.txt")
        == publisher_guid
    )
    assert current["control"]["guid"] == before["control"]["guid"]
    assert extra.read_text(encoding="utf-8") == "value = 1\n"
    manager.uninstall("vendor/plugin")
    assert extra.is_file() and author_bytecode.is_file()
    assert extra.read_text(encoding="utf-8") == "value = 1\n"


def test_local_edits_require_explicit_overwrite_and_leave_pin_unchanged(installed):
    manager, source, root = installed
    note = root / "runtime/note.txt"
    note.write_text("local change", encoding="utf-8")
    package = next_package(source)
    registry_before = Path(manager.registry.path).read_bytes()
    with pytest.raises(PackageUpdateConflict) as caught:
        manager.install_package(package, update=True, install_dependencies=False)
    assert "Packages/vendor/plugin/runtime/note.txt" in caught.value.paths
    assert Path(manager.registry.path).read_bytes() == registry_before
    assert note.read_text() == "local change"
    manager.install_package(package, update=True, overwrite_modified=True, install_dependencies=False)
    assert note.read_text() == "second"


def test_regenerated_guid_is_rejected_even_with_overwrite_consent(installed):
    manager, source, root = installed
    meta = source / "runtime/note.txt.meta"
    document = json.loads((root / "runtime/note.txt.meta").read_bytes())
    document["metadata"]["guid"]["value"] = "a" * 32
    meta.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PackageConflictError, match="retain its GUID"):
        manager.install_package(next_package(source), update=True, overwrite_modified=True, install_dependencies=False)
    assert (root / "runtime/note.txt").read_text() == "first"


def test_update_registry_failure_restores_previous_files_and_pin(installed, monkeypatch):
    manager, source, root = installed
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    registry_before = Path(manager.registry.path).read_bytes()
    original = manager.registry.record_install

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("publication failed")

    monkeypatch.setattr(manager.registry, "record_install", fail)
    with pytest.raises(OSError, match="publication failed"):
        manager.install_package(next_package(source), update=True, install_dependencies=False)
    after = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert after == before
    assert Path(manager.registry.path).read_bytes() == registry_before


def test_missing_original_cache_does_not_silently_overwrite(installed):
    manager, source, root = installed
    record = manager.registry.installed_record("vendor/plugin")
    Path(manager._installed_archive_path(record)).unlink()
    with pytest.raises(PackageUpdateConflict):
        manager.install_package(next_package(source), update=True, install_dependencies=False)
    assert (root / "runtime/note.txt").read_text() == "first"


def test_cloned_project_updates_using_the_new_users_cache(installed, tmp_path, monkeypatch):
    from infernux.plugins.cache import SharedPackageCache

    manager, source, root = installed
    original_project = Path(manager.project_root)
    record = manager.registry.installed_record("vendor/plugin")
    original_archive = manager._installed_archive_path(record)
    clone = tmp_path / "another-checkout"
    shutil.copytree(original_project, clone, ignore=shutil.ignore_patterns("Library", "Cache", ".runtime"))
    new_cache = SharedPackageCache(tmp_path / "another-user-cache")
    new_archive = new_cache.store(original_archive, reference="vendor/plugin", version="1.0.0")
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", new_cache.root)
    registry_text = (clone / "ProjectSettings/InxPlugins.json").read_text(encoding="utf-8")
    assert str(original_project) not in registry_text
    assert str(tmp_path / "hub-package-cache") not in registry_text
    cloned_manager = PluginManager(str(clone), runtime=False)
    cloned_manager.install_package(next_package(source), update=True, install_dependencies=False)
    assert (clone / "Packages/vendor/plugin/runtime/note.txt").read_text() == "second"
    assert (root / "runtime/note.txt").read_text() == "first"
    assert cloned_manager.registry.installed_record("vendor/plugin")["control"]["guid"] == record["control"]["guid"]
    assert Path(new_archive).is_file()


def test_missing_synced_package_file_blocks_player_startup(installed):
    manager, source, root = installed
    manager.shutdown()
    (root / "runtime/note.txt").unlink()
    with pytest.raises(RuntimeError, match="Project package checkout is incomplete"):
        PluginManager.startup(manager.project_root, runtime=True)
    assert PluginManager.instance() is None


def test_player_plugin_state_uses_cooked_guids_without_editor_control_manifest(installed, monkeypatch):
    from infernux.application import Application
    from infernux.engine import project_context

    manager, source, root = installed
    record = manager.registry.installed_record("vendor/plugin")
    payloads = {item["guid"].casefold(): f"cooked/{item['guid']}.inxasset" for item in record["files"]}
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(project_context, "resolve_runtime_asset_guid", payloads.get)
    # No source files, metadata, or installation manifest survive the cook.
    shutil.rmtree(root)
    state = manager._rebuild_states()[0]
    assert state.loaded, state.error
    assert set(state.resources.values()) == set(payloads.values())
    missing = record["files"][0]
    del payloads[missing["guid"].casefold()]
    state = manager._rebuild_states()[0]
    assert not state.loaded
    assert f"Missing package asset vendor/plugin: {missing['logical_path']}" in state.error


def test_shared_removed_asset_ownership_transfers_to_remaining_plugin(installed):
    manager, source, root = installed
    current = manager.registry.installed_record("vendor/plugin")
    shared = next(item for item in current["files"] if item["logical_path"] == "runtime/retired.txt")
    manager.registry.record_install(
        {"reference": "vendor/other", "version": "1.0.0"},
        files=[{**shared, "owned": False}],
        control={"guid": "f" * 32, "owned": True, "path_hint": "Packages/vendor/other/inx_package.json"},
    )
    manager.install_package(next_package(source), update=True, install_dependencies=False)
    other = manager.registry.installed_record("vendor/other")
    assert other["files"][0]["owned"] is True
    assert (root / "runtime/retired.txt").exists()


def test_update_relinquishes_dropped_python_dependency_and_preserves_other_owner(installed, monkeypatch):
    manager, source, root = installed
    document = manager.registry.load()
    document["installed"][0]["python_requirements"] = [{"name": "old-dep", "requirement": "old-dep==1"}]
    document["python_dependencies"] = [
        {
            "name": "old-dep",
            "managed": True,
            "baseline_version": "",
            "installed_version": "1",
            "owners": [
                {"reference": "vendor/plugin", "requirements": ["old-dep==1"]},
                {"reference": "vendor/other", "requirements": ["old-dep==1"]},
            ],
        }
    ]
    manager.registry.save(document)
    monkeypatch.setattr(manager, "_project_python_executable", lambda: "python")
    monkeypatch.setattr(manager, "_python_environment_snapshot", lambda executable: {"old-dep": "1"})
    monkeypatch.setattr(
        manager, "_run_process", lambda *args, **kwargs: pytest.fail("The other owner still needs old-dep")
    )
    manager.install_package(next_package(source), update=True)
    assert manager.registry.installed_record("vendor/plugin")["python_requirements"] == []
    ledger = manager.registry.load()["python_dependencies"]
    assert ledger[0]["owners"] == [{"reference": "vendor/other", "requirements": ["old-dep==1"]}]


def test_update_failure_restores_released_python_environment(installed, monkeypatch):
    manager, source, root = installed
    document = manager.registry.load()
    document["installed"][0]["python_requirements"] = [{"name": "old-dep", "requirement": "old-dep==1"}]
    document["python_dependencies"] = [
        {
            "name": "old-dep",
            "managed": True,
            "baseline_version": "",
            "installed_version": "1",
            "owners": [{"reference": "vendor/plugin", "requirements": ["old-dep==1"]}],
        }
    ]
    manager.registry.save(document)
    environment = {"old-dep": "1"}
    monkeypatch.setattr(manager, "_project_python_executable", lambda: "python")
    monkeypatch.setattr(manager, "_python_environment_snapshot", lambda executable: dict(environment))
    monkeypatch.setattr(manager, "_run_process", lambda *args, **kwargs: environment.pop("old-dep"))
    monkeypatch.setattr(manager, "_restore_python_environment", lambda baseline, **kwargs: environment.update(baseline))

    def fail(*args, **kwargs):
        assert environment == {}
        raise OSError("publication failed")

    monkeypatch.setattr(manager.registry, "record_install", fail)
    with pytest.raises(OSError, match="publication failed"):
        manager.install_package(next_package(source), update=True)
    assert environment == {"old-dep": "1"}
    assert manager.registry.load() == document


def test_update_with_real_binary_archives_and_shared_cache(tmp_path):
    from infernux.engine import player_package_native

    player_package_native.set_test_backend(None)
    project = _project(tmp_path / "project")
    manager = PluginManager(str(project), runtime=True)
    source = _source(tmp_path / "source", "vendor/plugin")
    (source / "runtime").mkdir()
    (source / "runtime/note.txt").write_text("first", encoding="utf-8")
    package = tmp_path / "first.inxpkg"
    InxPackage.export_source(str(source), str(package))
    manager.install_package(str(package), install_dependencies=False)
    manager.install_package(next_package(source), update=True, install_dependencies=False)
    assert (project / "Packages/vendor/plugin/runtime/note.txt").read_text() == "second"
    installed = manager.registry.installed_record("vendor/plugin")
    assert InxPackage.inspect(manager._installed_archive_path(installed)).metadata["version"] == "2.0.0"
    assert installed["package_path"] == "packages/vendor/plugin/2.0.0.inxpkg"


def test_conflicting_python_upgrade_does_not_break_other_installed_package(installed, monkeypatch):
    from infernux.plugins.manager import _PipInstallEffect

    manager, source, root = installed
    document = manager.registry.load()
    document["python_dependencies"] = [
        {
            "name": "shared-dep",
            "managed": True,
            "baseline_version": "",
            "installed_version": "1",
            "owners": [{"reference": "vendor/other", "requirements": ["shared-dep==1"]}],
        }
    ]
    manager.registry.save(document)
    effect = _PipInstallEffect(
        {"shared-dep": "1"},
        {"shared-dep": "2"},
        ({"name": "shared-dep", "requirement": "shared-dep==2"},),
        ("pip",),
        "",
    )
    monkeypatch.setattr(manager, "_install_requirements", lambda *args, **kwargs: ((), effect))
    restored = []
    monkeypatch.setattr(manager, "_restore_python_environment", lambda baseline, **kwargs: restored.append(baseline))
    with pytest.raises(PackageConflictError, match="Python requirements conflict"):
        manager.install_package(next_package(source), update=True)
    assert restored == [{"shared-dep": "1"}]
    assert (root / "runtime/note.txt").read_text() == "first"
    assert manager.registry.load() == document
