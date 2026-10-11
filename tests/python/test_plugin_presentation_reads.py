from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess

import pytest

from infernux.plugins import PluginManager
from infernux.plugins.registry import PluginRegistry
from infernux.core.file_read_cache import read_model_frame


def publish(registry, *references):
    document = registry.load()
    document["packages"] = [{"reference": value, "name": value} for value in references]
    registry.save(document)


def test_ordinary_queries_reuse_unchanged_registry_across_frames(tmp_path, monkeypatch):
    registry = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    original = registry._load
    calls = []

    def load():
        calls.append("read")
        return original()

    monkeypatch.setattr(registry, "_load", load)
    for frame in range(2):
        with read_model_frame():
            assert registry.available()[0]["reference"] == "team/first"
            assert registry.find("team/first") is not None
            assert registry.installed() == ()
            assert registry.installed_record("team/first") is None
            with read_model_frame():
                assert registry.find("team/first") is not None
        assert len(calls) == 1
    # Standalone consumers get the same automatic parsing reuse.
    assert registry.find("team/first") is not None
    assert len(calls) == 1


def test_external_publication_is_coherent_for_frame_but_mutation_reads_are_fresh(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    with read_model_frame():
        assert registry.available()[0]["reference"] == "team/first"
        publish(peer, "team/second")
        # Frame labels agree with one another. An actual mutation does not
        # consume this presentation and can still see concurrent publication.
        assert registry.available()[0]["reference"] == "team/first"
        assert registry.load()["packages"][0]["reference"] == "team/second"
    with read_model_frame():
        assert registry.available()[0]["reference"] == "team/second"


def test_worker_queries_do_not_inherit_the_ui_snapshot(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    with ThreadPoolExecutor(max_workers=1) as worker:
        with read_model_frame():
            assert registry.available()[0]["reference"] == "team/first"
            publish(peer, "team/second")
            assert worker.submit(registry.available).result()[0]["reference"] == "team/second"
            assert registry.available()[0]["reference"] == "team/first"


def test_frame_exception_releases_presentation_state(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    with pytest.raises(RuntimeError, match="frame error"):
        with read_model_frame():
            assert registry.find("team/first") is not None
            raise RuntimeError("frame error")
    publish(peer, "team/second")
    assert registry.find("team/second") is not None


def test_synchronous_action_replaces_its_presentation_and_preserves_peer_changes(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    with read_model_frame():
        assert registry.find("team/first") is not None
        publish(peer, "team/peer")
        document = registry.load()
        document["packages"].append({"reference": "team/ours"})
        registry.save(document)
        assert {row["reference"] for row in registry.available()} == {"team/peer", "team/ours"}


def test_presentation_does_not_weaken_stale_write_rejection(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    old = registry.load()
    with read_model_frame():
        assert registry.find("team/first") is not None
        publish(peer, "team/peer")
        with pytest.raises(RuntimeError, match="changed outside"):
            registry.save(old)
    assert registry.find("team/peer") is not None


def test_manager_reuses_cache_owner_but_observes_new_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "shared"))
    manager = PluginManager(str(tmp_path / "project"), runtime=True)
    document = manager.registry.load()
    document["packages"] = [{"reference": "team/plugin", "version": "1.0.0"}]
    manager.registry.save(document)
    cache = manager._package_cache()
    try:
        assert manager.cached_reference_path("team/plugin") == ""
        from pathlib import Path
        archive = Path(cache.path("team/plugin", "1.0.0"))
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"newly published download")
        assert manager._package_cache() is cache
        assert manager.cached_reference_path("team/plugin") == str(archive)
        archive.unlink()
        assert manager.cached_reference_path("team/plugin") == ""
    finally:
        manager.shutdown()


def test_installed_metadata_projection_avoids_ownership_and_is_detached(tmp_path, monkeypatch):
    class OwnershipLedger(list):
        def __deepcopy__(self, memo):
            raise AssertionError("Presentation copied the asset ownership ledger")

    registry = PluginRegistry(str(tmp_path))
    record = {"reference": "team/plugin", "version": "1.0", "enabled": True,
              "pages": [{"id": "intro", "path": "plugin_pages/intro.md"}],
              "source": {"type": "local"}, "files": OwnershipLedger(), "control": {"guid": "control"}}
    document = {"packages": [], "installed": [record]}
    monkeypatch.setattr(registry, "_query_document", lambda: document)
    row, = registry.installed_metadata()
    assert "files" not in row and "control" not in row
    assert registry.is_installed("TEAM/PLUGIN")
    assert not registry.is_installed("team/missing")
    assert registry.catalog_counts() == {"available": 0, "installed": 1}
    row["source"]["type"] = "changed"
    row["pages"][0]["path"] = "changed"
    assert registry.installed_metadata()[0]["source"] == {"type": "local"}
    assert registry.installed_metadata()[0]["pages"][0]["path"] == "plugin_pages/intro.md"


def test_lightweight_catalog_queries_follow_external_publication(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    document = registry.load()
    document["packages"] = [{"reference": "team/plugin"}]
    document["installed"] = [{"reference": "team/plugin", "version": "1.0", "files": [],
                              "control": {"guid": "control-guid"}}]
    registry.save(document)
    with read_model_frame():
        assert registry.is_installed("team/plugin")
        assert registry.catalog_counts() == {"available": 1, "installed": 1}
        new = peer.load()
        new["installed"] = []
        new["packages"] = []
        peer.save(new)
        assert registry.is_installed("team/plugin")
        assert registry.installed_metadata()[0]["version"] == "1.0"
        assert registry.load()["installed"] == []
    with read_model_frame():
        assert not registry.is_installed("team/plugin")
        assert registry.installed_metadata() == ()
        assert registry.catalog_counts() == {"available": 0, "installed": 0}


def _installed_page_manager(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    root = project / "Packages/team/plugin"
    root.mkdir(parents=True)
    manager = PluginManager(str(project), runtime=True)
    document = manager.registry.load()
    document["installed"] = [{"reference": "team/plugin", "version": "1.0", "files": [],
                              "control": {"guid": "control-guid"}}]
    manager.registry.save(document)
    return manager, root


def test_page_asset_resolution_reuses_paths_and_observes_creation_deletion(tmp_path, monkeypatch):
    from infernux.plugins import manager as manager_module
    manager, root = _installed_page_manager(tmp_path)
    (root / "images").mkdir()
    image = root / "images/preview.png"
    original = manager_module.resolve_plugin_page_asset
    calls = []

    def resolve(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(manager_module, "resolve_plugin_page_asset", resolve)
    record, page = {"reference": "team/plugin"}, {"path": "plugin_pages/guide.md"}
    try:
        assert manager.content_asset_path(record, page, "../images/preview.png") == ""
        # Both project layouts are initially missing. Unchanged queries must
        # not repeat their physical resolution or containment checks.
        count = len(calls)
        assert manager.content_asset_path(record, page, "../images/preview.png") == ""
        assert len(calls) == count
        image.write_bytes(b"new image")
        assert manager.content_asset_path(record, page, "../images/preview.png") == str(image)
        count = len(calls)
        assert manager.content_asset_path(record, page, "../images/preview.png") == str(image)
        assert len(calls) == count
        image.unlink()
        assert manager.content_asset_path(record, page, "../images/preview.png") == ""
    finally:
        manager.shutdown()


def test_page_asset_cached_path_rejects_retargeted_directory_link(tmp_path):
    manager, root = _installed_page_manager(tmp_path)
    inside, outside = root / "local_images", tmp_path / "outside_images"
    inside.mkdir()
    outside.mkdir()
    for folder in (inside, outside):
        (folder / "preview.png").write_bytes(b"same bytes")
    link = root / "images"

    def create_link(target):
        if os.name == "nt":
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            command = ("New-Item -ItemType Junction -Path " + quote(link) + " -Target " + quote(target)
                       + " -ErrorAction Stop | Out-Null")
            subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, check=True)
            assert link.is_junction()
        else:
            os.symlink(target, link, target_is_directory=True)

    def remove_link():
        if os.name == "nt":
            assert link.is_junction()
            os.rmdir(link)
        else:
            link.unlink()

    try:
        create_link(inside)
        record, page = {"reference": "team/plugin"}, {"path": "plugin_pages/guide.md"}
        assert manager.content_asset_path(record, page, "../images/preview.png") == str(inside / "preview.png")
        remove_link()
        create_link(outside)
        assert manager.content_asset_path(record, page, "../images/preview.png") == ""
        remove_link()
        create_link(inside)
        assert manager.content_asset_path(record, page, "../images/preview.png") == str(inside / "preview.png")
    finally:
        manager.shutdown()


@pytest.mark.parametrize("source", ["..", "../", "../images", "/", "https://example.com/image.png"])
def test_page_asset_cache_rejects_non_file_sources(tmp_path, source):
    manager, root = _installed_page_manager(tmp_path)
    (root / "images").mkdir()
    try:
        assert manager.content_asset_path({"reference": "team/plugin"}, {"path": "plugin_pages/guide.md"}, source) == ""
    finally:
        manager.shutdown()
