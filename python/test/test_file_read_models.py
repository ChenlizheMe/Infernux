"""Automatic read models must preserve external edits and command freshness."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from infernux.core.file_read_cache import FileReadCache, read_model_frame
from infernux.plugins import InxPackage, PluginManager
from infernux.plugins.registry import PluginRegistry


def test_same_size_atomic_replacement_with_preserved_mtime_invalidates(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("first", encoding="utf-8")
    stamp = path.stat()
    calls = []
    cache = FileReadCache()

    def prepare(observed):
        observed.watch(path)
        calls.append(1)
        return path.read_text(encoding="utf-8")

    assert cache.get("value", prepare) == "first"
    replacement = tmp_path / "next.txt"
    replacement.write_text("other", encoding="utf-8")
    os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    replacement.replace(path)
    assert cache.get("value", prepare) == "other"
    assert cache.get("value", prepare) == "other"
    assert len(calls) == 2


def test_in_place_copy_preserving_size_and_write_time_invalidates(tmp_path):
    path = tmp_path / "原样时间.txt"
    path.write_text("first", encoding="utf-8")
    stamp = path.stat()
    cache = FileReadCache()

    def prepare(observed):
        observed.watch(path)
        return path.read_text(encoding="utf-8")

    assert cache.get("value", prepare) == "first"
    path.write_text("other", encoding="utf-8")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert path.stat().st_size == stamp.st_size
    assert path.stat().st_mtime_ns == stamp.st_mtime_ns
    assert cache.get("value", prepare) == "other"


@pytest.mark.skipif(os.name != "nt", reason="Windows native path and change-time contract")
@pytest.mark.parametrize("extended", [False, True])
def test_windows_observations_support_long_unicode_paths(tmp_path, extended):
    root = tmp_path / ("资料" * 30) / ("文件" * 30) / ("目录" * 30)
    root.mkdir(parents=True)
    original = root / "拼图🧩.txt"
    original.write_text("first", encoding="utf-8")
    assert len(str(original)) > 260
    path = Path("\\\\?\\" + str(original)) if extended else original
    cache = FileReadCache()

    def prepare(observed):
        observed.watch(path)
        return path.read_text(encoding="utf-8") if path.exists() else None

    assert cache.get("unicode", prepare) == "first"
    assert cache.get("unicode", prepare) == "first"
    path.unlink()
    assert cache.get("unicode", prepare) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows junction contract")
def test_windows_junction_targets_and_retargeting_are_observed(tmp_path):
    import _winapi

    targets = [tmp_path / name for name in ("first", "second")]
    for target in targets:
        target.mkdir()
        (target / "guide.txt").write_text(target.name, encoding="utf-8")
    junction = tmp_path / "linked"
    _winapi.CreateJunction(str(targets[0]), str(junction))
    cache = FileReadCache()

    def prepare(observed):
        observed.watch_tree(junction)
        return (junction / "guide.txt").read_text(encoding="utf-8")

    try:
        assert cache.get("linked", prepare) == "first"
        (targets[0] / "guide.txt").write_text("edited", encoding="utf-8")
        assert cache.get("linked", prepare) == "edited"
        os.rmdir(junction)
        _winapi.CreateJunction(str(targets[1]), str(junction))
        assert cache.get("linked", prepare) == "second"
    finally:
        if junction.exists():
            os.rmdir(junction)
    assert targets[0].is_dir() and targets[1].is_dir()


def test_changes_during_preparation_are_not_retained(tmp_path):
    path = tmp_path / "changing.txt"
    path.write_text("before", encoding="utf-8")
    cache = FileReadCache()

    def prepare(observed):
        observed.watch(path)
        value = path.read_text(encoding="utf-8")
        if value == "before":
            path.write_text("after edit", encoding="utf-8")
        return value

    assert cache.get("file", prepare) == "before"
    assert cache.get("file", prepare) == "after edit"


def test_bounded_eviction_and_failed_reads_never_serve_previous_success(tmp_path):
    cache = FileReadCache(capacity=2)
    path = tmp_path / "doc.txt"
    path.write_text("valid", encoding="utf-8")
    calls = []

    def prepare(observed):
        observed.watch(path)
        calls.append(1)
        return path.read_text(encoding="utf-8")

    for key in ("a", "b", "a", "c", "b"):
        assert cache.get(key, prepare) == "valid"
    assert len(calls) == 4
    path.unlink()
    with pytest.raises(FileNotFoundError):
        cache.get("b", prepare)
    path.write_text("repaired", encoding="utf-8")
    assert cache.get("b", prepare) == "repaired"


def test_current_query_and_publication_bypass_active_presentation(tmp_path):
    path = tmp_path / "value.txt"
    path.write_text("first", encoding="utf-8")
    cache = FileReadCache()

    def prepare(observed):
        observed.watch(path)
        return path.read_text(encoding="utf-8")

    with read_model_frame():
        assert cache.get("value", prepare) == "first"
        path.write_text("modified", encoding="utf-8")
        assert cache.get("value", prepare) == "first"
        assert cache.get("value", prepare, current=True) == "modified"
        cache.clear()
        assert cache.get("value", prepare) == "modified"


def test_registry_results_do_not_share_mutable_catalog_nodes(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    value = registry.load()
    value["packages"] = [{"reference": "team/plugin", "source": {"tags": ["original"]}}]
    registry.save(value)
    for result in (registry.available()[0], registry.find("team/plugin")):
        result["source"]["tags"].append("caller edit")
        assert registry.find("team/plugin")["source"]["tags"] == ["original"]


def test_registry_observes_local_environment_and_corruption(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    registry.save(registry.load())
    assert registry._query_document()["python_installs"] == []
    environment = Path(registry.environment_path)
    document = json.loads(environment.read_text(encoding="utf-8"))
    document["python_installs"] = [{"command": "external"}]
    environment.write_text(json.dumps(document), encoding="utf-8")
    assert registry._query_document()["python_installs"] == [{"command": "external"}]
    Path(registry.path).write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        registry.available()


def test_plugin_documents_update_without_panel_cache_management(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    manager = PluginManager(str(tmp_path / "project"), runtime=True)
    record = {"reference": "team/docs", "intro": "No documents yet"}
    root = tmp_path / "project/Packages/team/docs/plugin_pages"
    try:
        assert manager.content_pages(record, locale="en")[0]["content"] == "No documents yet"
        root.mkdir(parents=True)
        path = root / "guide.md"
        path.write_text("# Guide\nfirst", encoding="utf-8")
        assert manager.content_pages(record, locale="en")[0]["content"] == "# Guide\nfirst"
        # Mutating a returned page cannot poison a future frame.
        manager.content_pages(record, locale="en")[0]["content"] = "caller"
        path.write_text("# Changed\nsecond revision", encoding="utf-8")
        assert manager.content_pages(record, locale="en")[0]["title"] == "Changed"
        localized = root / "guide.zh-CN.md"
        localized.write_text("# 中文\n内容", encoding="utf-8")
        assert manager.content_pages(record, locale="zh")[0]["content"] == "# 中文\n内容"
        assert manager.content_pages(record, locale="en")[0]["title"] == "Changed"
        nested = root / "nested"
        nested.mkdir()
        (nested / "extra.md").write_text("# Extra", encoding="utf-8")
        assert len(manager.content_pages(record, locale="en")) == 2
        (nested / "extra.md").rename(nested / "renamed.md")
        assert any(page["path"].endswith("renamed.md") for page in manager.content_pages(record, locale="en"))
        path.unlink()
        localized.unlink()
        assert len(manager.content_pages(record, locale="en")) == 1
        (nested / "renamed.md").unlink()
        assert manager.content_pages(record, locale="en")[0]["content"] == "No documents yet"
    finally:
        manager.shutdown()


def test_unchanged_documents_are_not_rediscovered_or_read(tmp_path, monkeypatch):
    import infernux.plugins.manager as module
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    manager = PluginManager(str(tmp_path), runtime=True)
    root = tmp_path / "Packages/team/docs/plugin_pages"
    root.mkdir(parents=True)
    (root / "guide.md").write_text("# Guide", encoding="utf-8")
    record = {"reference": "team/docs"}
    try:
        assert manager.content_pages(record, locale="en")[0]["title"] == "Guide"

        def unexpected(*args, **kwargs):
            pytest.fail("unchanged content was prepared again")

        monkeypatch.setattr(module, "read_plugin_pages", unexpected)
        monkeypatch.setattr(module, "discover_plugin_pages", unexpected)
        for _ in range(5):
            assert manager.content_pages(record, locale="en")[0]["title"] == "Guide"
    finally:
        manager.shutdown()


def test_archive_preview_observes_in_place_copy_preserving_time(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    manager = PluginManager(str(tmp_path / "project"), runtime=True)
    source = tmp_path / "source"
    pages = source / "plugin_pages"
    pages.mkdir(parents=True)
    (source / "inx_package.json").write_text(json.dumps({
        "reference": "team/docs", "name": "Docs", "version": "1.0.0",
    }), encoding="utf-8")
    guide = pages / "guide.md"
    guide.write_text("# Guide\naaaaa", encoding="utf-8")
    archive = Path(manager._package_cache().path("team/docs", "1.0.0"))
    archive.parent.mkdir(parents=True)
    InxPackage.export_source(str(source), str(archive))
    record = manager.registry.add_package(
        "team/docs", version="1.0.0", source={"type": "local", "location": str(archive)},
    )
    try:
        assert manager.content_pages(record, locale="en")[0]["content"] == "# Guide\naaaaa"
        stamp = archive.stat()
        guide.write_text("# Guide\nbbbbb", encoding="utf-8")
        replacement = tmp_path / "replacement.inxpkg"
        InxPackage.export_source(str(source), str(replacement))
        assert replacement.stat().st_size == stamp.st_size
        shutil.copyfile(replacement, archive)
        os.utime(archive, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        assert manager.content_pages(record, locale="en")[0]["content"] == "# Guide\nbbbbb"
    finally:
        manager.shutdown()
