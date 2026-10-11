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


def test_shared_file_dependencies_are_observed_once_per_panel_submission(tmp_path, monkeypatch):
    from infernux.core import file_read_cache as readers

    path = tmp_path / 'shared.txt'
    path.write_text('first', encoding='utf-8')
    probes = []
    original = readers._file_probe

    def instrumented(name):
        probe = original(name)
        def observe():
            probes.append(name)
            return probe()
        return observe

    monkeypatch.setattr(readers, '_file_probe', instrumented)
    caches = [FileReadCache(), FileReadCache()]
    def prepare(observed):
        observed.watch(path)
        return path.read_text(encoding='utf-8')

    for cache in caches:
        assert cache.get('value', prepare) == 'first'
    probes.clear()
    with read_model_frame():
        assert [cache.get('value', prepare) for cache in caches] == ['first', 'first']
    assert len(probes) == 1
    path.write_text('next revision', encoding='utf-8')
    with read_model_frame():
        assert [cache.get('value', prepare) for cache in caches] == ['next revision', 'next revision']


def test_explicit_current_reads_and_publication_bypass_shared_file_observations(tmp_path):
    path = tmp_path / 'shared.txt'
    path.write_text('first', encoding='utf-8')
    first, second = FileReadCache(), FileReadCache()
    def prepare(observed):
        observed.watch(path)
        return path.read_text(encoding='utf-8')

    assert first.get('value', prepare) == second.get('value', prepare) == 'first'
    with read_model_frame():
        assert first.get('value', prepare) == 'first'
        path.write_text('updated', encoding='utf-8')
        assert second.get('value', prepare, current=True) == 'updated'
        first.clear()
        assert first.get('value', prepare) == 'updated'
        assert second.get('value', prepare) == 'updated'


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


@pytest.mark.skipif(os.name != "nt", reason="Windows native query path reuse")
def test_warm_file_observation_reuses_query_paths_but_requeries_state(tmp_path, monkeypatch):
    from infernux.core import _windows_file_observation as windows

    path = tmp_path / "每帧🧩.txt"
    path.write_text("first", encoding="utf-8")
    cache = FileReadCache()

    def prepare(observed):
        observed.watch(path)
        return path.read_text(encoding="utf-8")

    assert cache.get("doc", prepare) == "first"
    create_buffer = windows.ctypes.create_unicode_buffer
    allocations = []

    def counted_buffer(*args, **kwargs):
        allocations.append(1)
        return create_buffer(*args, **kwargs)

    monkeypatch.setattr(windows.ctypes, "create_unicode_buffer", counted_buffer)
    for _ in range(20):
        with read_model_frame():
            assert cache.get("doc", prepare) == "first"
    assert allocations == []
    path.write_text("changed", encoding="utf-8")
    assert cache.get("doc", prepare) == "changed"
    assert len(allocations) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows concurrent native metadata queries")
def test_one_windows_query_path_can_be_observed_concurrently(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from infernux.core._windows_file_observation import FileProbe, file_stamp

    path = tmp_path / "并发🧩.txt"
    path.write_text("content", encoding="utf-8")
    probe = FileProbe(str(path))
    expected = file_stamp(str(path))
    with ThreadPoolExecutor(max_workers=4) as workers:
        assert list(workers.map(lambda _: probe(), range(64))) == [expected] * 64
    path.unlink()
    assert probe() is None


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


@pytest.mark.parametrize("query", ["available", "find", "installed", "installed_metadata", "installed_record"])
def test_all_registry_read_projections_detach_json_containers(tmp_path, query):
    registry = PluginRegistry(str(tmp_path))
    document = registry.load()
    record = {
        "reference": "team/plugin", "version": "1.0",
        "source": {"type": "github", "nested": [{"values": [None, True, 7, 2.5, "中文"]}]},
        "files": [{"guid": "11111111111111111111111111111111", "owned": True}],
        "control": {"guid": "22222222222222222222222222222222", "owned": True},
    }
    document["packages"] = [record]
    document["installed"] = [record]
    registry.save(document)

    def read():
        method = getattr(registry, query)
        return method("team/plugin") if query in {"find", "installed_record"} else method()[0]

    with read_model_frame():
        first = read()
        assert first["source"]["nested"][0]["values"] == [None, True, 7, 2.5, "中文"]
        first["source"]["nested"][0]["values"][1] = "caller mutation"
        first["source"]["nested"].append({"values": []})
        if "files" in first:
            first["files"][0]["owned"] = False
            first["control"]["guid"] = "caller"
        assert read()["source"]["nested"] == record["source"]["nested"]
        if query != "installed_metadata":
            assert read()["files"] == record["files"]
            assert read()["control"] == record["control"]
        else:
            assert "files" not in read() and "control" not in read()


def test_download_lookup_does_not_copy_catalog_payload_and_observes_publication(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERNUX_PACKAGE_CACHE_ROOT", str(tmp_path / "cache"))
    manager = PluginManager(str(tmp_path / "project"), runtime=True)
    peer = PluginRegistry(str(tmp_path / "project"))
    try:
        value = peer.load()
        value["packages"] = [{
            "reference": "team/plugin", "version": "1.0",
            "source": {"cache_location": "packages/team/plugin/1.0/package.inxpkg"},
            "pages": [{"languages": ["en", "zh"], "path": "guide.md"}],
        }]
        peer.save(value)
        archive = tmp_path / "cache/packages/team/plugin/1.0/package.inxpkg"
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"download")
        assert Path(manager.cached_reference_path("team/plugin")) == archive

        def unexpected_copy(*args, **kwargs):
            pytest.fail("Download lookup copied the full catalog record")

        # Ordinary APIs still return independent mutable copies. This internal
        # query only needs two immutable strings, regardless of payload size.
        monkeypatch.setattr(manager.registry, "find", unexpected_copy)
        with read_model_frame():
            assert Path(manager.cached_reference_path("team/plugin")) == archive
            revised = peer.load()
            revised["packages"][0]["source"]["cache_location"] = "packages/team/plugin/2.0/package.inxpkg"
            revised["packages"][0]["version"] = "2.0"
            peer.save(revised)
            assert Path(manager.cached_reference_path("team/plugin")) == archive
        # Peer publication is observed at the next normal panel submission.
        with read_model_frame():
            assert manager.cached_reference_path("team/plugin") == ""
        next_archive = tmp_path / "cache/packages/team/plugin/2.0/package.inxpkg"
        next_archive.parent.mkdir(parents=True)
        next_archive.write_bytes(b"new download")
        assert Path(manager.cached_reference_path("team/plugin")) == next_archive
        next_archive.unlink()
        assert manager.cached_reference_path("team/plugin") == ""
    finally:
        manager.shutdown()


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
