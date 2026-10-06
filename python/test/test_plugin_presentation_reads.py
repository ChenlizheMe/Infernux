from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from infernux.plugins import PluginManager
from infernux.plugins.registry import PluginRegistry


def publish(registry, *references):
    document = registry.load()
    document["packages"] = [{"reference": value, "name": value} for value in references]
    registry.save(document)


def test_one_presentation_reads_registry_once_and_next_frame_reads_again(tmp_path, monkeypatch):
    registry = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    original = registry._load
    calls = []

    def load():
        calls.append("read")
        return original()

    monkeypatch.setattr(registry, "_load", load)
    for frame in range(2):
        with registry.presentation_reads():
            assert registry.available()[0]["reference"] == "team/first"
            assert registry.find("team/first") is not None
            assert registry.installed() == ()
            assert registry.installed_record("team/first") is None
            with registry.presentation_reads():
                assert registry.find("team/first") is not None
        assert len(calls) == frame + 1


def test_external_publication_is_coherent_for_frame_but_mutation_reads_are_fresh(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    with registry.presentation_reads():
        assert registry.available()[0]["reference"] == "team/first"
        publish(peer, "team/second")
        # Frame labels agree with one another. An actual mutation does not
        # consume this presentation and can still see concurrent publication.
        assert registry.available()[0]["reference"] == "team/first"
        assert registry.load()["packages"][0]["reference"] == "team/second"
    with registry.presentation_reads():
        assert registry.available()[0]["reference"] == "team/second"


def test_worker_queries_do_not_inherit_the_ui_snapshot(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    with ThreadPoolExecutor(max_workers=1) as worker:
        with registry.presentation_reads():
            assert registry.available()[0]["reference"] == "team/first"
            publish(peer, "team/second")
            assert worker.submit(registry.available).result()[0]["reference"] == "team/second"
            assert registry.available()[0]["reference"] == "team/first"


def test_frame_exception_releases_presentation_state(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    with pytest.raises(RuntimeError, match="frame error"):
        with registry.presentation_reads():
            assert registry.find("team/first") is not None
            raise RuntimeError("frame error")
    publish(peer, "team/second")
    assert registry.find("team/second") is not None


def test_synchronous_action_replaces_its_presentation_and_preserves_peer_changes(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    peer = PluginRegistry(str(tmp_path))
    publish(registry, "team/first")
    with registry.presentation_reads():
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
    with registry.presentation_reads():
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
