from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

from infernux.core.document_store import DocumentStore
from infernux.plugins.registry import PluginRegistry


def test_registry_and_lock_use_the_shared_document_writer(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    document = registry.load()
    registry.save(document)
    store = DocumentStore.instance()
    for path in (registry.path, registry.lock_path):
        metrics = store.get_metrics(path)
        assert metrics.latest_succeeded_generation > 0
        with open(path, encoding="utf-8") as stream:
            assert isinstance(json.load(stream), dict)


def test_external_checkout_change_cannot_be_overwritten_by_a_stale_snapshot(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    registry.save(registry.load())
    snapshot = registry.load()
    external = dict(snapshot)
    external["packages"] = [{"reference": "team/new", "source": {"type": "local", "location": "Packages/team/new"}}]
    external_bytes = (json.dumps(external) + "\n").encode("utf-8")
    Path(registry.path).write_bytes(external_bytes)
    with pytest.raises(RuntimeError, match="changed outside"):
        registry.save(snapshot)
    assert Path(registry.path).read_bytes() == external_bytes
    registry.save(registry.load())


def test_transaction_rollback_does_not_overwrite_a_concurrent_registry_change(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    before = registry.load()
    registry.save(before)
    changed = registry.load()
    changed["packages"] = [{"reference": "team/ours"}]
    registry.save(changed)
    registry.restore(before)
    assert registry.load()["packages"] == []
    Path(registry.path).write_text(json.dumps(dict(changed)), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed outside"):
        registry.restore(before)
    assert registry.load()["packages"] == [{"reference": "team/ours"}]


def test_local_python_environment_cannot_replace_the_shared_resolved_version(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    document = registry.load()
    document["python_dependencies"] = [{
        "name": "teamlib", "managed": True, "baseline_version": "0.9",
        "installed_version": "1.5",
        "owners": [{"reference": "@project", "requirements": ["teamlib>=1,<2"]}],
    }]
    registry.save(document)
    shared = Path(registry.path).read_bytes()
    lock = Path(registry.lock_path).read_bytes()
    environment = json.loads(Path(registry.environment_path).read_bytes())
    environment["dependencies"]["teamlib"]["installed_version"] = "1.8"
    Path(registry.environment_path).write_text(json.dumps(environment), encoding="utf-8")
    loaded = registry.load()
    assert loaded["python_dependencies"][0]["installed_version"] == "1.5"
    registry.save(loaded)
    assert Path(registry.path).read_bytes() == shared
    assert Path(registry.lock_path).read_bytes() == lock


@pytest.mark.skipif(sys.platform != "win32", reason="Windows delete-sharing contract")
@pytest.mark.parametrize("attribute", ("path", "lock_path"))
def test_registry_publication_survives_a_brief_windows_reader(tmp_path, attribute):
    registry = PluginRegistry(str(tmp_path))
    document = registry.load()
    registry.save(document)
    # Python's ordinary Windows file reader does not share delete access.
    # Exercise the actual OS sharing violation, not a mocked PermissionError.
    stream = open(getattr(registry, attribute), encoding="utf-8")
    release = threading.Timer(0.1, stream.close)
    release.start()
    try:
        registry.save(document)
    finally:
        release.join()
        stream.close()
    assert registry.load() == document
