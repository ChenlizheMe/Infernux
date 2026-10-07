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


@pytest.mark.parametrize("environment", (None, {}, {"teamlib": ""}))
def test_dependency_publication_requires_a_resolved_version(tmp_path, environment):
    registry = PluginRegistry(str(tmp_path))
    registry.save(registry.load())
    before = {path: Path(path).read_bytes() for path in (registry.path, registry.lock_path)}
    with pytest.raises(ValueError, match="Resolved Python dependency version is missing: teamlib"):
        registry.record_python_install(
            syntax="pip install teamlib>=1,<2", command=(), output="", owner="@project",
            requirements=["teamlib>=1,<2"],
            dependency_requirements=[{"name": "teamlib", "requirement": "teamlib>=1,<2"}],
            python_environment=environment,
        )
    assert {path: Path(path).read_bytes() for path in before} == before
    assert registry.load()["python_dependencies"] == []


def test_reusing_a_distribution_pins_it_without_taking_uninstall_ownership(tmp_path):
    registry = PluginRegistry(str(tmp_path))
    requirement = {"name": "teamlib", "requirement": "teamlib>=1,<2"}
    control = {"guid": "62a3482c971c469485877f2e5b41a2b5", "path_hint": "Packages/team/plugin/inx_package.json"}
    registry.record_install(
        {"reference": "team/plugin", "version": "1.0"}, files=(), control=control,
        python_requirements=[requirement], python_environment={"teamlib": "1.5"},
    )
    dependency = registry.load()["python_dependencies"][0]
    assert dependency["installed_version"] == "1.5"
    assert dependency["managed"] is False
    assert dependency["baseline_version"] == ""
    assert registry.python_release_plan("team/plugin")[0]["managed"] is False
    # An asset-only package update keeps dependencies untouched, with no
    # environment query or fabricated installation event.
    registry.record_install(
        {"reference": "team/plugin", "version": "1.1"}, files=(), control=control,
        python_requirements=[requirement],
    )
    assert registry.load()["python_dependencies"][0] == dependency


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
