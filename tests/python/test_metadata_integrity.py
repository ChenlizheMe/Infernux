"""Native metadata input and published-snapshot integrity."""
from __future__ import annotations

import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("value", [-(2**31), -1, 0, 4, 2**31 - 1])
def test_metadata_int_preserves_legal_values(value):
    from infernux.lib import ResourceMeta
    meta = ResourceMeta()
    meta.deserialize_document({"metadata": {"count": {"type": "int", "value": value}}})
    assert meta.get_int("count") == value


@pytest.mark.parametrize("value", [-(2**31) - 1, 2**31, 2**32 + 4, 2**63 - 1, 2**64 - 1])
def test_metadata_int_rejects_overflow_without_publishing_partial_document(value):
    from infernux.lib import ResourceMeta
    meta = ResourceMeta()
    baseline = {"metadata": {"max_bones_per_vertex": {"type": "int", "value": 4}}}
    meta.deserialize_document(baseline)
    with pytest.raises(ValueError, match="range"):
        meta.deserialize_document({"metadata": {
            "a_valid_field": {"type": "string", "value": "must not commit"},
            "max_bones_per_vertex": {"type": "int", "value": value},
        }})
    assert meta.serialize_document() == baseline


@pytest.mark.parametrize("lookup", ["guid", "path"])
def test_asset_metadata_query_is_read_only(tmp_path, lookup):
    import infernux
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(tmp_path), lookup],
        env=dict(os.environ, PYTHONPATH=str(Path(infernux.__file__).resolve().parent.parent)),
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "METADATA_SNAPSHOT_OK" in result.stdout


def exercise(project, lookup):
    from infernux.core.assets import AssetManager
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode, ResourceMeta
    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        path = project / "Assets" / "Shared.txt"
        path.write_text("first source", encoding="utf-8")
        result = AssetManager.import_asset(str(path))
        assert result
        database = engine._engine.get_asset_database()
        read = (lambda: database.get_meta_by_guid(result.guid)) if lookup == "guid" else (
            lambda: database.get_meta_by_path(str(path)))
        snapshot = read()
        baseline = snapshot.serialize_document()
        generation = database.query_generation
        disk = Path(str(path) + ".meta").read_bytes()
        edited = snapshot.serialize_document()
        edited["metadata"]["injected"] = {"type": "string", "value": "must not enter snapshot"}
        with pytest.raises(AttributeError):
            snapshot.deserialize_document(edited)
        # Detached authoring documents still support explicit editing.
        independent = ResourceMeta()
        independent.deserialize_document(edited)
        assert independent.get_string("injected") == "must not enter snapshot"
        assert read().serialize_document() == baseline
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as worker:
            assert worker.submit(lambda: read().serialize_document()).result() == baseline
        assert database.query_generation == generation
        assert Path(str(path) + ".meta").read_bytes() == disk
        path.write_text("a longer replacement source", encoding="utf-8")
        assert AssetManager.reimport_asset(str(path))
        assert database.query_generation > generation
        assert read().serialize_document() != baseline
        assert snapshot.serialize_document() == baseline
        assert database.get_meta_by_guid("missing") is None
        assert database.get_meta_by_path(str(project / "Assets/missing.txt")) is None
        print("METADATA_SNAPSHOT_OK")
    finally:
        engine.exit()
    # The query owns its retained generation, independently of the database.
    assert snapshot.get_guid() == result.guid
    assert snapshot.serialize_document() == baseline


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), sys.argv[2])
