"""Refresh distinguishes source files from model-owned derived assets."""
import json
from pathlib import Path

import pytest

from infernux.core.assets import AssetManager
from test_model_animation_clips import model
from test_model_embedded_textures import embed, records
from test_model_material_defaults import imported_model


def assert_unchanged_refresh(database):
    database.refresh()
    generation = database.query_generation, database.catalog_generation
    index = Path(database.asset_index_path)
    before = index.read_bytes(), index.stat().st_mtime_ns
    database.refresh()
    assert database.last_refresh_imported_paths == []
    assert (database.query_generation, database.catalog_generation) == generation
    assert (index.read_bytes(), index.stat().st_mtime_ns) == before
    assert database.last_refresh_index_build_ms == 0
    assert database.last_refresh_publish_ms == 0


def test_unchanged_animated_model_preserves_query_and_index(model):
    database, source, _ = model
    assert json.loads(database.get_meta_by_path(str(source)).get_string("model_animations"))
    assert_unchanged_refresh(database)


def prepare_embedded_texture(imported_model):
    _, document, source, database, _ = imported_model
    embed(document)
    source.write_text(json.dumps(document), encoding="utf-8")
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    record, = records(database, source)
    return database, source, record["guid"]


def test_unchanged_embedded_texture_preserves_query_and_index(imported_model):
    database, _, _ = prepare_embedded_texture(imported_model)
    assert_unchanged_refresh(database)


@pytest.mark.parametrize("change", ["missing", "invalid_header"])
def test_refresh_rebuilds_embedded_texture_through_its_owner(imported_model, change):
    database, source, guid = prepare_embedded_texture(imported_model)
    database.refresh()
    index = json.loads(Path(database.asset_index_path).read_text(encoding="utf-8"))
    record = next(entry for entry in index["entries"] if entry["guid"] == guid)
    artifact = Path(database.project_root) / record["artifact_path"]
    payload = artifact.read_bytes()
    if change == "missing":
        artifact.unlink()
    else:
        artifact.write_bytes(b"obsolete texture artifact")
    database.refresh()
    assert source in map(Path, database.last_refresh_imported_paths)
    assert artifact.read_bytes() == payload
    assert database.get_path_from_guid(guid)
    assert_unchanged_refresh(database)


def test_refresh_removes_children_when_their_owner_disappears(imported_model):
    database, source, guid = prepare_embedded_texture(imported_model)
    database.refresh()
    owner_guid = database.get_guid_from_path(str(source))
    source.unlink()
    Path(str(source) + ".meta").unlink()
    database.refresh()
    assert not database.get_path_from_guid(owner_guid)
    assert not database.get_path_from_guid(guid)
    assert_unchanged_refresh(database)
