"""Source materials follow model selection/reimport without erasing author edits."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from Infernux.core.assets import AssetManager


@pytest.fixture
def imported_model(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / "MaterialSelection.gltf"
    fixture = Path(__file__).resolve().parents[2] / "cpp/tests/fixtures/model_hierarchy.gltf"
    document = json.loads(fixture.read_text(encoding="utf-8"))
    document["materials"] = [
        {"name": name, "pbrMetallicRoughness": {"baseColorFactor": color}}
        for name, color in (("Red", [1, 0, 0, 1]), ("Green", [0, 1, 0, 1]))
    ]
    document["meshes"].append(copy.deepcopy(document["meshes"][0]))
    for index, mesh in enumerate(document["meshes"]):
        mesh["primitives"][0]["material"] = index
    document["nodes"][3]["mesh"] = 1
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(document), encoding="utf-8")
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    go = scene.create_game_object("Model material defaults")
    renderer = go.add_component("MeshRenderer")._require_cpp_component()
    renderer.set_mesh_asset_guid(imported.guid)
    yield renderer, document, source, database, scene
    scene.destroy_game_object(go)


def color(renderer, slot=0):
    value = renderer.get_material(slot).get_color("baseColor")
    return [value.x, value.y, value.z, value.w]


def select(renderer, index, mode):
    if mode == "submesh":
        renderer.submesh_index = index
    else:
        document = renderer.serialize_document()
        if index >= 0:
            document["nodeGroup"] = index
        else:
            document.pop("nodeGroup", None)
        assert renderer.deserialize_document(document)


@pytest.mark.parametrize("mode", ["node", "submesh"])
def test_model_material_selection_and_scene_restore(imported_model, mode):
    renderer, _, _, _, scene = imported_model
    assert renderer.material_count == 2
    np.testing.assert_allclose(color(renderer, 0), [1, 0, 0, 1])
    np.testing.assert_allclose(color(renderer, 1), [0, 1, 0, 1])
    assert renderer.serialize_document()["materials"] == [None, None]
    for index in (1, 0, 1):
        select(renderer, index, mode)
        assert renderer.material_count == 1
        np.testing.assert_allclose(color(renderer), [0, 1, 0, 1] if index else [1, 0, 0, 1])
    saved = renderer.serialize_document()
    assert saved["materials"] == [None]
    restored = scene.create_game_object("Restored").add_component("MeshRenderer")._require_cpp_component()
    saved["component_id"] = restored.component_id
    assert restored.deserialize_document(saved)
    np.testing.assert_allclose(color(restored), [0, 1, 0, 1])
    select(restored, -1, mode)
    assert restored.material_count == 2
    np.testing.assert_allclose(color(restored, 0), [1, 0, 0, 1])
    np.testing.assert_allclose(color(restored, 1), [0, 1, 0, 1])


@pytest.mark.parametrize("override", ["none", "edit", "assign_same", "guid", "inline_document"])
def test_reimport_updates_only_unmodified_source_materials(imported_model, override):
    renderer, source_document, source, database, scene = imported_model
    renderer.submesh_index = 1
    expected = [0, 0, 1, 1]
    if override == "edit":
        renderer.get_material(0).set_color("baseColor", [1, 0, 1, 1])
        expected = [1, 0, 1, 1]
    elif override == "assign_same":
        renderer.set_material(0, renderer.get_material(0))
        expected = [0, 1, 0, 1]
    elif override == "guid":
        target = source.with_suffix(".mat")
        material = renderer.get_material(0).serialize_document()
        target.write_text(json.dumps(material), encoding="utf-8")
        result = AssetManager.import_asset(str(target), database=database)
        assert result, result.error
        renderer.set_material(0, result.guid)
        expected = [0, 1, 0, 1]
    elif override == "inline_document":
        # Existing scene snapshots are authored data; never infer their origin
        # from names or file paths and silently replace them on reimport.
        saved = renderer.serialize_document()
        saved["materials"] = [{"material": renderer.get_material(0).serialize_document()}]
        assert renderer.deserialize_document(saved)
        expected = [0, 1, 0, 1]

    saved = renderer.serialize_document()
    assert (saved["materials"] == [None]) == (override == "none")
    restored = scene.create_game_object("Restored override").add_component("MeshRenderer")._require_cpp_component()
    saved["component_id"] = restored.component_id
    assert restored.deserialize_document(saved)
    source_document["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"] = [0, 0, 1, 1]
    source.write_text(json.dumps(source_document), encoding="utf-8")
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    np.testing.assert_allclose(color(renderer), expected)
    np.testing.assert_allclose(color(restored), expected)
    assert (renderer.serialize_document()["materials"] == [None]) == (override == "none")


def test_clearing_model_keeps_existing_inline_appearance(imported_model):
    renderer, _, _, _, _ = imported_model
    renderer.clear_mesh_asset()
    saved = renderer.serialize_document()
    assert "meshAssetGuid" not in saved
    assert all("material" in slot for slot in saved["materials"])


def test_unchanged_model_materials_keep_runtime_instances(imported_model):
    renderer, _, source, database, _ = imported_model
    before = [renderer.get_material(index) for index in range(2)]
    result = AssetManager.reimport_asset(str(source), database=database)
    assert result, result.error
    assert [renderer.get_material(index) for index in range(2)] == before
    assert renderer.serialize_document()["materials"] == [None, None]
