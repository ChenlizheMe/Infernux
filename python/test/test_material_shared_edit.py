"""The rendering lesson edits one asset shared by scene renderer slots."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from Infernux.core.assets import AssetManager
from Infernux.engine.bootstrap_inspector._materials import _rebuild_material_entries
from Infernux.engine.ui.asset_details_renderer import _load_material
from Infernux.engine.ui.inspector_material import _get_inline_material_extra
from Infernux.engine.ui.project_file_ops import MATERIAL_TEMPLATE
from Infernux.lib import AssetRegistry


def _rgba(material):
    value = material.get_color("baseColor")
    return value.x, value.y, value.z, value.w


@pytest.mark.parametrize("color_type", [3, 7], ids=["v040-float4", "color"])
def test_asset_edits_reach_assigned_cube_and_cached_inline_inspector(engine, scene, tmp_path, monkeypatch, color_type):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    path = Path(database.assets_root) / tmp_path.name / "FirstCube.mat"
    path.parent.mkdir(parents=True)
    document = json.loads(MATERIAL_TEMPLATE.format(material_name="FirstCube"))
    document["properties"]["baseColor"]["type"] = color_type
    path.write_text(json.dumps(document), encoding="utf-8")
    imported = AssetManager.import_asset(str(path), database=database)
    assert imported, imported.error
    renderer = scene.create_game_object("Cube").add_component("MeshRenderer")
    renderer.set_material(0, imported.guid)
    entries = _rebuild_material_entries([(renderer, 1, (imported.guid,), ())])
    bound = entries[0]["material"]
    panel = SimpleNamespace(_inline_material_cache={})
    assert _get_inline_material_extra(panel, bound)["cached_data"]["properties"]["baseColor"]["value"] == [1, 1, 1, 1]

    _, extra = _load_material(str(path))
    authored = extra["native_mat"]
    assert authored is bound
    document = authored.serialize_document()
    document["shaders"]["fragment"] = {"guid": "", "shader_id": "Lit", "path_hint": ""}
    assert authored.deserialize_document(document)
    for color in ((0.9, 0.1, 0.2, 1), (0.2, 0.7, 0.1, 1)):
        authored.set_color("baseColor", color)
        assert renderer.get_effective_material(0) is authored
        assert _rgba(bound) == pytest.approx(color)
        cache = _get_inline_material_extra(panel, bound)
        assert cache["cached_data"]["properties"]["baseColor"]["value"] == pytest.approx(color)
        path.write_text(json.dumps(authored.serialize_document()), encoding="utf-8")
        assert AssetRegistry.instance().reload_asset(imported.guid)
        assert AssetRegistry.instance().load_material(str(path)) is bound
        assert _rgba(bound) == pytest.approx(color)
