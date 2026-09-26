"""Blend shapes survive model import, runtime publication and Cook reload."""

from pathlib import Path

import numpy as np
import pytest

import infernux as inx
from Infernux.core.assets import AssetManager
from Infernux.core.asset_types import read_mesh_import_settings
from Infernux.lib import AssetRegistry


def test_model_blend_shape_publication_and_authoring_switch(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    fixture = Path(__file__).resolve().parents[2] / "cpp/tests/fixtures/model_morph_target.gltf"
    source = Path(database.assets_root) / tmp_path.name / fixture.name
    source.parent.mkdir(parents=True)
    source.write_bytes(fixture.read_bytes())

    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    mesh = inx.Mesh.load_guid(imported.guid)
    assert mesh.is_readable is False
    with pytest.raises(RuntimeError, match="Read/Write"):
        mesh.get_morph_target(0)
    settings = read_mesh_import_settings(str(source))
    settings.is_readable = True
    readable = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert readable, readable.error
    assert mesh.is_readable is True
    assert mesh.morph_target_names == ("Smile",)
    target = mesh.get_morph_target(0)
    assert target["name"] == "Smile"
    assert target["default_weight"] == np.float32(0.4)
    assert target["position_deltas"].shape == (mesh.vertex_count, 3)
    np.testing.assert_allclose(target["position_deltas"][1], [0.0, 0.0, 0.25])
    assert target["normal_deltas"].shape == (0, 3)
    assert target["tangent_deltas"].shape == (0, 3)

    # The registry reload reads the same engine-owned cooked artifact rather
    # than reparsing glTF, and therefore proves Player-format preservation.
    assert AssetRegistry.instance().reload_asset(imported.guid)
    assert mesh.morph_target_names == ("Smile",)
    np.testing.assert_allclose(mesh.get_morph_target(0)["position_deltas"], target["position_deltas"])

    settings = read_mesh_import_settings(str(source))
    assert settings.import_blend_shapes
    settings.import_blend_shapes = False
    disabled = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert disabled, disabled.error
    assert mesh.morph_target_names == ()
    assert read_mesh_import_settings(str(source)).import_blend_shapes is False
