"""Basis policies survive model Apply, publication and cooked-asset reload."""
import base64
import json
import time
from pathlib import Path

import numpy as np
import pytest

import infernux as inx
from Infernux.core.assets import AssetManager
from Infernux.core.asset_types import read_mesh_import_settings
from Infernux.lib import AssetRegistry


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fixture", ["model_authored_normals.obj", "model_uv_basis.gltf"])
def test_basis_apply_publishes_and_reloads(engine, tmp_path, monkeypatch, asynchronous, fixture):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / fixture
    source.parent.mkdir(parents=True)
    source.write_bytes((Path(__file__).resolve().parents[2] / "cpp/tests/fixtures" / fixture).read_bytes())
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    registry = AssetRegistry.instance()
    mesh = inx.Mesh.load_guid(imported.guid)
    node_names = [node["name"] for node in mesh.model_nodes]
    settings = read_mesh_import_settings(str(source))
    for normal_mode, tangent_mode in (("none", "import"), ("calculate", "calculate"),
                                     ("import", "none"), ("source_only", "source_only"), ("import", "import")):
        settings.normal_mode, settings.tangent_mode = normal_mode, tangent_mode
        if asynchronous:
            owner = AssetManager.begin_model_reimport(str(source), settings)
            deadline = time.monotonic() + 30
            while (result := AssetManager.poll_model_reimport(owner)) is None:
                assert time.monotonic() < deadline
                time.sleep(.002)
        else:
            result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        assert result, result.error
        assert read_mesh_import_settings(str(source)) == settings
        assert [node["name"] for node in mesh.model_nodes] == node_names
        before = mesh.vertex_buffer
        if normal_mode == "none":
            assert not np.any(before["normals"])
            assert not np.any(before["tangents"])
        else:
            np.testing.assert_allclose(np.linalg.norm(before["normals"], axis=1), 1, atol=1.e-5)
            if tangent_mode == "none":
                assert not np.any(before["tangents"])
        assert registry.reload_asset(imported.guid)
        for key, values in before.items():
            np.testing.assert_array_equal(mesh.vertex_buffer[key], values)


def test_old_basis_sidecar_migrates_on_reimport(engine, tmp_path):
    database = engine.get_asset_database()
    source = Path(database.assets_root) / tmp_path.name / "Legacy.obj"
    source.parent.mkdir(parents=True)
    source.write_bytes((Path(__file__).resolve().parents[2] / "cpp/tests/fixtures/model_authored_normals.obj").read_bytes())
    first = database.import_asset(str(source))
    assert first, first.error
    sidecar = Path(str(source) + ".meta")
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    for channel in ("normal", "tangent"):
        del document["metadata"][channel + "_mode"]
        document["metadata"]["generate_" + channel + "s"] = {"type": "bool", "value": False}
    sidecar.write_text(json.dumps(document), encoding="utf-8")
    result = database.reimport_asset(str(source))
    assert result, result.error
    assert result.guid == first.guid
    settings = read_mesh_import_settings(str(source))
    assert settings.normal_mode == settings.tangent_mode == "source_only"
    migrated = json.loads(sidecar.read_text(encoding="utf-8"))["metadata"]
    assert "generate_normals" not in migrated and "generate_tangents" not in migrated
    mesh = inx.Mesh.load_guid(first.guid)
    np.testing.assert_allclose(mesh.vertex_buffer["normals"], np.tile([1, 0, 0], (mesh.vertex_count, 1)))


def test_calculate_tangents_replaces_authored_uv_basis(engine, tmp_path):
    database = engine.get_asset_database()
    fixture = Path(__file__).resolve().parents[2] / "cpp/tests/fixtures/model_uv_basis.gltf"
    document = json.loads(fixture.read_text(encoding="utf-8"))
    payload = bytearray(base64.b64decode(document["buffers"][0]["uri"].split(",", 1)[1]))
    tangent_view = document["bufferViews"][document["accessors"][4]["bufferView"]]
    tangents = np.frombuffer(payload, dtype="<f4", count=12, offset=tangent_view["byteOffset"]).reshape(3, 4)
    # Still a valid orthogonal basis, but deliberately authored for UV1, not UV0.
    tangents[:, :3] = np.array([1, 1, -2]) / np.sqrt(6)
    document["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(payload).decode("ascii")
    source = Path(database.assets_root) / tmp_path.name / "AuthoredTangents.gltf"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(document), encoding="utf-8")
    imported = database.import_asset(str(source))
    assert imported, imported.error
    mesh = inx.Mesh.load_guid(imported.guid)
    authored = mesh.vertex_buffer["tangents"].copy()
    settings = read_mesh_import_settings(str(source))
    settings.tangent_mode = "calculate"
    result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert result, result.error
    calculated = mesh.vertex_buffer["tangents"]
    expected = np.array([-2, -3, 0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(calculated[:, :3], np.tile(expected, (mesh.vertex_count, 1)), atol=1.e-5)
    assert not np.allclose(authored[:, :3], calculated[:, :3])
    settings.tangent_mode = "source_only"
    result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert result, result.error
    np.testing.assert_allclose(mesh.vertex_buffer["tangents"], authored, atol=1.e-5)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_normal_weighting_apply_and_binary_reload(engine, tmp_path, monkeypatch, asynchronous):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / "Weighting.obj"
    source.parent.mkdir(parents=True)
    source.write_bytes((Path(__file__).resolve().parents[2] / "cpp/tests/fixtures/model_weighted_normals.obj").read_bytes())
    imported = database.import_asset(str(source))
    assert imported, imported.error
    mesh = inx.Mesh.load_guid(imported.guid)
    settings = read_mesh_import_settings(str(source))
    assert settings.tangent_algorithm == "mikktspace"
    for mode, ratio in (("unweighted", 1), ("area", 2), ("angle", 2), ("area_angle", 4)):
        settings.normal_weighting = mode
        if asynchronous:
            owner = AssetManager.begin_model_reimport(str(source), settings)
            deadline = time.monotonic() + 30
            while (result := AssetManager.poll_model_reimport(owner)) is None:
                assert time.monotonic() < deadline
                time.sleep(.002)
        else:
            result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        assert result, result.error
        assert read_mesh_import_settings(str(source)).normal_weighting == mode
        assert AssetRegistry.instance().reload_asset(imported.guid)
        vertices = mesh.vertex_buffer
        mask = np.all(vertices["positions"] == 0, axis=1)
        assert mask.any()
        expected = np.array([0, 1, ratio], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(vertices["normals"][mask], np.tile(expected, (mask.sum(), 1)), atol=1.e-5)
