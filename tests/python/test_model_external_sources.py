"""Composite model files resolve beside the source, independently of process cwd."""
import base64
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import infernux as inx
from infernux.core.asset_types import read_mesh_import_settings
from infernux.core.assets import AssetManager
from infernux.lib import AssetDependencyGraph, AssetRegistry
from model_test_support import remove_model_test_folder


@pytest.fixture
def model_folder(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    cwd = Path.cwd()
    try:
        yield folder, database
        assert Path.cwd() == cwd
    finally:
        remove_model_test_folder(database, folder)


def write_triangle(source, *, external, width=2):
    blob = struct.pack("<9f3H", 0, 0, 0, width, 0, 0, 0, 1, 0, 0, 1, 2)
    buffer_path = source.parent / "buffers" / "triangle.bin"
    buffer_path.parent.mkdir(exist_ok=True)
    buffer_path.write_bytes(blob)
    uri = "buffers/triangle.bin" if external else "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob), "uri": uri}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                        {"buffer": 0, "byteOffset": 36, "byteLength": 6}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0, 0, 0], "max": [width, 1, 0]},
                      {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}],
        "materials": [{"name": "Authored", "pbrMetallicRoughness": {"baseColorFactor": [.2, .4, .6, 1]}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "nodes": [{"name": "Triangle", "mesh": 0}], "scenes": [{"nodes": [0]}], "scene": 0,
    }
    source.write_text(json.dumps(document), encoding="utf-8")
    return buffer_path


@pytest.mark.parametrize("name", ["ascii", "模型 空间🧩", "long"], ids=["ascii", "unicode", "long"])
@pytest.mark.parametrize("external", [False, True])
def test_gltf_geometry_material_and_artifact_reload_use_model_directory(model_folder, name, external):
    folder, database = model_folder
    source = folder / name / "Triangle.gltf"
    if name == "long":
        while len(str(source)) <= 280:
            source = source.parent / ("目录-" + "x" * 30) / source.name
        assert len(str(source)) > 260
    source.parent.mkdir(parents=True)
    write_triangle(source, external=external)
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    settings = read_mesh_import_settings(str(source))
    settings.is_readable = True
    assert AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    mesh = inx.Mesh.load_guid(imported.guid)
    registry = AssetRegistry.instance()
    for _ in range(2):
        assert mesh.vertex_count == mesh.index_count == 3
        np.testing.assert_allclose(mesh.vertex_buffer["positions"].max(axis=0), [2, 1, 0])
        material = registry.load_mesh(str(source)).get_material_slot_data()[0]
        assert material["base_color"] == pytest.approx([.2, .4, .6, 1])
        assert registry.reload_asset(imported.guid)


@pytest.mark.parametrize("name", ["ascii", "材质 相对路径🧩"])
def test_obj_reads_mtl_values_and_texture_guid_beside_source(model_folder, name):
    folder, database = model_folder
    source = folder / name / "Ship.obj"
    source.parent.mkdir()
    texture = source.with_name("船身.png")
    Image.new("RGB", (2, 2), (40, 80, 120)).save(texture)
    image = AssetManager.import_asset(str(texture), database=database)
    assert image, image.error
    source.with_suffix(".mtl").write_text(
        "newmtl Hull\nKd 0.2 0.4 0.6\nmap_Kd 船身.png\n", encoding="utf-8")
    source.write_text("mtllib Ship.mtl\no Hull\nv 0 0 0\nv 2 0 0\nv 0 1 0\n"
                      "vt 0 0\nvt 1 0\nvt 0 1\nusemtl Hull\nf 1/1 2/2 3/3\n", encoding="utf-8")
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    registry = AssetRegistry.instance()
    mesh = registry.load_mesh(str(source))
    for _ in range(2):
        material = next(item for item in mesh.get_material_slot_data() if item["source_id"] == "material/Hull")
        assert material["base_color"] == pytest.approx([.2, .4, .6, 1])
        assert material["base_color_texture_guid"] == image.guid
        assert image.guid in AssetDependencyGraph.instance().get_dependencies(imported.guid)
        assert mesh.get_model_nodes()[0]["name"] == source.name
        assert registry.reload_asset(imported.guid)


def test_external_buffer_reimport_updates_geometry_and_rejects_missing_source(model_folder):
    folder, database = model_folder
    source = folder / "Triangle.gltf"
    buffer_path = write_triangle(source, external=True)
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    settings = read_mesh_import_settings(str(source))
    settings.is_readable = True
    assert AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    mesh = inx.Mesh.load_guid(imported.guid)
    np.testing.assert_allclose(mesh.vertex_buffer["positions"].max(axis=0), [2, 1, 0])
    source_bytes = source.read_bytes()
    buffer_path.write_bytes(struct.pack("<9f3H", .5, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 2))
    assert AssetManager.reimport_asset(str(source), database=database)
    assert source.read_bytes() == source_bytes
    assert mesh.vertex_buffer["positions"][:, 0].sum() == pytest.approx(2.5)
    write_triangle(source, external=True, width=3)
    assert AssetManager.reimport_asset(str(source), database=database)
    np.testing.assert_allclose(mesh.vertex_buffer["positions"].max(axis=0), [3, 1, 0])
    buffer_path.unlink()
    generation = mesh.generation
    rejected = AssetManager.reimport_asset(str(source), database=database)
    assert not rejected
    assert mesh.generation == generation
    np.testing.assert_allclose(mesh.vertex_buffer["positions"].max(axis=0), [3, 1, 0])
