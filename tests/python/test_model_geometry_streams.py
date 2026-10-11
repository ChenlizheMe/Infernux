"""Every public mesh channel follows the same node-local or whole-asset geometry."""
import base64
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from infernux.core.assets import AssetManager
from infernux.core.asset_types import read_mesh_import_settings
from model_test_support import remove_model_test_folder


def triangle_document(transform, *, instances=1):
    payload = struct.pack("<9f6f3H", 0, 0, 0, 1, 0, 0, 0, 1, 0,
                          0, 0, 1, 0, 0, 1, 0, 1, 2)
    return {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Root", "children": list(range(1, instances + 1)), **transform}]
        + [{"name": f"Surface{i}", "mesh": 0, "translation": [i * 3, 0, 0]}
           for i in range(instances)],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2}]}],
        "buffers": [{"byteLength": len(payload), "uri": "data:application/octet-stream;base64,"
                     + base64.b64encode(payload).decode("ascii")}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                        {"buffer": 0, "byteOffset": 36, "byteLength": 24},
                        {"buffer": 0, "byteOffset": 60, "byteLength": 6}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0, 0, 0], "max": [1, 1, 0]},
                      {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
                      {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}],
    }


def assert_consistent_streams(renderer):
    positions = np.asarray(renderer.get_positions(), dtype=float)
    normals = np.asarray(renderer.get_normals(), dtype=float)
    uvs = np.asarray(renderer.get_uvs(), dtype=float)
    tangents = np.asarray(renderer.get_tangents(), dtype=float)
    indices = np.asarray(renderer.get_indices(), dtype=int)
    assert positions.shape == normals.shape == (renderer.vertex_count, 3)
    assert tangents.shape == (len(positions), 4)
    assert uvs.shape == (len(positions), 2)
    assert len(indices) == renderer.index_count and len(indices) % 3 == 0
    for a, b, c in indices.reshape(-1, 3):
        duv1, duv2 = uvs[b] - uvs[a], uvs[c] - uvs[a]
        determinant = duv1[0] * duv2[1] - duv1[1] * duv2[0]
        assert abs(determinant) > .5
        edge1, edge2 = positions[b] - positions[a], positions[c] - positions[a]
        tangent = (edge1 * duv2[1] - edge2 * duv1[1]) / determinant
        bitangent = (edge2 * duv1[0] - edge1 * duv2[0]) / determinant
        for vertex in (a, b, c):
            normal = normals[vertex]
            expected = tangent - normal * np.dot(normal, tangent)
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(tangents[vertex, :3], expected, atol=1e-5)
            assert abs(tangents[vertex, 3]) == pytest.approx(1)
            assert np.dot(np.cross(normal, expected) * tangents[vertex, 3], bitangent) > 0
    return positions, tangents


@pytest.mark.parametrize("transform", [
    {},
    {"rotation": [0, 0, 2 ** -.5, 2 ** -.5]},
    {"rotation": [2 ** -.5, 0, 0, 2 ** -.5], "scale": [2, 3, .5]},
    {"rotation": [0, 0, 2 ** -.5, 2 ** -.5], "scale": [-2, 3, 1]},
], ids=["identity", "rotated", "nonuniform", "mirrored"])
@pytest.mark.parametrize("instances", [1, 2])
@pytest.mark.parametrize("node_local", [False, True], ids=["whole_asset", "node_local"])
def test_imported_model_channels_share_geometry(engine, scene, tmp_path, transform, instances, node_local):
    database = engine.get_asset_database()
    folder = Path(database.assets_root) / tmp_path.name
    folder.mkdir()
    source = folder / "Triangle.gltf"
    source.write_text(json.dumps(triangle_document(transform, instances=instances)), encoding="utf-8")
    objects = []
    try:
        imported = database.import_asset(str(source))
        assert imported, imported.error
        settings = read_mesh_import_settings(str(source))
        settings.is_readable = True
        ready = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        assert ready, ready.error
        if node_local:
            model = scene.create_from_model(imported.guid)
            objects.append(model)
            renderers = []
            def visit(node):
                component = node.get_component("MeshRenderer")
                if component is not None:
                    renderers.append(component)
                for child in node.get_children():
                    visit(child)
            visit(model)
            assert len(renderers) == instances
        else:
            obj = scene.create_game_object("Whole asset geometry")
            objects.append(obj)
            renderers = [obj.add_component("MeshRenderer")]
            renderers[0].set_mesh_asset_guid(imported.guid)
        for renderer in renderers:
            positions, _ = assert_consistent_streams(renderer)
            if node_local:
                assert renderer.serialize_document()["modelNodePath"][-1].startswith("Surface")
                # Node transforms remain in the hierarchy, not baked into query channels.
                assert np.min(positions[:, 0]) == pytest.approx(0)
                assert np.max(positions[:, 0]) == pytest.approx(1)
    finally:
        for obj in objects:
            scene._remove_game_object_immediately(obj)
        remove_model_test_folder(database, folder)


def test_inline_model_channels_share_geometry(scene):
    obj = scene.create_game_object("Inline triangle geometry")
    try:
        renderer = obj.add_component("MeshRenderer")
        renderer._require_cpp_component().set_inline_mesh_data(
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), None,
            np.asarray([[0, 0], [1, 0], [0, 1]], np.float32),
            np.asarray([0, 1, 2], np.uint32))
        assert_consistent_streams(renderer)
    finally:
        scene._remove_game_object_immediately(obj)
