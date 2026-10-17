"""Imported mesh selections must drive both render bounds and collision."""

import numpy as np
import pytest

import infernux as inx
from infernux.lib import Physics, Vector3


@pytest.mark.parametrize("raw_native,selection,convex", [(False, "submesh", False), (True, "node", True)])
def test_model_selection_collision_tracks_only_selected_geometry(scene, convex, selection, raw_native):
    cube = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )
    triangles = np.array(
        [
            0,
            2,
            1,
            0,
            3,
            2,
            4,
            5,
            6,
            4,
            6,
            7,
            0,
            1,
            5,
            0,
            5,
            4,
            3,
            7,
            6,
            3,
            6,
            2,
            0,
            4,
            7,
            0,
            7,
            3,
            1,
            2,
            6,
            1,
            6,
            5,
        ],
        dtype=np.uint32,
    )
    mesh = inx.Mesh.from_data(
        np.concatenate([cube, cube + np.array([10, 0, 0], dtype=np.float32)]),
        np.concatenate([triangles, triangles + 8]),
        submeshes=[
            dict(index_start=i * 36, index_count=36, vertex_start=i * 8, vertex_count=8, node_group=i) for i in range(2)
        ],
        name="Two separate model nodes",
    )
    go = scene.create_game_object("selected model collision")
    wrapper = go.add_component("MeshRenderer")
    native = wrapper._require_cpp_component()
    renderer = native if raw_native else wrapper
    renderer.set_mesh_asset_guid(mesh.guid)
    collider = go.add_component("MeshCollider")
    collider.convex = convex
    published_revision = 0

    def select(index):
        nonlocal published_revision
        if selection == "submesh":
            renderer.submesh_index = index
        else:
            document = renderer.serialize_document()
            document["nodeGroup"] = index
            assert renderer.deserialize_document(document)
        Physics.sync_transforms()
        assert collider.shape_error == ""
        assert collider.collision_geometry_revision > published_revision
        published_revision = collider.collision_geometry_revision

    def hit(x):
        return Physics.raycast(Vector3(x, 5, 0), Vector3(0, -1, 0), 10)

    try:
        select(0)
        assert hit(0) is not None
        assert hit(10) is None
        assert hit(5) is None  # convex hull must not span both source nodes
        np.testing.assert_allclose(native.get_world_bounds(), [-1, -1, -1, 1, 1, 1])

        select(1)
        assert hit(0) is None
        assert hit(10) is not None
        assert hit(5) is None
        np.testing.assert_allclose(native.get_world_bounds(), [9, -1, -1, 11, 1, 1])

        if selection == "submesh":
            renderer.mesh_pivot_offset = Vector3(-10, 0, 0)
            Physics.sync_transforms()
            assert hit(0) is not None
            assert hit(10) is None
            np.testing.assert_allclose(native.get_world_bounds(), [-1, -1, -1, 1, 1, 1])
            # Restoring a renderer publishes selection before mesh resolution.
            document = renderer.serialize_document()
            renderer.submesh_index = 0
            assert renderer.deserialize_document(document)
            Physics.sync_transforms()
            assert hit(0) is not None
            assert hit(10) is None
            np.testing.assert_allclose(native.get_world_bounds(), [-1, -1, -1, 1, 1, 1])
    finally:
        scene.destroy_game_object(go)
        mesh.destroy()
