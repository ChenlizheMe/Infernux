from __future__ import annotations

import numpy as np

import infernux as inx
from Infernux.components.builtin import MeshRenderer
from Infernux.lib import AssetRegistry


def _quad(*, z: float = 0.0):
    positions = np.array(
        [[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]], dtype=np.float32
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    submeshes = [
        {
            "name": "lower",
            "index_start": 0,
            "index_count": 3,
            "vertex_start": 0,
            "vertex_count": 4,
            "material_slot": 0,
        },
        {
            "name": "upper",
            "index_start": 3,
            "index_count": 3,
            "vertex_start": 0,
            "vertex_count": 4,
            "material_slot": 1,
        },
    ]
    return positions, indices, uvs, submeshes


def test_public_mesh_resource_create_share_copy_and_publish(engine, scene):
    positions, indices, uvs, submeshes = _quad()
    mesh = inx.Mesh.from_data(
        positions,
        indices,
        uvs=uvs,
        submeshes=submeshes,
        material_slots=("Ground", "Top"),
        name="Runtime Quad",
    )
    registry = AssetRegistry.instance()
    assert mesh.guid.startswith("runtime-mesh:")
    assert mesh.vertex_count == 4
    assert mesh.index_count == 6
    assert mesh.submesh_count == 2
    assert mesh.material_slots == ("Ground", "Top")
    assert mesh.get_submesh(1)["material_slot"] == 1
    np.testing.assert_array_equal(mesh.vertex_buffer["positions"], positions)
    np.testing.assert_array_equal(mesh.vertex_buffer["uvs"], uvs)
    np.testing.assert_array_equal(mesh.index_buffer, indices)

    first = scene.create_game_object("shared-a")
    second = scene.create_game_object("shared-b")
    cloned = None
    try:
        first_renderer = first.add_component("MeshRenderer")
        second_renderer = second.add_component("MeshRenderer")
        wrapper = MeshRenderer._get_or_create_wrapper(first_renderer, first)
        wrapper.mesh = mesh
        second_renderer.set_mesh_asset_guid(mesh.guid)
        assert wrapper.mesh.native is mesh.native
        assert wrapper.shared_mesh.native is mesh.native
        assert first_renderer.get_mesh_asset() is second_renderer.get_mesh_asset()
        cloned = scene._clone_game_object(first)
        cloned_renderer = cloned.get_component("MeshRenderer")
        assert cloned_renderer.get_mesh_asset() is mesh.native

        version = registry.get_asset_residency(mesh.guid).runtime_version
        replacement = np.array([[2, 3, 4]], dtype=np.float32)
        mesh.update_vertices(1, positions=replacement)
        replacement[:] = 99
        assert registry.get_asset_residency(mesh.guid).runtime_version == version + 1
        assert first_renderer.get_positions()[1] == (2.0, 3.0, 4.0)
        assert second_renderer.get_positions()[1] == (2.0, 3.0, 4.0)

        copied = mesh.copy("Independent Quad")
        assert copied.guid != mesh.guid
        copied_positions = copied.vertex_buffer["positions"]
        copied_positions[:, 2] = 8
        copied.update_vertices(0, positions=copied_positions)
        assert copied.vertex_buffer["positions"][0, 2] == 8
        assert mesh.vertex_buffer["positions"][0, 2] == 0

        new_positions, new_indices, _, _ = _quad(z=5)
        copied.set_data(new_positions, new_indices)
        assert copied.submesh_count == 1
        assert copied.vertex_buffer["positions"][0, 2] == 5
        assert mesh.submesh_count == 2

        mesh.destroy()
        assert not registry.is_loaded(mesh.guid)
        assert first_renderer.get_mesh_asset() is None
        assert second_renderer.get_mesh_asset() is None
        assert cloned_renderer.get_mesh_asset() is None
    finally:
        if cloned is not None:
            scene.destroy_game_object(cloned)
        scene.destroy_game_object(first)
        scene.destroy_game_object(second)


def test_public_mesh_range_validation_is_atomic(engine):
    positions, indices, _, _ = _quad()
    mesh = inx.Mesh.from_data(positions, indices)
    version = AssetRegistry.instance().get_asset_residency(mesh.guid).runtime_version
    before = mesh.vertex_buffer

    for kwargs in (
        {"positions": np.zeros((2, 2), np.float32)},
        {
            "positions": np.zeros((2, 3), np.float32),
            "normals": np.zeros((1, 3), np.float32),
        },
        {"positions": np.full((1, 3), np.nan, np.float32)},
    ):
        try:
            mesh.update_vertices(0, **kwargs)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid Mesh publication unexpectedly succeeded")

    assert AssetRegistry.instance().get_asset_residency(mesh.guid).runtime_version == version
    for name, values in before.items():
        np.testing.assert_array_equal(mesh.vertex_buffer[name], values)


def test_runtime_mesh_survives_play_mode_scene_rebuild(engine, scene):
    from Infernux.engine.play_mode import PlayModeManager

    positions, indices, _, _ = _quad()
    mesh = inx.Mesh.from_data(positions, indices, name="Play Runtime Quad")
    owner = scene.create_game_object("Play Runtime Mesh")
    owner.add_component("MeshRenderer").set_mesh_asset_guid(mesh.guid)
    snapshot = scene._capture_play_mode_snapshot()

    previous_manager = PlayModeManager.instance()
    manager = PlayModeManager()
    manager.set_asset_database(AssetRegistry.instance().get_asset_database())
    try:
        assert manager._rebuild_active_scene(snapshot, for_play=True)
        runtime_owner = scene.find("Play Runtime Mesh")
        assert runtime_owner.get_component("MeshRenderer").get_mesh_asset() is mesh.native

        assert manager._rebuild_active_scene(snapshot, for_play=False)
        restored_owner = scene.find("Play Runtime Mesh")
        assert restored_owner.get_component("MeshRenderer").get_mesh_asset() is mesh.native
    finally:
        PlayModeManager._instance = previous_manager
