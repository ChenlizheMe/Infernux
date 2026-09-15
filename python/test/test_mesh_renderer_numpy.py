"""Public NumPy mesh upload uses the existing native mesh contract."""
import json

import numpy as np
import pytest

from Infernux.components.builtin.mesh_renderer import MeshRenderer


def test_renderer_parameters_layer_over_shared_material_without_mutation(scene):
    first = scene.create_game_object("parameter override A").add_component("MeshRenderer")
    second = scene.create_game_object("parameter override B").add_component("MeshRenderer")
    shared = first.get_effective_material()
    assert shared is second.get_effective_material()
    original = tuple(shared.get_color("baseColor"))

    first.set_parameter("baseColor", (0.8, 0.2, 0.1, 1.0), persistent=True)
    first.set_parameter("metallic", 0.4, persistent=True)
    first.set_parameter("baseColor", (0.1, 0.7, 0.2, 1.0))

    assert first.get_parameter("baseColor") == pytest.approx((0.1, 0.7, 0.2, 1.0))
    assert first.get_parameter("baseColor", persistent_only=True) == pytest.approx((0.8, 0.2, 0.1, 1.0))
    assert first.get_parameter("metallic") == pytest.approx(0.4)
    assert tuple(shared.get_color("baseColor")) == original
    assert tuple(second.get_effective_material().get_color("baseColor")) == original

    document = json.loads(first._require_cpp_component().serialize())
    assert document["parameterOverrides"][0]["baseColor"]["type"] == 7
    assert document["parameterOverrides"][0]["baseColor"]["value"] == pytest.approx([0.8, 0.2, 0.1, 1.0])
    assert document["parameterOverrides"][0]["metallic"]["value"] == pytest.approx(0.4)

    assert first.remove_parameter("baseColor") is True
    assert first.get_parameter("baseColor") == pytest.approx((0.8, 0.2, 0.1, 1.0))
    assert first.remove_parameter("baseColor") is False
    first.clear_parameters(persistent=True)
    assert first.get_parameter("baseColor") is None
    assert first.get_parameter("metallic") is None

    restored_object = scene.create_game_object("restored parameter override")
    restored = restored_object.add_component("MeshRenderer")
    assert restored.deserialize_document(document) is True
    restored = restored_object.get_component("MeshRenderer")
    assert restored.get_parameter("baseColor") == pytest.approx((0.8, 0.2, 0.1, 1.0))
    assert restored.get_parameter("metallic") == pytest.approx(0.4)


def test_renderer_parameters_use_reflected_material_types(scene):
    renderer = scene.create_game_object("typed renderer parameters").add_component("MeshRenderer")
    with pytest.raises(KeyError, match="no parameter"):
        renderer.set_parameter("not_declared_by_shader", 1.0)
    with pytest.raises((TypeError, ValueError)):
        renderer.set_parameter("baseColor", (1.0, 0.5, 0.25))
    with pytest.raises((TypeError, ValueError)):
        renderer.set_parameter("metallic", (1.0, 0.0))


def test_renderer_parameter_slots_clone_and_field_removal_are_isolated(scene):
    source_object = scene.create_game_object("two-slot renderer parameters")
    renderer = source_object.add_component("MeshRenderer")
    renderer.set_material_slot_count(2)
    renderer.set_parameter("baseColor", (0.8, 0.1, 0.1, 1.0), material_slot=0, persistent=True)
    renderer.set_parameter("baseColor", (0.1, 0.8, 0.1, 1.0), material_slot=1, persistent=True)
    renderer.set_parameter("metallic", 0.75, material_slot=1)

    document = json.loads(renderer._require_cpp_component().serialize())
    assert len(document["parameterOverrides"]) == 2
    assert document["parameterOverrides"][0]["baseColor"]["value"] == pytest.approx([0.8, 0.1, 0.1, 1.0])
    assert document["parameterOverrides"][1]["baseColor"]["value"] == pytest.approx([0.1, 0.8, 0.1, 1.0])
    assert "metallic" not in document["parameterOverrides"][1]

    clone_object = scene._clone_game_object(source_object)
    clone = clone_object.get_component("MeshRenderer")
    assert clone.get_parameter("baseColor", material_slot=0) == pytest.approx((0.8, 0.1, 0.1, 1.0))
    assert clone.get_parameter("baseColor", material_slot=1) == pytest.approx((0.1, 0.8, 0.1, 1.0))
    assert clone.get_parameter("metallic", material_slot=1) is None

    assert renderer.remove_parameter("baseColor", material_slot=0, persistent=True)
    assert renderer.get_parameter("baseColor", material_slot=0) is None
    assert renderer.get_parameter("baseColor", material_slot=1) == pytest.approx((0.1, 0.8, 0.1, 1.0))
    assert renderer.get_parameter("metallic", material_slot=1) == pytest.approx(0.75)
    renderer.clear_parameters(material_slot=1, persistent=True)
    assert renderer.get_parameter("baseColor", material_slot=1) is None
    assert renderer.get_parameter("metallic", material_slot=1) == pytest.approx(0.75)

    with pytest.raises((IndexError, ValueError), match="material slot"):
        renderer.set_parameter("metallic", 0.5, material_slot=2)


def test_runtime_parameter_owners_do_not_erase_each_other(scene):
    renderer = scene.create_game_object("owned runtime renderer parameters").add_component("MeshRenderer")
    renderer.set_parameter("metallic", 0.25, owner="animation")
    renderer.set_parameter("metallic", 0.75, owner="gameplay")
    renderer.set_parameter("smoothness", 0.4, owner="animation")

    assert renderer.get_parameter("metallic") == pytest.approx(0.75)
    assert renderer.get_parameter("metallic", owner="animation") == pytest.approx(0.25)
    assert renderer.remove_parameter("metallic", owner="gameplay")
    assert renderer.get_parameter("metallic") == pytest.approx(0.25)

    renderer.clear_parameters(owner="animation")
    assert renderer.get_parameter("metallic") is None
    assert renderer.get_parameter("smoothness") is None


def test_renderer_texture_parameter_uses_asset_guid_boundary(scene):
    renderer = scene.create_game_object("renderer texture boundary").add_component("MeshRenderer")
    material = renderer.get_effective_material().clone()
    material.set_texture_guid("texSampler", "white")
    renderer.set_material(material)
    renderer.set_parameter("texSampler", "white")
    assert renderer.get_parameter("texSampler") == "white"
    with pytest.raises(ValueError, match="texture GUID does not exist"):
        renderer.set_parameter("texSampler", "missing-texture-guid")

    document = json.loads(renderer._require_cpp_component().serialize())
    document["parameterOverrides"] = [{"texSampler": {"type": 0, "value": 1.0}}]
    restored = scene.create_game_object("invalid renderer document").add_component("MeshRenderer")
    assert restored.deserialize_document(document) is False


def test_numpy_mesh_upload_copies_and_replaces_native_storage(scene):
    go = scene.create_game_object("numpy mesh")
    native = go.add_component("MeshRenderer")
    renderer = MeshRenderer._get_or_create_wrapper(native, go)
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (3, 1))
    uvs = np.zeros((3, 2), dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    renderer.set_inline_mesh_data(positions, normals, uvs, indices, "triangle")
    assert renderer.vertex_count == renderer.index_count == 3
    np.testing.assert_array_equal(renderer.get_positions(), positions)
    positions[1, 0] = 2
    assert renderer.get_positions()[1][0] == 1
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    assert renderer.get_positions()[1][0] == 2
    with pytest.raises(ValueError, match="positions must have shape"):
        renderer.set_inline_mesh_data(positions.ravel(), normals, uvs, indices)


def test_inline_mesh_exposes_canonical_interleaved_vertex_storage(scene):
    renderer = scene.create_game_object("resident mesh source").add_component("MeshRenderer")
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.tile(np.array([0, 0, 1], dtype=np.float32), (3, 1))
    uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    renderer.set_inline_mesh_data(positions, normals, uvs, np.array([0, 1, 2], dtype=np.uint32))
    words = renderer._require_cpp_component().get_vertex_buffer_data()
    assert words.shape == (3, 23)
    np.testing.assert_array_equal(words[:, 0:3], positions)
    np.testing.assert_array_equal(words[:, 3:6], normals)
    np.testing.assert_array_equal(words[:, 13:15], uvs)
    tangents = np.asarray(renderer.get_tangents())
    np.testing.assert_allclose(tangents[:, :3], [[1, 0, 0]] * 3, atol=1e-6)
    np.testing.assert_array_equal(tangents[:, 3], 1)


def test_vertex_buffer_requires_gpu_inx_buffer(scene):
    from Infernux.compute import buffer

    renderer = scene.create_game_object("resident mesh contract").add_component("MeshRenderer")
    renderer.set_inline_mesh_data(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.tile(np.array([0, 0, 1], dtype=np.float32), (3, 1)),
        np.zeros((3, 2), dtype=np.float32),
        np.array([0, 1, 2], dtype=np.uint32),
    )
    cpu = buffer(shape=(3, 23), dtype=np.float32, device="cpu")
    with pytest.raises(TypeError, match="GPU inx.buffer"):
        renderer.set_vertex_buffer(cpu, (-1, -1, -1), (1, 1, 1))


def test_vertex_buffer_capacity_is_explicit_and_cannot_shrink_below_mesh(scene):
    renderer = scene.create_game_object("resident mesh capacity").add_component("MeshRenderer")
    renderer.set_inline_mesh_data(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        np.tile(np.array([0, 0, 1], dtype=np.float32), (3, 1)),
        np.zeros((3, 2), dtype=np.float32),
        np.array([0, 1, 2], dtype=np.uint32),
    )
    with pytest.raises(ValueError, match="smaller than vertex_count"):
        renderer.create_vertex_buffer(capacity=2)
    with pytest.raises(TypeError, match="must be an integer"):
        renderer.create_vertex_buffer(capacity=3.5)


def test_unbound_numpy_mesh_upload_is_not_silently_ignored():
    with pytest.raises(ReferenceError):
        MeshRenderer().set_inline_mesh_data(None, None, None, None)


def test_native_normal_generation_is_area_weighted_and_manual_normals_are_preserved(scene):
    renderer = scene.create_game_object("generated normals").add_component("MeshRenderer")
    positions = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 2],
                          [10, 0, 0]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 3, 1], dtype=np.uint32)
    uvs = np.zeros((len(positions), 2), dtype=np.float32)
    renderer.set_inline_mesh_data(positions, None, uvs, indices)
    shared = np.array([0, 4, 2]) / np.sqrt(20)
    np.testing.assert_allclose(renderer.get_normals(),
                               [shared, shared, [0, 0, 1], [0, 1, 0], [0, 0, 0]], atol=1e-6)
    manual = np.tile([1, 0, 0], (len(positions), 1)).astype(np.float32)
    renderer.set_inline_mesh_data(positions, manual, uvs, indices)
    np.testing.assert_array_equal(renderer.get_normals(), manual)


def test_native_tangent_generation_and_manual_override(scene):
    renderer = scene.create_game_object("generated tangents").add_component("MeshRenderer")
    positions = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0]], dtype=np.float32)
    normals = np.tile([0, 0, 1], (3, 1)).astype(np.float32)
    uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    np.testing.assert_allclose(renderer.get_tangents(), [[1, 0, 0, 1]] * 3, atol=1e-6)

    manual = np.tile([0, 1, 0, -1], (3, 1)).astype(np.float32)
    renderer.set_inline_mesh_data(positions, normals, uvs, indices, tangents=manual)
    np.testing.assert_array_equal(renderer.get_tangents(), manual)

    positions[2] = [0, 1, 1]
    renderer.set_inline_mesh_data(positions, None, uvs, indices, tangents=manual)
    renderer.recalculate_tangents()
    rebuilt = np.asarray(renderer.get_tangents())
    generated_normals = np.asarray(renderer.get_normals())
    np.testing.assert_allclose(np.linalg.norm(rebuilt[:, :3], axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.sum(rebuilt[:, :3] * generated_normals, axis=1), 0, atol=1e-6)


def test_tangent_generation_normalizes_authored_normals_for_frame_projection(scene):
    renderer = scene.create_game_object("non-unit normals").add_component("MeshRenderer")
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    # Imported/procedural callers are allowed to provide a scaled normal. The
    # authored value remains intact, while the derived tangent frame must be
    # orthogonal to its normalized direction.
    normals = np.tile([0, 0, 4], (3, 1)).astype(np.float32)
    uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    tangents = np.asarray(renderer.get_tangents(), dtype=np.float32)
    np.testing.assert_allclose(np.linalg.norm(tangents[:, :3], axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(tangents[:, :3,] @ np.array([0, 0, 1], dtype=np.float32), 0.0, atol=1e-6)


def test_generated_normals_preserve_split_vertices_and_degenerate_faces(scene):
    renderer = scene.create_game_object("split normals").add_component("MeshRenderer")
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    uvs = np.zeros((len(positions), 2), dtype=np.float32)
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 6, 6], dtype=np.uint32)
    renderer.set_inline_mesh_data(positions, None, uvs, indices)
    expected = [[0, 0, 1]] * 3 + [[0, 1, 0]] * 3 + [[0, 0, 0]]
    np.testing.assert_array_equal(renderer.get_normals(), expected)
    with pytest.raises(ValueError, match="triangles"):
        renderer.set_inline_mesh_data(positions, None, uvs, indices[:2])
    np.testing.assert_array_equal(renderer.get_normals(), expected)
    renderer.set_inline_mesh_data(positions[:0], None, uvs[:0], indices[:0])
    assert renderer.vertex_count == renderer.index_count == 0
    assert renderer.get_normals() == []


def test_inline_geometry_generation_and_world_bounds_are_authoritative(scene):
    from Infernux.lib import Vector3

    go = scene.create_game_object("versioned bounds")
    renderer = go.add_component("MeshRenderer")
    positions = np.array([[-2, -1, -3], [4, 5, 6], [1, 2, -2]], dtype=np.float32)
    uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    initial_version = renderer.inline_mesh_version

    renderer.set_inline_mesh_data(positions, None, uvs, indices)
    assert renderer.inline_mesh_version == initial_version + 1
    renderer.recalculate_normals()
    assert renderer.inline_mesh_version == initial_version + 2
    renderer.recalculate_tangents()
    assert renderer.inline_mesh_version == initial_version + 3
    renderer.recalculate_bounds()
    assert renderer.inline_mesh_version == initial_version + 3

    go.transform.position = Vector3(10, 20, 30)
    go.transform.local_scale = Vector3(2, 3, 4)
    np.testing.assert_allclose(renderer._require_cpp_component().get_world_bounds(),
                               [6, 17, 18, 18, 35, 54], atol=1e-6)

    with pytest.raises(ValueError, match="outside positions"):
        renderer.set_inline_mesh_data(positions, None, uvs, np.array([0, 1, 7], dtype=np.uint32))
    assert renderer.inline_mesh_version == initial_version + 3


@pytest.mark.parametrize("raw_native", [False, True])
def test_visual_mesh_update_requires_explicit_collision_recook(scene, raw_native):
    from Infernux.lib import MeshCollider as NativeMeshCollider, Physics, Vector3

    NativeMeshCollider.clear_cooking_cache()
    go = scene.create_game_object("deforming surface")
    renderer = go.add_component("MeshRenderer")
    if raw_native:
        renderer = renderer._require_cpp_component()
    positions = np.array([[-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1]], dtype=np.float32)
    normals = np.tile([0, 1, 0], (4, 1)).astype(np.float32)
    uvs = np.zeros((4, 2), dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    collider = go.add_component("MeshCollider")
    if raw_native:
        collider = collider._require_cpp_component()
    Physics.sync_transforms()
    before = NativeMeshCollider.get_cooking_cache_stats()["async_submissions"]
    assert Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(0)

    for height in (0.5, 1.0, 2.0):
        positions[:, 1] = height
        renderer.set_inline_mesh_data(positions, normals, uvs, indices)
        Physics.sync_transforms()
        assert NativeMeshCollider.get_cooking_cache_stats()["async_submissions"] == before
        assert Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(0)
    collider.recook()
    Physics.sync_transforms()
    assert NativeMeshCollider.get_cooking_cache_stats()["async_submissions"] == before + 1
    assert collider.shape_error == ""
    assert Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(2)


def test_unbound_collision_recook_is_not_silently_ignored():
    from Infernux.components.builtin.mesh_collider import MeshCollider

    with pytest.raises(ReferenceError):
        MeshCollider().recook()


@pytest.mark.parametrize("supersede", [False, True])
def test_recook_uses_requested_geometry_while_visual_mesh_keeps_changing(scene, supersede):
    from Infernux.lib import GameObject, MeshCollider as NativeMeshCollider, Physics, Vector3

    NativeMeshCollider.clear_cooking_cache()
    go = scene.create_game_object("cooking snapshot")
    renderer = go.add_component("MeshRenderer")
    positions = np.array([[-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1]], dtype=np.float32)
    normals = np.tile([0, 1, 0], (4, 1)).astype(np.float32)
    uvs = np.zeros((4, 2), dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    collider = go.add_component("MeshCollider")
    Physics.sync_transforms()
    positions[:, 1] = 1
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    collider.recook()
    assert collider.is_cooking
    requested_height = 1.0
    if supersede:
        requested_height = 1.25
        positions[:, 1] = requested_height
        renderer.set_inline_mesh_data(positions, normals, uvs, indices)
        collider.recook()
    positions[:, 1] = 2
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    Physics.sync_transforms()
    assert not collider.is_cooking
    assert NativeMeshCollider.get_cooking_cache_stats()["async_submissions"] == 2 + int(supersede)
    assert Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(requested_height)
    clone = GameObject.instantiate(go)
    clone.transform.position = Vector3(5, 0, 0)
    Physics.sync_transforms()
    assert Physics.raycast(Vector3(5, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(requested_height)
    clone.get_component("MeshCollider").recook()
    Physics.sync_transforms()
    assert Physics.raycast(Vector3(5, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(2)
    assert Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(requested_height)
    # A collider-only edit must also use its chosen source, not the latest mesh.
    collider.center = Vector3(0, 0.5, 0)
    Physics.sync_transforms()
    assert Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(requested_height + 0.5)
    go.transform.local_scale = Vector3(1, 2, 1)
    Physics.sync_transforms()
    assert not collider.is_cooking
    assert Physics.raycast(Vector3(0, 6, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx((requested_height + 0.5) * 2)
    go.transform.local_scale = Vector3(1, 1, 1)
    Physics.sync_transforms()
    collider.recook()
    Physics.sync_transforms()
    assert Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(2.5)


@pytest.mark.parametrize("compound", [False, True])
@pytest.mark.parametrize("worker_failure", [False, True])
def test_failed_recook_preserves_complete_collision_and_can_be_replaced(scene, compound, worker_failure):
    from Infernux.lib import Physics, Vector3

    go = scene.create_game_object("atomic collision replacement")
    renderer = go.add_component("MeshRenderer")
    positions = np.array([[-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1]], dtype=np.float32)
    normals = np.tile([0, 1, 0], (4, 1)).astype(np.float32)
    uvs = np.zeros((4, 2), dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    collider = go.add_component("MeshCollider")
    if compound:
        box = go.add_component("BoxCollider")
        box.size = Vector3(1, 1, 1)
        box.center = Vector3(5, 0, 0)
    Physics.sync_transforms()

    if worker_failure:
        renderer.set_inline_mesh_data(np.zeros_like(positions), normals, uvs, indices)
    else:
        renderer.set_inline_mesh_data(positions, normals, uvs, indices[:2])
    collider.recook()
    Physics.sync_transforms()
    assert collider.shape_error
    assert not collider.is_cooking
    hit = Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10)
    assert hit is not None, "A rejected mesh must not remove an existing compound child"
    assert hit.point.y == pytest.approx(0)

    positions[:, 1] = 2
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    collider.recook()
    Physics.sync_transforms()
    assert collider.shape_error == ""
    assert Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(2)
    if compound:
        assert Physics.raycast(Vector3(5, 5, 0), Vector3(0, -1, 0), 10).point.y == pytest.approx(0.5)


def test_destroyed_cooking_owner_cannot_publish_to_replacement(scene):
    from Infernux.lib import MeshCollider as NativeMeshCollider, Physics, Vector3

    NativeMeshCollider.clear_cooking_cache()
    positions = np.array([[-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1]], dtype=np.float32)
    normals = np.tile([0, 1, 0], (4, 1)).astype(np.float32)
    uvs = np.zeros((4, 2), dtype=np.float32)
    indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)
    for height in range(4):
        go = scene.create_game_object("retired cooking owner")
        renderer = go.add_component("MeshRenderer")
        positions[:, 1] = -1
        renderer.set_inline_mesh_data(positions, normals, uvs, indices)
        collider = go.add_component("MeshCollider")
        Physics.sync_transforms()
        positions[:, 1] = height
        renderer.set_inline_mesh_data(positions, normals, uvs, indices)
        collider.recook()
        assert collider.is_cooking
        scene.destroy_game_object(go)
    replacement = scene.create_game_object("new collision owner")
    renderer = replacement.add_component("MeshRenderer")
    positions[:, 1] = 7
    renderer.set_inline_mesh_data(positions, normals, uvs, indices)
    collider = replacement.add_component("MeshCollider")
    Physics.sync_transforms()
    assert collider.shape_error == ""
    assert NativeMeshCollider.get_cooking_cache_stats()["pending"] == 0
    hit = Physics.raycast(Vector3(0, 10, 0), Vector3(0, -1, 0), 20)
    assert hit is not None
    assert hit.point.y == pytest.approx(7)
