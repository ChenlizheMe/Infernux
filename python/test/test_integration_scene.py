"""Integration tests — Scene management and GameObject hierarchy (real engine)."""
from __future__ import annotations

import pytest

from InfEngine.lib import SceneManager, Vector3, PrimitiveType, quatf


# ═══════════════════════════════════════════════════════════════════════════
# Scene creation & querying
# ═══════════════════════════════════════════════════════════════════════════

class TestSceneLifecycle:
    def test_create_scene(self, scene):
        assert scene is not None
        assert scene.name == "pytest_scene"

    def test_active_scene(self, scene):
        sm = SceneManager.instance()
        assert sm.get_active_scene() is scene

    def test_scene_starts_empty(self, scene):
        assert len(scene.get_root_objects()) == 0
        assert len(scene.get_all_objects()) == 0


# ═══════════════════════════════════════════════════════════════════════════
# GameObject CRUD
# ═══════════════════════════════════════════════════════════════════════════

class TestGameObject:
    def test_create_game_object(self, scene):
        go = scene.create_game_object("TestObj")
        assert go.name == "TestObj"
        assert go.active is True

    def test_unique_ids(self, scene):
        a = scene.create_game_object("A")
        b = scene.create_game_object("B")
        assert a.id != b.id

    def test_find_by_name(self, scene):
        go = scene.create_game_object("Searchable")
        found = scene.find("Searchable")
        assert found is not None
        assert found.id == go.id

    def test_find_by_id(self, scene):
        go = scene.create_game_object("ById")
        found = scene.find_by_id(go.id)
        assert found is not None
        assert found.name == "ById"

    def test_find_nonexistent_returns_none(self, scene):
        assert scene.find("$$$nonexistent$$$") is None

    def test_get_all_objects(self, scene):
        scene.create_game_object("X")
        scene.create_game_object("Y")
        assert len(scene.get_all_objects()) == 2

    def test_destroy_game_object(self, scene):
        go = scene.create_game_object("Temp")
        scene.destroy_game_object(go)
        scene.process_pending_destroys()
        assert scene.find("Temp") is None

    def test_deactivate_game_object(self, scene):
        go = scene.create_game_object("Toggle")
        go.active = False
        assert go.active is False
        go.active = True
        assert go.active is True


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchy (parent / child)
# ═══════════════════════════════════════════════════════════════════════════

class TestHierarchy:
    def test_set_parent(self, scene):
        parent = scene.create_game_object("Parent")
        child = scene.create_game_object("Child")
        child.set_parent(parent)
        assert child.get_parent().id == parent.id
        assert len(parent.get_children()) == 1

    def test_unparent(self, scene):
        parent = scene.create_game_object("P")
        child = scene.create_game_object("C")
        child.set_parent(parent)
        child.set_parent(None)
        assert child.get_parent() is None

    def test_root_objects_exclude_children(self, scene):
        parent = scene.create_game_object("Root")
        child = scene.create_game_object("Leaf")
        child.set_parent(parent)
        roots = scene.get_root_objects()
        root_ids = {o.id for o in roots}
        assert parent.id in root_ids
        assert child.id not in root_ids

    def test_multiple_children(self, scene):
        parent = scene.create_game_object("P")
        for i in range(5):
            c = scene.create_game_object(f"C{i}")
            c.set_parent(parent)
        assert len(parent.get_children()) == 5


# ═══════════════════════════════════════════════════════════════════════════
# Transform
# ═══════════════════════════════════════════════════════════════════════════

class TestTransform:
    def test_default_position_is_origin(self, scene):
        go = scene.create_game_object("T")
        pos = go.transform.position
        assert (pos.x, pos.y, pos.z) == pytest.approx((0, 0, 0))

    def test_set_position(self, scene):
        go = scene.create_game_object("T")
        go.transform.position = Vector3(1, 2, 3)
        pos = go.transform.position
        assert (pos.x, pos.y, pos.z) == pytest.approx((1, 2, 3))

    def test_local_vs_world_position(self, scene):
        parent = scene.create_game_object("P")
        parent.transform.position = Vector3(10, 0, 0)
        child = scene.create_game_object("C")
        child.set_parent(parent)
        child.transform.local_position = Vector3(0, 5, 0)
        world = child.transform.position
        assert world.x == pytest.approx(10)
        assert world.y == pytest.approx(5)

    def test_scale(self, scene):
        go = scene.create_game_object("S")
        go.transform.local_scale = Vector3(2, 3, 4)
        s = go.transform.local_scale
        assert (s.x, s.y, s.z) == pytest.approx((2, 3, 4))

    def test_rotation_euler(self, scene):
        go = scene.create_game_object("R")
        go.transform.euler_angles = Vector3(0, 90, 0)
        angles = go.transform.euler_angles
        assert angles.y == pytest.approx(90, abs=0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Primitives
# ═══════════════════════════════════════════════════════════════════════════

class TestPrimitives:
    @pytest.mark.parametrize("ptype", [
        PrimitiveType.Cube,
        PrimitiveType.Sphere,
        PrimitiveType.Plane,
        PrimitiveType.Cylinder,
        PrimitiveType.Capsule,
    ])
    def test_create_primitive(self, scene, ptype):
        go = scene.create_primitive(ptype, f"Prim_{ptype.name}")
        assert go is not None
        comps = [c.type_name for c in go.get_components()]
        assert "Transform" in comps
        assert "MeshRenderer" in comps

    def test_primitive_has_mesh_data(self, scene):
        cube = scene.create_primitive(PrimitiveType.Cube, "Cube")
        mr = cube.get_component("MeshRenderer")
        positions = mr.get_positions()
        indices = mr.get_indices()
        assert len(positions) > 0
        assert len(indices) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Instantiate (clone)
# ═══════════════════════════════════════════════════════════════════════════

class TestInstantiate:
    def test_clone_game_object(self, scene):
        original = scene.create_game_object("Original")
        original.transform.position = Vector3(5, 5, 5)
        clone = scene.instantiate_game_object(original)
        assert clone is not None
        assert "Clone" in clone.name
        pos = clone.transform.position
        assert pos.x == pytest.approx(5)

    def test_clone_with_components(self, scene):
        go = scene.create_game_object("WithRB")
        rb = go.add_component("Rigidbody")
        rb.mass = 7.5
        clone = scene.instantiate_game_object(go)
        clone_rb = clone.get_component("Rigidbody")
        assert clone_rb is not None
        assert clone_rb.mass == pytest.approx(7.5)


# ═══════════════════════════════════════════════════════════════════════════
# Scene serialization
# ═══════════════════════════════════════════════════════════════════════════

class TestSceneSerialization:
    def test_serialize_produces_json(self, scene):
        scene.create_game_object("SerObj")
        json_str = scene.serialize()
        assert len(json_str) > 0
        assert "SerObj" in json_str

    def test_save_and_load(self, scene):
        go = scene.create_game_object("Persistent")
        go.transform.position = Vector3(42, 0, 0)
        json_str = scene.serialize()

        # Create a new scene and deserialize
        sm = SceneManager.instance()
        scene2 = sm.create_scene("loaded_scene")
        sm.set_active_scene(scene2)
        scene2.deserialize(json_str)
        found = scene2.find("Persistent")
        assert found is not None
        assert found.transform.position.x == pytest.approx(42)
