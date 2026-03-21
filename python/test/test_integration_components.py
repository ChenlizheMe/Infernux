"""Integration tests — Components, rendering objects, assets (real engine)."""
from __future__ import annotations

import pytest

from InfEngine.lib import (
    SceneManager,
    Vector3,
    PrimitiveType,
    TextureLoader,
    InfMaterial,
    LightType,
    LightShadows,
)


# ═══════════════════════════════════════════════════════════════════════════
# Component add / remove / query
# ═══════════════════════════════════════════════════════════════════════════

class TestComponentLifecycle:
    def test_add_and_get_component(self, scene):
        go = scene.create_game_object("GO")
        rb = go.add_component("Rigidbody")
        assert rb is not None
        fetched = go.get_component("Rigidbody")
        assert fetched is not None

    def test_transform_always_present(self, scene):
        go = scene.create_game_object("GO")
        t = go.get_component("Transform")
        assert t is not None
        assert t.type_name == "Transform"

    def test_get_components_lists_all(self, scene):
        go = scene.create_game_object("GO")
        go.add_component("Rigidbody")
        go.add_component("BoxCollider")
        names = [c.type_name for c in go.get_components()]
        assert "Transform" in names
        assert "Rigidbody" in names
        assert "BoxCollider" in names

    def test_remove_component(self, scene):
        go = scene.create_game_object("GO")
        rb = go.add_component("Rigidbody")
        go.remove_component(rb)
        assert go.get_component("Rigidbody") is None

    def test_cannot_remove_transform(self, scene):
        go = scene.create_game_object("GO")
        t = go.get_component("Transform")
        result = go.remove_component(t)
        assert result is False
        assert go.get_component("Transform") is not None

    @pytest.mark.parametrize("comp_type", [
        "Rigidbody", "BoxCollider", "SphereCollider", "CapsuleCollider",
        "MeshCollider", "MeshRenderer", "Light", "Camera",
        "AudioSource", "AudioListener",
    ])
    def test_all_component_types_addable(self, scene, comp_type):
        go = scene.create_game_object(f"GO_{comp_type}")
        comp = go.add_component(comp_type)
        assert comp is not None
        assert comp.type_name == comp_type


# ═══════════════════════════════════════════════════════════════════════════
# Collider properties
# ═══════════════════════════════════════════════════════════════════════════

class TestColliders:
    def test_box_collider_size(self, scene):
        go = scene.create_game_object("BC")
        bc = go.add_component("BoxCollider")
        bc.size = Vector3(2, 3, 4)
        s = bc.size
        assert (s.x, s.y, s.z) == pytest.approx((2, 3, 4))

    def test_sphere_collider_radius(self, scene):
        go = scene.create_game_object("SC")
        sc = go.add_component("SphereCollider")
        sc.radius = 2.5
        assert sc.radius == pytest.approx(2.5)

    def test_capsule_collider_properties(self, scene):
        go = scene.create_game_object("CC")
        cc = go.add_component("CapsuleCollider")
        cc.radius = 1.0
        cc.height = 3.0
        assert cc.radius == pytest.approx(1.0)
        assert cc.height == pytest.approx(3.0)

    def test_collider_is_trigger(self, scene):
        go = scene.create_game_object("T")
        bc = go.add_component("BoxCollider")
        bc.is_trigger = True
        assert bc.is_trigger is True
        bc.is_trigger = False
        assert bc.is_trigger is False


# ═══════════════════════════════════════════════════════════════════════════
# Camera
# ═══════════════════════════════════════════════════════════════════════════

class TestCamera:
    def test_camera_defaults(self, scene):
        go = scene.create_game_object("Cam")
        cam = go.add_component("Camera")
        assert cam.field_of_view == pytest.approx(60.0)
        assert cam.near_clip > 0
        assert cam.far_clip > cam.near_clip

    def test_camera_fov_round_trip(self, scene):
        go = scene.create_game_object("Cam")
        cam = go.add_component("Camera")
        cam.field_of_view = 90.0
        assert cam.field_of_view == pytest.approx(90.0)

    def test_camera_depth(self, scene):
        go = scene.create_game_object("Cam")
        cam = go.add_component("Camera")
        cam.depth = 5
        assert cam.depth == pytest.approx(5)


# ═══════════════════════════════════════════════════════════════════════════
# Light
# ═══════════════════════════════════════════════════════════════════════════

class TestLight:
    def test_light_defaults(self, scene):
        go = scene.create_game_object("L")
        light = go.add_component("Light")
        assert light.light_type == LightType.Directional
        assert light.intensity == pytest.approx(1.0)

    def test_light_type_point(self, scene):
        go = scene.create_game_object("PL")
        light = go.add_component("Light")
        light.light_type = LightType.Point
        assert light.light_type == LightType.Point

    def test_light_intensity_round_trip(self, scene):
        go = scene.create_game_object("L")
        light = go.add_component("Light")
        light.intensity = 2.5
        assert light.intensity == pytest.approx(2.5)

    def test_light_color(self, scene):
        go = scene.create_game_object("L")
        light = go.add_component("Light")
        light.color = Vector3(1, 0, 0)
        c = light.color
        assert c.x == pytest.approx(1.0)
        assert c.y == pytest.approx(0.0)

    def test_light_shadows(self, scene):
        go = scene.create_game_object("L")
        light = go.add_component("Light")
        light.shadows = LightShadows.Hard
        assert light.shadows == LightShadows.Hard


# ═══════════════════════════════════════════════════════════════════════════
# MeshRenderer
# ═══════════════════════════════════════════════════════════════════════════

class TestMeshRenderer:
    def test_primitive_mesh_has_data(self, scene):
        cube = scene.create_primitive(PrimitiveType.Cube, "Cube")
        mr = cube.get_component("MeshRenderer")
        assert mr is not None
        positions = mr.get_positions()
        normals = mr.get_normals()
        indices = mr.get_indices()
        assert len(positions) > 0
        assert len(normals) > 0
        assert len(indices) > 0

    def test_sphere_has_more_verts_than_cube(self, scene):
        cube = scene.create_primitive(PrimitiveType.Cube, "C")
        sphere = scene.create_primitive(PrimitiveType.Sphere, "S")
        cube_verts = len(cube.get_component("MeshRenderer").get_positions())
        sphere_verts = len(sphere.get_component("MeshRenderer").get_positions())
        assert sphere_verts > cube_verts

    def test_shadow_properties(self, scene):
        cube = scene.create_primitive(PrimitiveType.Cube, "C")
        mr = cube.get_component("MeshRenderer")
        mr.casts_shadows = False
        assert mr.casts_shadows is False
        mr.casts_shadows = True
        assert mr.casts_shadows is True


# ═══════════════════════════════════════════════════════════════════════════
# Texture (real GPU-side creation)
# ═══════════════════════════════════════════════════════════════════════════

class TestTextureCreation:
    def test_solid_color(self, engine):
        tex = TextureLoader.create_solid_color(32, 32, 255, 0, 0, 255)
        assert tex.width == 32
        assert tex.height == 32

    def test_different_sizes(self, engine):
        for size in [1, 16, 64, 256]:
            tex = TextureLoader.create_solid_color(size, size, 0, 0, 0, 255)
            assert tex.width == size
            assert tex.height == size


# ═══════════════════════════════════════════════════════════════════════════
# Material
# ═══════════════════════════════════════════════════════════════════════════

class TestMaterial:
    def test_create_default_lit(self, engine):
        mat = InfMaterial.create_default_lit()
        assert mat is not None

    def test_material_assignable_to_renderer(self, scene):
        cube = scene.create_primitive(PrimitiveType.Cube, "MatCube")
        mr = cube.get_component("MeshRenderer")
        mat = InfMaterial.create_default_lit()
        mr.render_material = mat
        assert mr.render_material is not None


# ═══════════════════════════════════════════════════════════════════════════
# Component serialization
# ═══════════════════════════════════════════════════════════════════════════

class TestComponentSerialization:
    def test_rigidbody_serializes(self, scene):
        go = scene.create_game_object("RB")
        rb = go.add_component("Rigidbody")
        rb.mass = 3.14
        json_str = rb.serialize()
        assert "mass" in json_str.lower() or "3.14" in json_str

    def test_round_trip_via_scene(self, scene):
        go = scene.create_game_object("Persist")
        go.transform.position = Vector3(1, 2, 3)
        rb = go.add_component("Rigidbody")
        rb.mass = 7.77
        go.add_component("SphereCollider").radius = 2.0

        json_str = scene.serialize()

        sm = SceneManager.instance()
        scene2 = sm.create_scene("reload")
        sm.set_active_scene(scene2)
        scene2.deserialize(json_str)

        found = scene2.find("Persist")
        assert found is not None
        assert found.transform.position.x == pytest.approx(1)
        rb2 = found.get_component("Rigidbody")
        assert rb2.mass == pytest.approx(7.77)
        sc2 = found.get_component("SphereCollider")
        assert sc2.radius == pytest.approx(2.0)
