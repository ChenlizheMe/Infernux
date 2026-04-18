"""Integration tests — Components, rendering objects, assets (real engine)."""
from __future__ import annotations

import importlib
import sys

import pytest

from Infernux.components import InxComponent
from Infernux.components.builtin import Camera as CameraComponent
from Infernux.renderstack.render_stack import RenderStack
from Infernux.renderstack.render_stack_pipeline import RenderStackPipeline

from Infernux.lib import (
    SceneManager,
    Vector3,
    PrimitiveType,
    TextureLoader,
    InxMaterial,
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

    def test_add_and_get_python_component_by_class(self, scene):
        class ProbeComponent(InxComponent):
            pass

        go = scene.create_game_object("GO")
        probe = go.add_component(ProbeComponent)

        assert isinstance(probe, ProbeComponent)
        assert go.get_component(ProbeComponent) is probe
        assert go.get_component("ProbeComponent") is probe
        assert go.get_components(ProbeComponent) == [probe]

    def test_add_and_get_builtin_component_by_class(self, scene):
        go = scene.create_game_object("CamGO")
        cam = go.add_component(CameraComponent)

        assert isinstance(cam, CameraComponent)
        assert go.get_component(CameraComponent) is cam
        assert go.get_components(CameraComponent) == [cam]
        assert go.remove_component(cam) is True

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

    def test_get_components_returns_python_instances_not_proxies(self, scene):
        class ProbeComponent(InxComponent):
            pass

        go = scene.create_game_object("GO")
        probe = go.add_component(ProbeComponent)

        components = go.get_components()

        assert probe in components
        assert all(type(component).__name__ != "PyComponentProxy" for component in components)

    def test_script_loader_preserves_class_identity_for_imports(self, scene, tmp_path):
        from Infernux.components.script_loader import load_component_from_file
        from Infernux.engine.project_context import (
            get_project_root,
            set_project_root,
            temporary_script_import_paths,
        )

        project_root = tmp_path / "project"
        assets_root = project_root / "Assets"
        assets_root.mkdir(parents=True)
        script_path = assets_root / "a2.py"
        script_path.write_text(
            "from Infernux.components import *\n\n"
            "class NewComponent1(InxComponent):\n"
            "    pass\n",
            encoding="utf-8",
        )

        previous_root = get_project_root()
        saved_modules = {name: sys.modules.get(name) for name in ("a2",)}
        for name in saved_modules:
            sys.modules.pop(name, None)

        set_project_root(str(project_root))
        try:
            loaded_class = load_component_from_file(str(script_path))
            with temporary_script_import_paths(str(script_path)):
                direct_module = importlib.import_module("a2")

            go = scene.create_game_object("GO")
            go.add_component(loaded_class)
            components = go.get_components()

            assert loaded_class is direct_module.NewComponent1
            assert any(isinstance(component, direct_module.NewComponent1) for component in components)
        finally:
            set_project_root(previous_root)
            for name in ("a2",):
                sys.modules.pop(name, None)
            for name, module in saved_modules.items():
                if module is not None:
                    sys.modules[name] = module

    def test_remove_component(self, scene):
        go = scene.create_game_object("GO")
        rb = go.add_component("Rigidbody")
        go.remove_component(rb)
        assert go.get_component("Rigidbody") is None

    def test_remove_box_collider_with_mesh_collider_and_rigidbody(self, scene):
        go = scene.create_primitive(PrimitiveType.Cube, "ColliderHost")
        mesh = go.add_component("MeshCollider")
        box = go.add_component("BoxCollider")
        go.add_component("Rigidbody")

        assert go.remove_component(box) is True
        assert go.get_component("BoxCollider") is None
        assert go.get_component("MeshCollider") is mesh
        assert go.get_component("Rigidbody") is not None

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

    def test_python_component_receives_disable_when_game_object_deactivates(self, scene):
        events = []

        class ProbeComponent(InxComponent):
            def awake(self):
                events.append("awake")

            def on_enable(self):
                events.append("on_enable")

            def on_disable(self):
                events.append("on_disable")

        go = scene.create_game_object("LifecycleGO")
        go.add_component(ProbeComponent)

        go.active = False
        go.active = True

        assert events == ["awake", "on_enable", "on_disable", "on_enable"]

    def test_adding_component_to_inactive_game_object_defers_awake_until_activation(self, scene):
        events = []

        class ProbeComponent(InxComponent):
            def awake(self):
                events.append("awake")

            def on_enable(self):
                events.append("on_enable")

        go = scene.create_game_object("InactiveLifecycleGO")
        go.active = False
        go.add_component(ProbeComponent)

        assert events == []

        go.active = True

        assert events == ["awake", "on_enable"]

    def test_component_added_during_update_starts_before_same_frame_late_update(self, scene):
        sm = SceneManager.instance()
        events = []

        class SpawnedComponent(InxComponent):
            def awake(self):
                events.append("spawned_awake")

            def on_enable(self):
                events.append("spawned_on_enable")

            def start(self):
                events.append("spawned_start")

            def late_update(self, delta_time: float):
                events.append("spawned_late_update")

        class SpawnerComponent(InxComponent):
            def awake(self):
                self._spawned = False

            def update(self, delta_time: float):
                if self._spawned:
                    return
                self._spawned = True
                events.append("spawner_update")
                self.game_object.add_component(SpawnedComponent)

            def late_update(self, delta_time: float):
                events.append("spawner_late_update")

        go = scene.create_game_object("StartTimingGO")
        go.add_component(SpawnerComponent)

        sm.play()
        sm.pause()
        events.clear()

        sm.step(1.0 / 60.0)

        assert events == [
            "spawner_update",
            "spawned_awake",
            "spawned_on_enable",
            "spawned_start",
            "spawner_late_update",
            "spawned_late_update",
        ]

    def test_disabling_component_does_not_stop_coroutines(self, scene):
        sm = SceneManager.instance()
        events = []

        class ProbeComponent(InxComponent):
            def awake(self):
                self.start_coroutine(self._runner())

            def _runner(self):
                events.append("coroutine_started")
                yield None
                events.append("coroutine_resumed")

            def update(self, delta_time: float):
                events.append("update")

        sm.play()
        sm.pause()

        go = scene.create_game_object("DisabledCoroutineGO")
        comp = go.add_component(ProbeComponent)
        comp.enabled = False
        events.clear()

        sm.step(1.0 / 60.0)

        assert events == ["coroutine_resumed"]

    def test_game_object_deactivation_stops_coroutines_even_when_component_is_disabled(self, scene):
        sm = SceneManager.instance()
        events = []

        class ProbeComponent(InxComponent):
            def awake(self):
                self.start_coroutine(self._runner())

            def _runner(self):
                events.append("coroutine_started")
                yield None
                events.append("coroutine_resumed")

        sm.play()
        sm.pause()

        go = scene.create_game_object("DeactivatedCoroutineGO")
        comp = go.add_component(ProbeComponent)
        comp.enabled = False
        events.clear()

        go.active = False
        go.active = True
        sm.step(1.0 / 60.0)

        assert events == []

    def test_awake_exception_disables_component(self, scene):
        events = []

        class ProbeComponent(InxComponent):
            def awake(self):
                events.append("awake")
                raise RuntimeError("boom")

            def on_enable(self):
                events.append("on_enable")

        go = scene.create_game_object("AwakeExceptionGO")
        comp = go.add_component(ProbeComponent)

        assert events == ["awake"]
        assert comp.enabled is False

    def test_python_component_destroy_skips_on_destroy_when_never_activated(self, scene):
        events = []

        class ProbeComponent(InxComponent):
            def on_destroy(self):
                events.append("on_destroy")

        go = scene.create_game_object("DormantDestroyGO")
        go.active = False
        go.add_component(ProbeComponent)

        scene.destroy_game_object(go)
        scene.process_pending_destroys()

        assert events == []

    def test_destroy_active_python_component_calls_disable_before_destroy(self, scene):
        events = []

        class ProbeComponent(InxComponent):
            def awake(self):
                events.append("awake")

            def on_enable(self):
                events.append("on_enable")

            def on_disable(self):
                events.append("on_disable")

            def on_destroy(self):
                events.append("on_destroy")

        go = scene.create_game_object("ActiveDestroyGO")
        go.add_component(ProbeComponent)
        events.clear()

        scene.destroy_game_object(go)
        scene.process_pending_destroys()

        assert events == ["on_disable", "on_destroy"]

    def test_renderstack_clears_active_instance_when_host_game_object_deactivates(self, scene):
        go = scene.create_game_object("RenderStackGO")
        stack = go.add_component(RenderStack)

        assert RenderStack.instance() is stack

        go.active = False

        assert RenderStack.instance() is None

    def test_renderstack_pipeline_ignores_inactive_game_objects(self, scene):
        go = scene.create_game_object("InactiveRenderStackGO")
        go.add_component(RenderStack)
        go.active = False

        RenderStack._active_instance = None


        class _Context:
            pass

        ctx = _Context()
        ctx.scene = scene

        pipeline = RenderStackPipeline()
        assert pipeline._find_render_stack(ctx) is None


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
        assert light.shadow_bias == pytest.approx(0.0)

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
        assert c[0] == pytest.approx(1.0)
        assert c[1] == pytest.approx(0.0)
        assert c[2] == pytest.approx(0.0)
        assert c[3] == pytest.approx(1.0)

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
        mat = InxMaterial.create_default_lit()
        assert mat is not None

    def test_material_assignable_to_renderer(self, scene):
        cube = scene.create_primitive(PrimitiveType.Cube, "MatCube")
        mr = cube.get_component("MeshRenderer")
        mat = InxMaterial.create_default_lit()
        mr.material = mat
        assert mr.get_material(0) is not None


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
