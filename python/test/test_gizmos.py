"""Tests for Infernux.gizmos — Gizmos drawing API and GizmosCollector."""

import math
from types import SimpleNamespace

import pytest

from Infernux.gizmos.gizmos import Gizmos
from Infernux.gizmos.collector import GizmosCollector
from Infernux.components.builtin import Camera
from Infernux.components.particle_system import ParticleBoundsMode, ParticleSystem
from Infernux.lib import Vector3
import numpy as np
from Infernux.particle import EmitterShape, EmitterShapeKind


def test_numpy_line_batch_captures_inputs_and_preserves_index_offsets():
    Gizmos._begin_frame()
    points=np.array([[0,0,0],[1,0,0],[1,1,0]],dtype=np.float32)
    edges=np.array([[0,1],[1,2]],dtype=np.int32)
    Gizmos.color=(1,0,0)
    Gizmos.draw_lines(points,edges)
    points[:]=99
    edges[:]=0
    Gizmos.draw_line((2,2,2),(3,3,3))
    vertices,count,indices,descriptors,batches=Gizmos._get_packed_data()
    assert count==5 and batches==1
    assert list(indices)==[0,1,1,2,3,4]
    assert list(vertices[:6])==[0,0,0,1,0,0]
    assert list(descriptors[:2])==[0,6]
    with pytest.raises(ValueError,match="outside"):
        Gizmos.draw_lines(points,np.array([[0,3]],dtype=np.uint32))
    with pytest.raises(TypeError,match="integer"):
        Gizmos.draw_lines(points,np.zeros((1,2),dtype=np.float32))
    Gizmos._begin_frame()


def test_line_batch_only_splits_when_world_matrix_changes():
    Gizmos._begin_frame()
    Gizmos.draw_line((0, 0, 0), (1, 0, 0))
    Gizmos.draw_line((0, 1, 0), (1, 1, 0))
    Gizmos.matrix = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        3, 0, 0, 1,
    ]
    Gizmos.draw_line((0, 2, 0), (1, 2, 0))

    _, _, _, descriptors, batches = Gizmos._get_packed_data()

    assert batches == 2
    assert list(descriptors[:2]) == [0, 4]
    assert list(descriptors[18:20]) == [4, 2]
    Gizmos._begin_frame()


def test_geometry_profile_counts_nested_helper_once(monkeypatch):
    from Infernux.gizmos import gizmos as gizmo_module

    if not gizmo_module._GEOMETRY_PROFILE_COMPILED:
        pytest.skip("Gizmo helper profiling is compiled out of this native configuration")

    clock = gizmo_module.time.perf_counter
    ticks = iter((1.0, 1.005))
    monkeypatch.setattr(gizmo_module.time, "perf_counter", lambda: next(ticks))
    Gizmos._begin_frame()
    Gizmos._geometry_profile_active = True
    Gizmos.draw_ray((0, 0, 0), (1, 0, 0))

    assert Gizmos._geometry_build_ms == pytest.approx(5.0)
    assert len(Gizmos._draw_batches) == 1
    Gizmos._begin_frame()
    assert Gizmos._geometry_build_ms == 0.0
    assert not Gizmos._geometry_profile_active

    monkeypatch.setattr(gizmo_module.time, "perf_counter", clock)
    Gizmos._geometry_profile_active = True
    with pytest.raises(TypeError, match="NumPy"):
        Gizmos.draw_lines([], [])
    assert Gizmos._geometry_profile_depth == 0
    Gizmos._begin_frame()


# ══════════════════════════════════════════════════════════════════════
# Per-frame state reset
# ══════════════════════════════════════════════════════════════════════

class TestFrameReset:
    def test_begin_frame_resets_color(self):
        Gizmos.color = (1, 0, 0)
        Gizmos._begin_frame()
        assert Gizmos.color == (1.0, 1.0, 1.0)

    def test_begin_frame_resets_matrix(self):
        Gizmos.matrix = [0] * 16
        Gizmos._begin_frame()
        assert Gizmos.matrix is None

    def test_begin_frame_clears_batches(self):
        Gizmos._draw_batches.append(("dummy",))
        Gizmos._begin_frame()
        assert len(Gizmos._draw_batches) == 0

    def test_begin_frame_clears_icons(self):
        Gizmos._icon_entries.append(("dummy",))
        Gizmos._begin_frame()
        assert len(Gizmos._icon_entries) == 0


# ══════════════════════════════════════════════════════════════════════
# draw_line
# ══════════════════════════════════════════════════════════════════════

class TestDrawLine:
    def test_single_line(self):
        Gizmos._begin_frame()
        Gizmos.color = (1, 0, 0)
        Gizmos.draw_line((0, 0, 0), (1, 1, 1))
        assert len(Gizmos._draw_batches) == 1
        verts, indices, matrix = Gizmos._draw_batches[0]
        assert len(verts) == 2
        assert indices == [0, 1]
        # Verify color embedded in vertices
        assert verts[0][3:6] == [1, 0, 0]

    def test_uses_identity_matrix_by_default(self):
        Gizmos._begin_frame()
        Gizmos.draw_line((0, 0, 0), (1, 0, 0))
        _, _, matrix = Gizmos._draw_batches[0]
        assert matrix == Gizmos._identity_matrix


# ══════════════════════════════════════════════════════════════════════
# draw_ray
# ══════════════════════════════════════════════════════════════════════

class TestDrawRay:
    def test_ray_endpoint(self):
        Gizmos._begin_frame()
        Gizmos.draw_ray((1, 2, 3), (10, 0, 0))
        verts, _, _ = Gizmos._draw_batches[0]
        # End = origin + direction
        assert verts[1][0] == pytest.approx(11.0)
        assert verts[1][1] == pytest.approx(2.0)
        assert verts[1][2] == pytest.approx(3.0)


# ══════════════════════════════════════════════════════════════════════
# draw_icon
# ══════════════════════════════════════════════════════════════════════

class TestDrawIcon:
    def test_icon_entry(self):
        Gizmos._begin_frame()
        Gizmos.color = (0, 1, 0)
        Gizmos.draw_icon((5, 5, 5), 42)
        assert len(Gizmos._icon_entries) == 1
        pos, obj_id, color, icon_kind = Gizmos._icon_entries[0]
        assert pos == (5, 5, 5)
        assert obj_id == 42
        assert color == (0, 1, 0)
        assert icon_kind == 0  # ICON_KIND_DEFAULT

    def test_icon_custom_color(self):
        Gizmos._begin_frame()
        Gizmos.draw_icon((0, 0, 0), 1, color=(1, 0, 0))
        _, _, color, _ = Gizmos._icon_entries[0]
        assert color == (1, 0, 0)


# ══════════════════════════════════════════════════════════════════════
# draw_wire_cube
# ══════════════════════════════════════════════════════════════════════

class TestDrawWireCube:
    def test_produces_8_verts_24_indices(self):
        Gizmos._begin_frame()
        Gizmos.draw_wire_cube((0, 0, 0), (2, 2, 2))
        verts, indices, _ = Gizmos._draw_batches[0]
        assert len(verts) == 8
        assert len(indices) == 24  # 12 edges × 2 indices


# ══════════════════════════════════════════════════════════════════════
# draw_wire_sphere (Python fallback)
# ══════════════════════════════════════════════════════════════════════

class TestDrawWireSphere:
    def test_generates_geometry(self):
        Gizmos._begin_frame()
        Gizmos.draw_wire_sphere((0, 0, 0), 1.0, segments=12)
        assert len(Gizmos._draw_batches) >= 1
        verts, indices, _ = Gizmos._draw_batches[0]
        assert len(verts) > 0
        assert len(indices) > 0

    def test_respects_current_color(self):
        Gizmos._begin_frame()
        Gizmos.color = (0.5, 0.5, 0.5)
        Gizmos.draw_wire_sphere((0, 0, 0), 1.0, segments=8)
        verts, _, _ = Gizmos._draw_batches[0]
        # Color channels in vertex data
        assert verts[0][3] == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════
# Custom matrix
# ══════════════════════════════════════════════════════════════════════

class TestCustomMatrix:
    def test_custom_matrix_used(self):
        Gizmos._begin_frame()
        custom = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]
        Gizmos.matrix = custom
        Gizmos.draw_line((0, 0, 0), (1, 0, 0))
        _, _, matrix = Gizmos._draw_batches[0]
        assert matrix == custom

    def test_custom_matrix_is_snapshotted_between_draws(self):
        Gizmos._begin_frame()
        matrix = list(Gizmos._identity_matrix)
        Gizmos.matrix = matrix
        Gizmos.draw_line((0, 0, 0), (1, 0, 0))
        matrix[12] = 5.0
        Gizmos.draw_line((0, 1, 0), (1, 1, 0))

        _, _, first = Gizmos._draw_batches[0]
        _, _, second = Gizmos._draw_batches[1]
        assert first[12] == 0.0
        assert second[12] == 5.0


class TestParticleEmitterShapes:
    def test_authored_shapes_generate_distinct_gizmo_geometry(self):
        shapes = (
            EmitterShape(EmitterShapeKind.POINT),
            EmitterShape(EmitterShapeKind.SPHERE, radius=2.0),
            EmitterShape(EmitterShapeKind.BOX, dimensions=(2.0, 3.0, 4.0)),
            EmitterShape(EmitterShapeKind.CONE, radius=1.5, angle_degrees=35.0),
            EmitterShape(EmitterShapeKind.MESH),
            EmitterShape(EmitterShapeKind.SDF, sdf_interface="shape-field"),
        )

        batch_counts = []
        component = ParticleSystem()
        for shape in shapes:
            Gizmos._begin_frame()
            component._draw_emitter_shape_gizmo(Gizmos, shape)
            batch_counts.append(len(Gizmos._draw_batches))

        assert all(count > 0 for count in batch_counts)
        assert batch_counts[3] > batch_counts[1]

    def test_manual_bounds_use_component_local_space_and_restore_gizmo_state(self):
        transform_matrix = [
            2.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 0.0,
            5.0, 6.0, 7.0, 1.0,
        ]
        component = ParticleSystem()
        component._particle_metadata = SimpleNamespace(emitters=())
        component._try_get_transform = lambda: SimpleNamespace(
            local_to_world_matrix=lambda: transform_matrix
        )
        component.bounds_mode = ParticleBoundsMode.MANUAL
        component.manual_bounds_center = Vector3(1.0, 2.0, 3.0)
        component.manual_bounds_size = Vector3(-4.0, 6.0, 8.0)

        Gizmos._begin_frame()
        original_matrix = [1.0] * 16
        original_color = (0.1, 0.2, 0.3)
        Gizmos.matrix = original_matrix
        Gizmos.color = original_color

        component.on_draw_gizmos_selected()

        assert len(Gizmos._draw_batches) == 1
        vertices, indices, matrix = Gizmos._draw_batches[0]
        assert len(vertices) == 8
        assert len(indices) == 24
        assert matrix == transform_matrix
        positions = [vertex[:3] for vertex in vertices]
        assert [min(axis) for axis in zip(*positions)] == pytest.approx(
            [-1.0, -1.0, -1.0]
        )
        assert [max(axis) for axis in zip(*positions)] == pytest.approx(
            [3.0, 5.0, 7.0]
        )
        assert Gizmos.matrix == original_matrix
        assert Gizmos.color == original_color


class TestCameraGizmos:
    def test_invalid_camera_projection_is_not_suppressed(self):
        class InvalidCamera:
            @property
            def projection_matrix(self):
                raise RuntimeError("stale camera projection")

        component = Camera()
        component._get_bound_native_component = lambda: InvalidCamera()

        with pytest.raises(RuntimeError, match="stale camera projection"):
            component.on_draw_gizmos_selected()


class TestGizmosCollectorSelectionCache:
    def test_icon_cache_spans_every_loaded_scene(self):
        collector = GizmosCollector()

        class Object:
            def __init__(self, identity, has_component):
                self.identity = identity
                self.has_component = has_component

            def get_cpp_component(self, _type_name):
                return object() if self.has_component else None

        first = Object("first", True)
        ignored = Object("ignored", False)
        second = Object("second", True)
        scenes = (
            SimpleNamespace(get_all_objects=lambda: [first, ignored]),
            SimpleNamespace(get_all_objects=lambda: [second]),
        )

        assert collector._get_icon_instances(scenes, "Light") == [first, second]

    def test_selected_subtree_is_reused_until_selection_or_scene_changes(self, monkeypatch):
        collector = GizmosCollector()
        scene = SimpleNamespace()
        calls = []

        def build(_scene, selected_id):
            calls.append(selected_id)
            return {selected_id, selected_id + 1}

        monkeypatch.setattr(collector, "_build_ancestor_set", build)

        assert collector._get_selected_ancestor_ids(scene, 7) == frozenset({7, 8})
        assert collector._get_selected_ancestor_ids(scene, 7) == frozenset({7, 8})
        assert calls == [7]

        assert collector._get_selected_ancestor_ids(scene, 9) == frozenset({9, 10})
        assert calls == [7, 9]

        collector.invalidate_cache()
        assert collector._get_selected_ancestor_ids(scene, 9) == frozenset({9, 10})
        assert calls == [7, 9, 9]

    def test_no_selection_does_not_cross_the_scene_binding(self, monkeypatch):
        collector = GizmosCollector()

        def unexpected_walk(*_args):
            raise AssertionError("no selection walk")

        monkeypatch.setattr(collector, "_build_ancestor_set", unexpected_walk)

        assert collector._get_selected_ancestor_ids(SimpleNamespace(), 0) == frozenset()


class TestGizmosCollectorActiveHierarchy:
    def test_inactive_hierarchy_blocks_selected_particle_system_gizmo(
        self, monkeypatch
    ):
        from Infernux.components.component import InxComponent
        import Infernux.lib as lib

        hierarchy_reads = []
        parent = SimpleNamespace(active_in_hierarchy=False)

        class GameObject:
            id = 51

            @property
            def active_in_hierarchy(self):
                hierarchy_reads.append("particle")
                return bool(parent.active_in_hierarchy)

            def get_children(self):
                return []

        game_object = GameObject()
        particle = ParticleSystem()
        particle._try_get_game_object = lambda: game_object
        callbacks = []
        particle._call_on_draw_gizmos = lambda: callbacks.append("always")
        particle._call_on_draw_gizmos_selected = lambda: callbacks.append("selected")
        scene = SimpleNamespace(
            world_id=1,
            structure_version=1,
            find_by_id=lambda object_id: game_object if object_id == 51 else None,
            get_all_objects=lambda: [game_object],
        )

        class SceneManager:
            @staticmethod
            def instance():
                return SimpleNamespace(
                    scene_count=1,
                    get_scene_at=lambda index: scene if index == 0 else None,
                )

        class Native:
            def __init__(self):
                self.icon_uploads = 0

            def clear_component_gizmos(self):
                pass

            def clear_component_cpu_gizmos(self):
                pass

            def upload_component_resident_gizmos(self, descriptors):
                assert descriptors == []

            def clear_component_gizmo_icons(self):
                pass

            def upload_component_gizmos(self, *_args):
                raise AssertionError("inactive ParticleSystem submitted gizmo geometry")

            def upload_component_gizmo_icons(self, *_args):
                self.icon_uploads += 1

        native = Native()
        engine = SimpleNamespace(
            get_native_engine=lambda: native,
            get_selected_object_id=lambda: 51,
        )
        collector = GizmosCollector()
        collector._builtin_registry = {}
        monkeypatch.setattr(lib, "SceneManager", SceneManager)
        monkeypatch.setattr(
            InxComponent, "_active_instances", {51: [particle]}
        )

        collector.collect_and_upload(engine)

        assert callbacks == []
        assert native.icon_uploads == 0
        assert hierarchy_reads


class TestGizmosCollectorWorkGates:
    @staticmethod
    def _scene_manager(monkeypatch, scene):
        import Infernux.lib as lib

        class SceneManager:
            @staticmethod
            def instance():
                return SimpleNamespace(
                    scene_count=1,
                    get_scene_at=lambda index: scene if index == 0 else None,
                )

        monkeypatch.setattr(lib, "SceneManager", SceneManager)

    def test_unselected_icon_only_gizmo_never_creates_wrapper(self, monkeypatch):
        from Infernux.components.component import InxComponent

        created = []
        callbacks = []

        class Wrapper:
            _gizmo_icon_color = (1.0, 1.0, 1.0)
            _gizmo_icon_kind = 1
            _always_show = False
            on_draw_gizmos = InxComponent.on_draw_gizmos

            def on_draw_gizmos_selected(self):
                pass

            @classmethod
            def _get_or_create_wrapper(cls, _component, _game_object):
                created.append(True)
                return SimpleNamespace(
                    _always_show=False,
                    _call_on_draw_gizmos=lambda: None,
                    _call_on_draw_gizmos_selected=lambda: callbacks.append(True),
                    _invalidate_native_binding=lambda: None,
                )

        game_object = SimpleNamespace(
            id=73,
            active_in_hierarchy=True,
            get_cpp_component=lambda name: SimpleNamespace(enabled=True)
            if name == "Light" else None,
            get_transform=lambda: SimpleNamespace(
                position=SimpleNamespace(x=0.0, y=1.0, z=0.0)
            ),
            get_children=lambda: [],
        )
        scene = SimpleNamespace(
            world_id=3,
            structure_version=1,
            find_by_id=lambda object_id: game_object if object_id == 73 else None,
            get_all_objects=lambda: [game_object],
        )
        self._scene_manager(monkeypatch, scene)

        class Native:
            def __init__(self):
                self.icon_uploads = 0

            def upload_component_gizmo_icons(self, *_args):
                self.icon_uploads += 1

        native = Native()
        selected = [0]
        collector = GizmosCollector()
        collector._builtin_registry = {"Light": Wrapper}
        engine = SimpleNamespace(
            get_native_engine=lambda: native,
            get_selected_object_id=lambda: selected[0],
        )

        collector.collect_and_upload(engine)
        assert native.icon_uploads == 1
        assert created == []
        assert callbacks == []

        selected[0] = 73
        collector.collect_and_upload(engine)
        assert native.icon_uploads == 2
        assert created == [True]
        assert callbacks == [True]

        class Replacement(Wrapper):
            @classmethod
            def _get_or_create_wrapper(cls, _component, _game_object):
                return SimpleNamespace(
                    _always_show=False,
                    _call_on_draw_gizmos=lambda: None,
                    _call_on_draw_gizmos_selected=lambda: callbacks.append("reloaded"),
                    _invalidate_native_binding=lambda: None,
                )

        collector._builtin_registry["Light"] = Replacement
        collector.collect_and_upload(engine)
        assert callbacks == [True, "reloaded"]

    def test_selected_scene_lookup_reuses_structure_and_refreshes_after_change(
        self, monkeypatch
    ):
        lookups = []
        game_object = SimpleNamespace(id=73, get_children=lambda: [])
        scene = SimpleNamespace(world_id=3, structure_version=1)

        def find_by_id(object_id):
            lookups.append(object_id)
            return game_object if object_id == 73 else None

        scene.find_by_id = find_by_id
        self._scene_manager(monkeypatch, scene)
        native = SimpleNamespace()
        engine = SimpleNamespace(
            get_native_engine=lambda: native,
            get_selected_object_id=lambda: 73,
        )
        collector = GizmosCollector()
        collector._builtin_registry = {}

        collector.collect_and_upload(engine)
        initial_lookups = len(lookups)
        assert initial_lookups > 0
        collector.collect_and_upload(engine)
        assert len(lookups) == initial_lookups

        scene.structure_version += 1
        collector.collect_and_upload(engine)
        assert len(lookups) > initial_lookups

    def test_release_profile_off_skips_timers_and_helper_wrapping(self, monkeypatch):
        import Infernux.gizmos.collector as collector_module
        import Infernux.gizmos.gizmos as gizmo_module

        monkeypatch.setattr(collector_module, "_GEOMETRY_PROFILE_COMPILED", False)
        monkeypatch.setattr(gizmo_module, "_GEOMETRY_PROFILE_COMPILED", False)

        def no_clock():
            raise AssertionError("profile-off collection read a CPU timer")

        monkeypatch.setattr(collector_module.time, "perf_counter", no_clock)
        original = lambda: None
        assert gizmo_module._profile_geometry_helper(original) is original

        scene = SimpleNamespace(world_id=3, structure_version=1)
        self._scene_manager(monkeypatch, scene)
        collector = GizmosCollector()
        collector._builtin_registry = {}
        collector.collect_and_upload(SimpleNamespace(
            get_native_engine=lambda: SimpleNamespace(),
            get_selected_object_id=lambda: 0,
        ))
        assert collector.last_observation.total_ms == 0.0
        assert not collector.last_observation.timing_enabled

    def test_disabled_large_python_gizmo_stops_geometry_and_upload_work(
        self, monkeypatch
    ):
        from Infernux.components.component import InxComponent
        import Infernux.gizmos.collector as collector_module

        from Infernux.gizmos import gizmos as gizmo_module

        profile_enabled = bool(gizmo_module._GEOMETRY_PROFILE_COMPILED)

        point_count = 4096
        points = np.zeros((point_count, 3), dtype=np.float32)
        points[:, 0] = np.arange(point_count, dtype=np.float32)
        edges = np.column_stack(
            (
                np.arange(point_count - 1, dtype=np.uint32),
                np.arange(1, point_count, dtype=np.uint32),
            )
        )
        callback_count = 0

        class Probe(InxComponent):
            def on_draw_gizmos(self):
                nonlocal callback_count
                callback_count += 1
                Gizmos.color = (0.25, 0.5, 0.75)
                Gizmos.matrix = [
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    7, 8, 9, 1,
                ]
                Gizmos.draw_lines(points, edges)

        probe = Probe()
        game_object = SimpleNamespace(id=71, get_children=lambda: [])
        scene = SimpleNamespace(
            world_id=2,
            structure_version=1,
            find_by_id=lambda object_id: game_object if object_id == 71 else None,
            get_all_objects=lambda: [game_object],
        )
        self._scene_manager(monkeypatch, scene)
        monkeypatch.setattr(
            collector_module,
            "component_owner_is_active_in_hierarchy",
            lambda _component: True,
        )
        monkeypatch.setattr(InxComponent, "_active_instances", {71: [probe]})

        class Native:
            def __init__(self):
                self.cpu_uploads = []
                self.cpu_clears = 0
                self.resident_uploads = 0
                self.icon_uploads = 0

            def upload_component_gizmos(self, *payload):
                self.cpu_uploads.append(payload)

            def clear_component_cpu_gizmos(self):
                self.cpu_clears += 1

            def upload_component_resident_gizmos(self, _descriptors):
                self.resident_uploads += 1

            def upload_component_gizmo_icons(self, *_payload):
                self.icon_uploads += 1

            def clear_component_gizmo_icons(self):
                raise AssertionError("an already empty icon buffer was cleared")

            def clear_component_gizmos(self):
                raise AssertionError("the loaded world must use transition clears")

        native = Native()
        engine = SimpleNamespace(
            get_native_engine=lambda: native,
            get_selected_object_id=lambda: 0,
        )
        collector = GizmosCollector()
        collector._builtin_registry = {}

        collector.collect_and_upload(engine)
        first = collector.last_observation
        assert callback_count == 1
        assert first.python_callbacks == 1
        assert first.timing_enabled is profile_enabled
        if profile_enabled:
            assert first.callback_ms >= first.geometry_build_ms > 0.0
            assert first.callback_non_helper_ms == pytest.approx(
                first.callback_ms - first.geometry_build_ms
            )
        else:
            assert first.callback_ms == 0.0
            assert first.callback_non_helper_ms == 0.0
            assert first.geometry_build_ms == 0.0
        assert not Gizmos._geometry_profile_active
        assert first.cpu_vertices == point_count
        assert first.cpu_line_indices == (point_count - 1) * 2
        assert first.cpu_draws == 1
        assert native.resident_uploads == 0
        assert native.icon_uploads == 0
        uploaded_vertices = native.cpu_uploads[0][0].reshape(-1, 6)
        assert uploaded_vertices[0, 3:6] == pytest.approx((0.25, 0.5, 0.75))
        uploaded_descriptor = native.cpu_uploads[0][3].reshape(-1, 18)[0]
        assert uploaded_descriptor[14:18] == pytest.approx((7, 8, 9, 1))

        probe._enabled = False
        collector.collect_and_upload(engine)
        disabled = collector.last_observation
        assert callback_count == 1
        assert disabled.skipped_disabled == 1
        assert disabled.python_callbacks == 0
        assert disabled.cpu_vertices == 0
        assert native.cpu_clears == 1

        # Remaining disabled must stay a zero-upload steady state, rather than
        # paying a Python/native clear every frame.
        collector.collect_and_upload(engine)
        assert callback_count == 1
        assert native.cpu_clears == 1
        assert len(native.cpu_uploads) == 1

    @pytest.mark.parametrize(
        ("scene_visible", "show_gizmos", "expected"),
        [(False, True, "clear"), (True, False, "clear"), (True, True, "tick")],
    )
    def test_scene_visibility_and_global_switch_gate_before_collection(
        self, scene_visible, show_gizmos, expected
    ):
        from Infernux.engine.engine import Engine
        from Infernux.engine.runtime_change_journal import RuntimeFrameBarrier

        events = []
        engine = Engine.__new__(Engine)
        engine._runtime_scene_manager = None
        engine._runtime_scheduler = SimpleNamespace(
            consume_native_barrier=lambda _barrier: None
        )
        engine._scene_view_visible = scene_visible
        engine._show_gizmos = show_gizmos
        engine._gizmo_collect_interval_edit = 0.0
        engine._gizmo_collect_interval_play = 0.0
        engine._next_gizmo_collect_time = 0.0
        engine._tick_gizmos = lambda: events.append("tick")
        engine._clear_uploaded_gizmos = lambda: events.append("clear")

        engine._consume_runtime_frame_barrier(RuntimeFrameBarrier.RENDER_EXTRACTION)

        assert events == [expected]

    def test_retirement_drops_python_frame_and_clears_native_once(self):
        collector = GizmosCollector()
        collector._cpu_uploaded = True
        collector._resident_uploaded = True
        collector._icons_uploaded = True
        Gizmos._begin_frame()
        Gizmos.draw_line((0, 0, 0), (1, 0, 0))
        Gizmos.draw_icon((0, 0, 0), 9)
        clears = []
        native = SimpleNamespace(
            clear_component_gizmos=lambda: clears.append("all")
        )

        collector.retire_uploaded(native)
        collector.retire_uploaded(native)

        assert clears == ["all"]
        assert Gizmos._draw_batches == []
        assert Gizmos._resident_draw_batches == []
        assert Gizmos._icon_entries == []
        assert collector.last_observation.total_ms == 0.0

    def test_disabled_builtin_skips_icon_transform_and_wrapper_creation(
        self, monkeypatch
    ):
        from Infernux.components.component import InxComponent

        class Wrapper:
            _gizmo_icon_color = (1.0, 1.0, 1.0)
            _gizmo_icon_kind = 1
            _always_show = True
            on_draw_gizmos_selected = InxComponent.on_draw_gizmos_selected

            def on_draw_gizmos(self):
                pass

            @classmethod
            def _get_or_create_wrapper(cls, *_args):
                raise AssertionError("disabled component created a Python wrapper")

        class GameObject:
            id = 73
            active_in_hierarchy = True

            @staticmethod
            def get_cpp_component(type_name):
                return SimpleNamespace(enabled=False) if type_name == "Light" else None

            @staticmethod
            def get_transform():
                raise AssertionError("disabled component read its world transform")

            @staticmethod
            def get_children():
                return []

        game_object = GameObject()
        scene = SimpleNamespace(
            world_id=3,
            structure_version=1,
            find_by_id=lambda object_id: game_object if object_id == 73 else None,
            get_all_objects=lambda: [game_object],
        )
        self._scene_manager(monkeypatch, scene)

        class Native:
            def upload_component_gizmos(self, *_args):
                raise AssertionError("disabled built-in uploaded geometry")

            def upload_component_resident_gizmos(self, *_args):
                raise AssertionError("disabled built-in uploaded resident geometry")

            def upload_component_gizmo_icons(self, *_args):
                raise AssertionError("disabled built-in uploaded an icon")

            def clear_component_cpu_gizmos(self):
                raise AssertionError("empty initial CPU state was cleared")

            def clear_component_gizmo_icons(self):
                raise AssertionError("empty initial icon state was cleared")

            def clear_component_gizmos(self):
                raise AssertionError("loaded world used a full clear")

        collector = GizmosCollector()
        collector._builtin_registry = {"Light": Wrapper}
        collector.collect_and_upload(
            SimpleNamespace(
                get_native_engine=lambda: Native(),
                get_selected_object_id=lambda: 0,
            )
        )

        observation = collector.last_observation
        assert observation.skipped_disabled == 1
        assert observation.builtin_callbacks == 0
        assert observation.icons == 0

    @pytest.mark.parametrize("type_name", ["Camera", "Light"])
    def test_inactive_hierarchy_blocks_builtin_icon_and_selected_gizmo(
        self, monkeypatch, type_name
    ):
        from Infernux.components.component import InxComponent
        import Infernux.lib as lib

        callbacks = []

        class Wrapper:
            _gizmo_icon_color = (1.0, 1.0, 1.0)
            _gizmo_icon_kind = 0
            _always_show = False
            on_draw_gizmos = InxComponent.on_draw_gizmos

            def on_draw_gizmos_selected(self):
                pass

            @classmethod
            def _get_or_create_wrapper(cls, _component, _game_object):
                return SimpleNamespace(
                    _always_show=False,
                    _call_on_draw_gizmos=lambda: callbacks.append("always"),
                    _call_on_draw_gizmos_selected=lambda: callbacks.append("selected"),
                )

        hierarchy_reads = []
        parent = SimpleNamespace(active_in_hierarchy=False)

        class GameObject:
            id = 42

            @property
            def active_in_hierarchy(self):
                hierarchy_reads.append(type_name)
                return bool(parent.active_in_hierarchy)

            def get_cpp_component(self, requested):
                return SimpleNamespace(enabled=True) if requested == type_name else None

            def get_transform(self):
                return SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0, z=0.0))

            def get_children(self):
                return []

        game_object = GameObject()
        scene = SimpleNamespace(
            world_id=1,
            structure_version=1,
            find_by_id=lambda object_id: game_object if object_id == 42 else None,
            get_all_objects=lambda: [game_object],
        )

        class SceneManager:
            @staticmethod
            def instance():
                return SimpleNamespace(
                    scene_count=1,
                    get_scene_at=lambda index: scene if index == 0 else None,
                )

        class Native:
            def __init__(self):
                self.icon_uploads = 0

            def clear_component_gizmos(self):
                pass

            def clear_component_cpu_gizmos(self):
                pass

            def upload_component_resident_gizmos(self, descriptors):
                assert descriptors == []

            def clear_component_gizmo_icons(self):
                pass

            def upload_component_gizmos(self, *_args):
                raise AssertionError("inactive selected object submitted gizmo geometry")

            def upload_component_gizmo_icons(self, *_args):
                self.icon_uploads += 1

        native = Native()
        engine = SimpleNamespace(
            get_native_engine=lambda: native,
            get_selected_object_id=lambda: 42,
        )
        collector = GizmosCollector()
        collector._builtin_registry = {type_name: Wrapper}
        monkeypatch.setattr(lib, "SceneManager", SceneManager)

        collector.collect_and_upload(engine)

        assert callbacks == []
        assert native.icon_uploads == 0
        assert hierarchy_reads
