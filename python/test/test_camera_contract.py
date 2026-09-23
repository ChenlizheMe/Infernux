"""Camera authoring writes must preserve a valid, serializable native state."""

import math
import pytest

from Infernux import lib


@pytest.fixture
def camera(scene):
    wrapper = scene.create_game_object("CameraContract").add_component("Camera")
    return wrapper._require_cpp_component()


@pytest.mark.parametrize("field,value", [
    ("field_of_view", 0.0), ("field_of_view", 180.0), ("field_of_view", float("nan")),
    ("aspect_ratio", 0.0), ("aspect_ratio", -1.0), ("aspect_ratio", float("inf")),
    ("orthographic_size", 0.0), ("orthographic_size", float("nan")),
    ("near_clip", 0.0), ("near_clip", 5000.0), ("near_clip", float("nan")),
    ("far_clip", 0.01), ("far_clip", float("inf")), ("depth", float("nan")),
    ("projection_mode", lib.CameraProjection(99)),
    ("clear_flags", lib.CameraClearFlags(99)),
])
def test_invalid_native_camera_setter_preserves_complete_document(camera, field, value):
    before = camera.serialize_document()
    with pytest.raises(ValueError, match="Camera"):
        setattr(camera, field, value)
    assert camera.serialize_document() == before


def test_clip_planes_can_move_atomically_past_the_old_far_plane(camera):
    camera.set_clip_planes(6000.0, 10000.0)
    assert (camera.near_clip, camera.far_clip) == (6000.0, 10000.0)
    before = camera.serialize_document()
    for near, far in [(7000.0, 6500.0), (0.0, 100.0), (1.0, float("nan"))]:
        with pytest.raises(ValueError, match="Camera"):
            camera.set_clip_planes(near, far)
        assert camera.serialize_document() == before


def test_tiny_positive_viewport_aspect_keeps_the_existing_floor(camera):
    camera.aspect_ratio = 0.001
    assert camera.aspect_ratio == pytest.approx(0.01)
    assert camera.deserialize_document(camera.serialize_document())


def test_document_clip_update_is_atomic_and_depth_invalidates_order(camera, scene):
    candidate = camera.serialize_document()
    candidate.update(nearClip=6000.0, farClip=10000.0, depth=7.0)
    before = camera.serialize_document()
    version = scene.structure_version
    camera.validate_document(candidate)
    assert camera.serialize_document() == before
    assert scene.structure_version == version
    assert camera.deserialize_document(candidate)
    assert (camera.near_clip, camera.far_clip, camera.depth) == (6000.0, 10000.0, 7.0)
    assert scene.structure_version > version
    before = camera.serialize_document()
    invalid = dict(before, nearClip=20000.0, depth=11.0)
    version = scene.structure_version
    with pytest.raises(ValueError, match="Camera clip planes"):
        camera.validate_document(invalid)
    assert not camera.deserialize_document(invalid)
    assert camera.serialize_document() == before
    assert scene.structure_version == version


def test_wrapper_atomic_clip_update_keeps_its_native_binding_after_rejection(scene):
    camera = scene.create_game_object("WrapperClip").add_component("Camera")
    camera.set_clip_planes(6000.0, 10000.0)
    camera.near_clip = 12000.0
    assert camera.near_clip == pytest.approx(9999.999)
    camera.far_clip = -10.0
    assert camera.far_clip == pytest.approx(10000.0)
    camera.set_clip_planes(1.0, 50.0)
    assert (camera.near_clip, camera.far_clip) == (1.0, 50.0)


def test_physical_camera_matches_unity_property_model_and_sensor_presets(scene):
    from Infernux.components.builtin.camera import Camera

    camera = scene.create_game_object("UnityPhysicalCamera").add_component("Camera")
    assert not hasattr(lib.CameraProjection, "Physical")
    assert camera.projection_mode == lib.CameraProjection.Perspective
    assert camera.use_physical_properties is False
    assert Camera.field_of_view.metadata.visible_when(camera) is True
    camera.use_physical_properties = True
    assert Camera.field_of_view.metadata.visible_when(camera) is False
    assert Camera.focal_length.metadata.visible_when(camera) is True
    assert camera.iso == 200
    assert camera.shutter_speed == pytest.approx(0.005)
    assert camera.aperture == pytest.approx(16.0)
    assert camera.focus_distance == pytest.approx(10.0)
    assert camera.blade_count == 5
    assert tuple(camera.curvature) == pytest.approx((2.0, 11.0))
    assert camera.barrel_clipping == pytest.approx(0.25)
    assert camera.anamorphism == pytest.approx(0.0)

    camera._require_cpp_component().sensor_type = lib.CameraSensorType.Film70mmImax
    assert tuple(camera.sensor_size) == pytest.approx((70.41, 52.63))
    assert camera.sensor_type == lib.CameraSensorType.Film70mmImax
    camera.sensor_size = lib.Vector2(36.0, 24.0)
    assert camera.sensor_type == lib.CameraSensorType.Custom
    document = camera.serialize_document()
    assert "sensorType" not in document
    assert document["sensorSize"] == pytest.approx([36.0, 24.0])
    with pytest.raises(AttributeError, match="read-only"):
        camera.sensor_type = lib.CameraSensorType.Film8mm


def test_physical_camera_projection_stays_finite_and_continuous_while_dragging(scene):
    camera = scene.create_game_object("PhysicalCameraDrag").add_component("Camera")
    camera.use_physical_properties = True
    camera.focal_length = 35.0
    previous = camera.projection_matrix.copy()
    structure_version = scene.structure_version
    for focal_length in [35.25 + index * 0.25 for index in range(120)]:
        camera.focal_length = focal_length
        current = camera.projection_matrix.copy()
        assert current.shape == (4, 4)
        assert all(math.isfinite(float(value)) for value in current.flat)
        assert abs(float(current[1, 1] - previous[1, 1])) < 0.05
        previous = current
    assert scene.structure_version == structure_version


def test_wrapper_culling_mask_writes_the_native_authoritative_field(scene):
    camera = scene.create_game_object("LayerCamera").add_component("Camera")
    for mask in (0, 1 << 3, 0xffffffff):
        camera.culling_mask = mask
        assert camera.culling_mask == mask
        assert camera._require_cpp_component().culling_mask == mask
        assert camera.serialize_document()["cullingMask"] == mask
    before = camera.serialize_document()
    with pytest.raises((TypeError, ValueError, OverflowError)):
        camera.culling_mask = -1
    assert camera.serialize_document() == before


def test_camera_inspector_uses_named_layer_popup_instead_of_numeric_mask(scene, monkeypatch):
    """The authoring surface must never make users type the 32-bit mask."""
    import Infernux.engine.ui.inspector_components as inspector_components
    import Infernux.engine.ui.inspector_utils as inspector_utils

    wrapper = scene.create_game_object("LayerInspectorCamera").add_component("Camera")
    wrapper.culling_mask = (1 << 0) | (1 << 2)

    class _LayerManager:
        @staticmethod
        def instance():
            return _LayerManager()

        def get_all_layers(self):
            return ["Gameplay", "UI", "Effects"]

    class _Context:
        def __init__(self):
            self.labels = []
            self.buttons = []
            self.checkboxes = []

        def button(self, label):
            self.buttons.append(label)
            return False

        def begin_popup(self, _popup_id):
            return True

        def end_popup(self):
            return None

        def checkbox(self, label, value):
            self.checkboxes.append((label, value))
            return value

        def same_line(self):
            return None

    ctx = _Context()
    def _render_custom_fields(_ctx, comp, _wrapper_cls, **kwargs):
        kwargs["custom_fields"]["culling_mask"](_ctx, comp, 0.0)

    monkeypatch.setattr(
        inspector_components, "render_builtin_via_setters", _render_custom_fields,
    )
    monkeypatch.setattr(inspector_utils, "field_label", lambda *args, **kwargs: None)
    monkeypatch.setattr(inspector_utils, "max_label_w", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(lib, "TagLayerManager", _LayerManager)
    wrapper.render_inspector(ctx)

    assert ctx.buttons == [
        "2 Layers##camera_culling_mask",
        "Everything##camera_culling_everything",
        "Nothing##camera_culling_nothing",
    ]
    assert [label for label, _ in ctx.checkboxes[:3]] == [
        "Gameplay##camera_layer_0", "UI##camera_layer_1", "Effects##camera_layer_2",
    ]
    assert len(ctx.checkboxes) == 32
    assert all("input" not in label.lower() for label, _ in ctx.checkboxes)


def test_physical_camera_rejects_invalid_sensor_and_lens_values(scene):
    camera = scene.create_game_object("PhysicalCameraContract").add_component("Camera")
    before = camera.serialize_document()
    for sensor in (lib.Vector2(0.0, 24.0), lib.Vector2(-1.0, 24.0)):
        with pytest.raises((TypeError, ValueError)):
            camera.sensor_size = sensor
        assert camera.serialize_document() == before
    with pytest.raises(ValueError):
        lib.Vector2(float("nan"), 0.0)
    assert camera.serialize_document() == before


@pytest.mark.parametrize('destroy_owner', [False, True])
def test_removing_preferred_camera_clears_borrowed_scene_reference(engine, scene, destroy_owner):
    owner = scene.create_game_object('PreferredCameraOwner')
    preferred = owner.add_component('Camera')
    scene.main_camera = preferred
    remaining = scene.create_game_object('RemainingCamera').add_component('Camera')
    expected = remaining.component_id
    assert preferred.component_id in engine.renderer_frame_snapshot['game_camera_ids']
    if destroy_owner:
        scene.destroy_game_object(owner)
        scene.process_pending_destroys()
    else:
        assert owner.remove_component(preferred)
    assert scene.main_camera is None
    assert scene.effective_game_camera.component_id == expected
    assert expected in engine.renderer_frame_snapshot['game_camera_ids']


def test_multi_camera_clip_edit_clamps_each_target_before_native_publication(scene):
    from Infernux.components.builtin.camera import Camera
    from Infernux.engine.ui.inspector_components import _apply_multi_builtin_change
    from Infernux.engine.undo import UndoManager

    first = scene.create_game_object("FirstCamera").add_component("Camera")
    second = scene.create_game_object("SecondCamera").add_component("Camera")
    first.set_clip_planes(0.01, 100.0)
    second.set_clip_planes(10.0, 100.0)
    before = (first.serialize_document(), second.serialize_document())
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        _apply_multi_builtin_change(
            (first, second), ("far_clip", "far_clip"), Camera.far_clip.metadata, 5.0,
        )
        assert first.far_clip == pytest.approx(5.0)
        assert second.far_clip == pytest.approx(10.001)
        assert len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        assert (first.serialize_document(), second.serialize_document()) == before
        _apply_multi_builtin_change(
            (first, second), ("far_clip", "far_clip"), Camera.far_clip.metadata, 50.0,
        )
        assert (first.far_clip, second.far_clip) == (50.0, 50.0)
        assert len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        assert (first.serialize_document(), second.serialize_document()) == before
        manager.redo()
        assert (first.far_clip, second.far_clip) == (50.0, 50.0)
    finally:
        manager.clear()
        UndoManager._instance = previous
