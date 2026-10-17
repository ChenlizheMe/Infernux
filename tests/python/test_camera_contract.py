"""Camera authoring writes must preserve a valid, serializable native state."""

import math
import pytest

from infernux import lib


@pytest.fixture
def camera(scene):
    wrapper = scene.create_game_object("CameraContract").add_component("Camera")
    return wrapper._require_cpp_component()


@pytest.mark.parametrize(
    "field,value",
    [("field_of_view", float("nan")), ("near_clip", 5000.0), ("projection_mode", lib.CameraProjection(99))],
)
def test_invalid_native_camera_setter_preserves_complete_document(camera, field, value):
    before = camera.serialize_document()
    with pytest.raises(ValueError, match="Camera"):
        setattr(camera, field, value)
    assert camera.serialize_document() == before


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
    for mask in (0, 1 << 3, 0xFFFFFFFF):
        camera.culling_mask = mask
        assert camera.culling_mask == mask
        assert camera._require_cpp_component().culling_mask == mask
        assert camera.serialize_document()["cullingMask"] == mask
    before = camera.serialize_document()
    with pytest.raises((TypeError, ValueError, OverflowError)):
        camera.culling_mask = -1
    assert camera.serialize_document() == before


@pytest.mark.parametrize("destroy_owner", [False, True])
def test_removing_preferred_camera_clears_borrowed_scene_reference(engine, scene, destroy_owner):
    owner = scene.create_game_object("PreferredCameraOwner")
    preferred = owner.add_component("Camera")
    scene.main_camera = preferred
    remaining = scene.create_game_object("RemainingCamera").add_component("Camera")
    expected = remaining.component_id
    assert preferred.component_id in engine.renderer_frame_snapshot["game_camera_ids"]
    if destroy_owner:
        scene.destroy_game_object(owner)
        scene.process_pending_destroys()
    else:
        assert owner.remove_component(preferred)
    assert scene.main_camera is None
    assert scene.effective_game_camera.component_id == expected
    assert expected in engine.renderer_frame_snapshot["game_camera_ids"]


def test_multi_camera_clip_edit_clamps_each_target_before_native_publication(scene):
    from infernux.components.builtin.camera import Camera
    from infernux.engine.ui.inspector_components import _apply_multi_builtin_change
    from infernux.engine.undo import UndoManager

    first = scene.create_game_object("FirstCamera").add_component("Camera")
    second = scene.create_game_object("SecondCamera").add_component("Camera")
    first.set_clip_planes(0.01, 100.0)
    second.set_clip_planes(10.0, 100.0)
    before = (first.serialize_document(), second.serialize_document())
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        _apply_multi_builtin_change(
            (first, second),
            ("far_clip", "far_clip"),
            Camera.far_clip.metadata,
            5.0,
        )
        assert first.far_clip == pytest.approx(5.0)
        assert second.far_clip == pytest.approx(10.001)
        assert len(manager.action_journal.applied_entries()) == 1
        manager.undo()
        assert (first.serialize_document(), second.serialize_document()) == before
        _apply_multi_builtin_change(
            (first, second),
            ("far_clip", "far_clip"),
            Camera.far_clip.metadata,
            50.0,
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
