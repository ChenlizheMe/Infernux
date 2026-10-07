"""Root motion composes identically across frame partitions and loop seams."""
import base64
import json
import math
from pathlib import Path
import struct

import numpy as np
import pytest

from infernux.components.skeletal_animator import SkeletalAnimator
from infernux.core.anim_state_machine import AnimState, AnimStateMachine
from infernux.core.animation_clip3d import AnimationClip3D, embedded_take_descriptors
from infernux.core.asset_ref import AnimStateMachineRef
from infernux.core.asset_types import read_mesh_import_settings, read_meta_file
from infernux.core.assets import AssetManager
from infernux.lib import Vector3
from model_test_support import remove_model_test_folder


def _write_model(path, start_angle):
    blob, views, accessors = bytearray(), [], []

    def put(fmt, values, kind, count, component=5126, **bounds):
        blob.extend(b"\0" * (-len(blob) % 4))
        data = struct.pack("<" + fmt, *values)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        accessors.append({"bufferView": len(views) - 1, "componentType": component,
                          "type": kind, "count": count, **bounds})
        return len(accessors) - 1

    positions = put("9f", [0, 0, 0, 1, 0, 0, 0, 1, 0], "VEC3", 3, min=[0, 0, 0], max=[1, 1, 0])
    indices = put("3H", [0, 1, 2], "SCALAR", 3, 5123)
    joints = put("12H", [0] * 12, "VEC4", 3, 5123)
    weights = put("12f", [1, 0, 0, 0] * 3, "VEC4", 3)
    identity = put("16f", np.eye(4).ravel(), "MAT4", 1)
    times = put("3f", [0, 1, 2], "SCALAR", 3, min=[0], max=[2])
    translation = put("9f", [3, -2, 1, 4, -2, 1, 5, -2, 1], "VEC3", 3)
    rotations = []
    for angle in (start_angle, start_angle + 45, start_angle + 90):
        half = math.radians(angle) / 2
        rotations.extend((0, 0, math.sin(half), math.cos(half)))
    rotation = put("12f", rotations, "VEC4", 3)
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(blob), "uri": "data:application/octet-stream;base64," +
                     base64.b64encode(blob).decode()}],
        "bufferViews": views, "accessors": accessors,
        "meshes": [{"name": "Triangle", "primitives": [{"attributes": {
            "POSITION": positions, "JOINTS_0": joints, "WEIGHTS_0": weights}, "indices": indices}]}],
        "skins": [{"name": "Rig", "inverseBindMatrices": identity, "skeleton": 0, "joints": [0]}],
        "nodes": [{"name": "Root"}, {"name": "Geometry", "mesh": 0, "skin": 0}],
        "scenes": [{"nodes": [0, 1]}], "scene": 0,
        "animations": [{"name": "MoveTurn", "samplers": [
            {"input": times, "output": translation, "interpolation": "LINEAR"},
            {"input": times, "output": rotation, "interpolation": "LINEAR"}],
            "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}},
                         {"sampler": 1, "target": {"node": 0, "path": "rotation"}}]}],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


@pytest.fixture(params=[(0., 1.), (30., .5)], ids=["identity-root", "offset-root-scaled"])
def motion_model(engine, tmp_path, monkeypatch, request):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / "MoveTurn.gltf"
    source.parent.mkdir()
    angle, scale = request.param
    _write_model(source, angle)
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported.succeeded, imported.error
    try:
        settings = read_mesh_import_settings(str(source))
        settings.animation_apply_root_motion = True
        settings.animation_loop_time = True
        settings.animation_reference_pose = "first_frame"
        settings.scale_factor = scale
        result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        assert result.succeeded, result.error
        descriptor, = embedded_take_descriptors(read_meta_file(str(source)))
        clip = AnimationClip3D.load(database.get_path_from_guid(descriptor["guid"]))
        assert clip.apply_root_motion and clip.duration_hint == pytest.approx(2.)
        yield database, source, imported.guid, descriptor["guid"], clip, angle, scale
    finally:
        remove_model_test_folder(database, source.parent)


def _matrix(time, start_angle):
    angle = math.radians(start_angle + 45. * time)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0, 3 + time], [s, c, 0, -2], [0, 0, 1, 1], [0, 0, 0, 1.]])


def _expected_delta(start, end, loop, angle, scale):
    origin = np.linalg.inv(_matrix(0., angle))

    def trajectory(time):
        if not loop:
            return origin @ _matrix(min(2., max(0., time)), angle)
        cycles = math.floor(time / 2.)
        return np.linalg.matrix_power(origin @ _matrix(2., angle), cycles) @ origin @ _matrix(time % 2., angle)

    result = np.linalg.inv(trajectory(start)) @ trajectory(end)
    result[:3, 3] *= scale
    return result


def _world_pose(transform):
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    result = np.eye(4)
    result[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    p = transform.position
    result[:3, 3] = (p.x, p.y, p.z)
    return result


def _owner(scene, model_guid, parented):
    owner = scene.create_game_object("Moving actor")
    if parented:
        parent = scene.create_game_object("Rotated scaled parent")
        parent.transform.euler_angles = Vector3(20., 35., 15.)
        parent.transform.local_scale = Vector3(2., .75, 1.5)
        owner.set_parent(parent, False)
        owner.transform.local_euler_angles = Vector3(5., 10., 25.)
    owner.transform.position = Vector3(7., -3., 4.)
    renderer = owner.add_component("SkinnedMeshRenderer")
    renderer.set_source_model_guid(model_guid)
    animator = owner.add_py_component(SkeletalAnimator())
    animator.awake()
    return owner, animator


@pytest.mark.parametrize("parented", [False, True])
@pytest.mark.parametrize("start,end,loop", [
    (0., 2., False), (1.5, 2.5, True), (0., 5., True), (.5, 6.5, True),
    (5., -1., True), (-3., 3., True), (2., 0., False), (-1., 3., False),
])
def test_imported_motion_matches_rigid_trajectory_across_partitions(motion_model, scene, start, end, loop, parented):
    _, _, model_guid, _, clip, angle, scale = motion_model
    expected_delta = _expected_delta(start, end, loop, angle, scale)
    for steps in (1, 2, 20, 120):
        owner, animator = _owner(scene, model_guid, parented)
        animator._current_clip = clip
        expected = _world_pose(owner.transform) @ expected_delta
        for index in range(steps):
            animator._apply_imported_root_motion(
                clip, start + (end-start)*index/steps, start + (end-start)*(index+1)/steps, loop,
            )
        np.testing.assert_allclose(_world_pose(owner.transform), expected, atol=3e-4, rtol=0., err_msg=f"steps={steps}")


@pytest.mark.parametrize("duration,loop", [(2., False), (2.5, True), (8., True)])
def test_real_fsm_update_keeps_root_motion_partition_independent(motion_model, scene, duration, loop):
    database, source, model_guid, clip_guid, _, angle, scale = motion_model
    path = source.with_name("Motion.animfsm")
    controller = AnimStateMachine(mode="3d", default_state="Move", states=[
        AnimState(name="Move", clip_guid=clip_guid, loop=loop),
    ])
    assert controller.save(str(path))
    imported = AssetManager.import_asset(str(path), database=database)
    assert imported.succeeded, imported.error
    for steps in (1, 2, 20, 120):
        owner, animator = _owner(scene, model_guid, True)
        animator.controller = AnimStateMachineRef(guid=imported.guid)
        animator.start()
        assert animator.current_state == "Move" and animator.is_playing
        expected = _world_pose(owner.transform) @ _expected_delta(0., duration, loop, angle, scale)
        for _ in range(steps):
            animator.update(duration / steps)
        np.testing.assert_allclose(_world_pose(owner.transform), expected, atol=3e-4, rtol=0., err_msg=f"steps={steps}")
