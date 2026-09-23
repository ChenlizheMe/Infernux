from types import SimpleNamespace

import numpy as np
import pytest

from Infernux.engine.runtime_screen_ui import (
    RuntimeScreenUISubmission,
    collect_runtime_ui_input_surfaces,
    map_runtime_ui_pointer,
    map_runtime_ui_pointers,
    map_world_ui_ray,
    pick_world_ui_object_ids,
)
from Infernux.lib import ScreenUIList, Vector3
from Infernux.ui import UIButton


def _camera():
    # Right-handed 90-degree perspective with a 128-pixel viewport.
    projection = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, -1.010101, -0.1010101], [0, 0, -1, 0],
    ], dtype=np.float32)
    return SimpleNamespace(
        culling_mask=0xffffffff,
        view_matrix=np.eye(4, dtype=np.float32),
        projection_matrix=projection,
        screen_point_to_ray=lambda *_: (Vector3(.1, 0, 0), Vector3(0, 0, -1)),
    )


def _element(scene):
    obj = scene.create_game_object("Camera-policy UI")
    obj.transform.position = Vector3(0, 0, -2)
    obj.transform.local_euler_angles = Vector3(0, 90, 0)
    element = UIButton()
    obj.add_py_component(element)
    element.width = element.height = 20
    return obj, element


def test_camera_policies_are_explicit_serialized_and_submitted(scene, monkeypatch):
    import Infernux.engine.runtime_screen_ui as runtime_ui

    _, element = _element(scene)
    assert element.world_billboard is False
    assert element.world_constant_screen_size is False
    element.world_billboard = element.world_constant_screen_size = True
    restored = UIButton()
    restored._deserialize_fields_document(element._serialize_fields_document())
    assert restored.world_billboard and restored.world_constant_screen_size
    calls = []
    renderer = SimpleNamespace(begin_world_object=lambda *args: calls.append(args),
                               end_world_element=lambda: None)
    monkeypatch.setattr(runtime_ui, "_ui_dispatch", lambda *_args, **_kwargs: None)
    RuntimeScreenUISubmission._submit_world_element(element, renderer, lambda _: 0, ScreenUIList)
    assert calls[-1][3:] == (False, True, True)


def test_billboard_and_constant_screen_size_share_input_projection(scene, monkeypatch):
    _, element = _element(scene)
    surfaces = collect_runtime_ui_input_surfaces(scene)
    camera = _camera()
    ray_origin, ray_direction = camera.screen_point_to_ray(0, 0, 128, 128)
    assert map_world_ui_ray(surfaces[0], ray_origin, ray_direction) is None
    element.world_billboard = True
    billboard = map_world_ui_ray(surfaces[0], ray_origin, ray_direction,
                                 camera=camera, viewport_height=128)
    assert billboard == pytest.approx((20., 10., 2.))
    element.world_constant_screen_size = True
    fixed = map_world_ui_ray(surfaces[0], ray_origin, ray_direction,
                             camera=camera, viewport_height=128)
    assert fixed == pytest.approx((13.2, 10., 2.), abs=1e-4)
    assert pick_world_ui_object_ids(scene, ray_origin, ray_direction,
                                    camera=camera, viewport_height=128)
    from Infernux.physics import Physics
    monkeypatch.setattr(Physics, "raycast", lambda *_args, **_kwargs: None)
    assert map_runtime_ui_pointer(surfaces, camera, 0, 0, 128, 128)[0] == pytest.approx(fixed)
    def no_occluders(origins, _directions, output, **_kwargs):
        output["hit"][:len(origins)] = 0
        return output
    monkeypatch.setattr(Physics, "raycast_batch", no_occluders)
    batched = map_runtime_ui_pointers(surfaces, camera, ((0, 0), (1, 0)), 128, 128)
    assert all(item[0] == pytest.approx(fixed) for item in batched)
    camera.projection_matrix = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, -1 / 9.9, -.1 / 9.9], [0, 0, 0, 1],
    ], dtype=np.float32)
    orthographic = map_world_ui_ray(surfaces[0], ray_origin, ray_direction,
                                     camera=camera, viewport_height=128)
    assert orthographic == pytest.approx((16.4, 10., 2.), abs=1e-4)
