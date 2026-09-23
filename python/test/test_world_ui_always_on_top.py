import math
from types import SimpleNamespace

from Infernux.engine.runtime_screen_ui import (
    RuntimeScreenUISubmission,
    WorldUIElementTarget,
    collect_runtime_ui_input_surfaces,
    map_runtime_ui_pointer,
    map_runtime_ui_pointers,
    pick_world_ui_object_ids,
)
from Infernux.lib import ScreenUIList, Vector3
from Infernux.physics import Physics
from Infernux.ui import UIButton


def _control(scene, name, depth, top=False):
    obj = scene.create_game_object(name)
    obj.transform.position = Vector3(0, 0, depth)
    element = UIButton()
    obj.add_py_component(element)
    element.width = 200.0
    element.height = 100.0
    element.world_always_on_top = top
    return obj, element


def _camera():
    return SimpleNamespace(
        culling_mask=0xffffffff,
        screen_point_to_ray=lambda *_: (Vector3(0, 0, 5), Vector3(0, 0, -1)),
    )


def test_top_policy_overrides_world_depth_order_without_changing_default(scene, monkeypatch):
    near, ordinary = _control(scene, "ordinary", 1.0)
    far, top = _control(scene, "top", 0.0, top=True)
    surfaces = collect_runtime_ui_input_surfaces(scene)
    priorities = [surface.input_priority((100, 50, 5 - surface.element.game_object.transform.position.z), i)
                  for i, surface in enumerate(surfaces)]
    assert priorities[1] > priorities[0]
    assert pick_world_ui_object_ids(scene, (0, 0, 5), (0, 0, -1))[:2] == (far.id, near.id)

    monkeypatch.setattr(Physics, "raycast", lambda *_args, **_kwargs: SimpleNamespace(distance=2.0))
    mapped = map_runtime_ui_pointer(surfaces, _camera(), 0, 0, 1920, 1080)
    assert math.isnan(mapped[0][0])
    assert mapped[1] == (100.0, 50.0, 5.0)

    top.world_always_on_top = False
    assert surfaces[1].input_priority(mapped[1], 1) < surfaces[0].input_priority((100, 50, 4), 0)
    mapped = map_runtime_ui_pointer(surfaces, _camera(), 0, 0, 1920, 1080)
    assert all(math.isnan(position[0]) for position in mapped)


def test_top_policy_survives_batched_touch_occlusion(scene, monkeypatch):
    _control(scene, "ordinary", 1.0)
    _control(scene, "top", 0.0, top=True)
    surfaces = collect_runtime_ui_input_surfaces(scene)

    def raycast_batch(origins, _directions, output, **_kwargs):
        count = origins.shape[0]
        output["hit"][:count] = 1
        output["distance"][:count] = 2.0
        return output

    monkeypatch.setattr(Physics, "raycast_batch", raycast_batch)
    mapped = map_runtime_ui_pointers(surfaces, _camera(), ((0, 0), (10, 10)), 1920, 1080)
    assert len(mapped) == 2
    for positions in mapped:
        assert math.isnan(positions[0][0])
        assert positions[1] == (100.0, 50.0, 5.0)


def test_top_policy_is_serialized_and_defaults_to_scene_depth():
    element = UIButton()
    assert element.world_always_on_top is False
    element.world_always_on_top = True
    document = element._serialize_fields_document()
    restored = UIButton()
    restored._deserialize_fields_document(document)
    assert restored.world_always_on_top is True


def test_world_submission_publishes_explicit_depth_policy(scene, monkeypatch):
    import Infernux.engine.runtime_screen_ui as runtime_ui

    _, element = _control(scene, "submitted", 0.0)
    calls = []
    renderer = SimpleNamespace(
        begin_world_object=lambda *args: calls.append(args),
        end_world_element=lambda: None,
    )
    monkeypatch.setattr(runtime_ui, "_ui_dispatch", lambda *_args, **_kwargs: None)
    RuntimeScreenUISubmission._submit_world_element(element, renderer, lambda _: 0, ScreenUIList)
    assert calls[-1][3] is False
    element.world_always_on_top = True
    RuntimeScreenUISubmission._submit_world_element(element, renderer, lambda _: 0, ScreenUIList)
    assert calls[-1][3] is True
