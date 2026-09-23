"""World input and Scene picking share live native Transform projection."""
import math
from types import SimpleNamespace

import pytest

from Infernux.lib import Vector3
from Infernux.ui import UIText, UIButton, UISlider, UICanvas, UIGroup, TextResizeMode
from Infernux.engine.runtime_screen_ui import (
    WorldUIElementTarget, collect_runtime_ui_input_surfaces, map_runtime_ui_pointer,
    map_runtime_ui_pointers, map_world_ui_ray, pick_world_ui_object_ids,
)
from Infernux.ui.ui_event_system import UIEventProcessor, UIPointerFrame
from Infernux.ui.ui_event_data import PointerType


def control(scene, kind=UIButton, position=(0, 0, 0)):
    obj = scene.create_game_object(kind.__name__)
    obj.transform.position = Vector3(*position)
    element = kind()
    obj.add_py_component(element)
    element.width, element.height = 200., 100.
    element.raycast_target = True
    return obj, element


def camera(origin=(0, 0, 5), direction=(0, 0, -1), mask=0xffffffff):
    return SimpleNamespace(culling_mask=mask, screen_point_to_ray=lambda *_: (
        Vector3(*origin), Vector3(*direction)))


def map_pointer(surfaces, view=None):
    return map_runtime_ui_pointer(surfaces, view or camera(), 0, 0, 1920, 1080)


@pytest.fixture(autouse=True)
def no_occlusion(monkeypatch):
    from Infernux.physics import Physics
    monkeypatch.setattr(Physics, 'raycast', lambda *_a, **_kw: None)


def test_batch_reads_native_pose_not_python_component_properties(scene, monkeypatch):
    obj, element = control(scene)
    targets = collect_runtime_ui_input_surfaces(scene)
    assert map_pointer(targets)[0] == pytest.approx((100, 50, 5))
    def unexpected(*_):
        raise AssertionError('Warm projection crossed a Python component owner/layout getter')
    monkeypatch.setattr(WorldUIElementTarget, 'game_object', property(unexpected))
    monkeypatch.setattr(UIButton, 'get_resolved_size', unexpected)
    for x in (0.2, 0.4, 0.6):
        obj.transform.position = Vector3(x, 0, 0)
        assert map_pointer(targets)[0] == pytest.approx((100 - 100*x, 50, 5))
    obj.layer = 31
    assert math.isnan(map_pointer(targets, camera(mask=1))[0][0])
    assert map_pointer(targets, camera(mask=1 << 31))[0] == pytest.approx((40, 50, 5))


@pytest.mark.parametrize('world_ui', [False, True])
def test_scene_mouse_hit_without_world_plane_intersection(scene, monkeypatch, world_ui):
    from Infernux.physics import Physics
    surfaces = ()
    if world_ui:
        control(scene)
        surfaces = collect_runtime_ui_input_surfaces(scene)
    hit = SimpleNamespace(game_object=object())
    queries = []
    def query(*args, **kwargs):
        queries.append(kwargs)
        return hit
    monkeypatch.setattr(Physics, 'raycast', query)
    # Parallel to the UI plane; ordinary scene input must still query once.
    positions, actual = map_runtime_ui_pointer(
        surfaces, camera(direction=(1, 0, 0), mask=5), 20, 30, 1920, 1080,
        include_scene_hit=True,
    )
    assert actual is hit
    assert queries == [dict(max_distance=1000.0, layer_mask=1, query_triggers=True)]
    assert len(positions) == int(world_ui)


def test_no_scene_query_when_input_disabled_or_camera_absent(monkeypatch):
    from Infernux.physics import Physics
    def unexpected(*args, **kwargs):
        pytest.fail('Disabled scene input must not issue a physics query')
    monkeypatch.setattr(Physics, 'raycast', unexpected)
    assert map_runtime_ui_pointer((), camera(), 0, 0, 1920, 1080) == ()
    assert map_runtime_ui_pointer((), None, 0, 0, 1920, 1080, include_scene_hit=True) == ((), None)


def test_blocking_screen_ui_skips_the_hidden_scene_query(scene, monkeypatch):
    from Infernux.physics import Physics

    canvas_object = scene.create_game_object('Screen Canvas')
    canvas = UICanvas()
    canvas_object.add_py_component(canvas)
    _, button = control(scene)
    button.game_object.set_parent(canvas_object)
    button.set_rect(0.0, 0.0, button.width, button.height, 1920.0, 1080.0)
    surfaces = collect_runtime_ui_input_surfaces(scene)

    def unexpected(*_args, **_kwargs):
        pytest.fail('Blocking screen UI must not query hidden 3D geometry')

    monkeypatch.setattr(Physics, 'raycast', unexpected)
    monkeypatch.setattr(Physics, 'raycast_all', unexpected)
    positions, hit = map_runtime_ui_pointer(
        surfaces, camera(mask=5), 10, 10, 1920, 1080,
        include_scene_hit=True,
    )
    assert len(positions) == 1
    assert hit is None


def test_nonblocking_screen_ui_preserves_the_scene_query(scene, monkeypatch):
    from Infernux.physics import Physics

    canvas_object = scene.create_game_object('Screen Canvas')
    canvas = UICanvas()
    canvas_object.add_py_component(canvas)
    _, button = control(scene)
    button.game_object.set_parent(canvas_object)
    button.set_rect(0.0, 0.0, button.width, button.height, 1920.0, 1080.0)
    button.raycast_target = False
    surfaces = collect_runtime_ui_input_surfaces(scene)
    expected = SimpleNamespace(game_object=object())
    calls = []

    def query(*_args, **kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(Physics, 'raycast', query)
    positions, hit = map_runtime_ui_pointer(
        surfaces, camera(mask=5), 10, 10, 1920, 1080,
        include_scene_hit=True,
    )
    assert len(positions) == 1
    assert hit is expected
    assert calls == [dict(max_distance=1000.0, layer_mask=1, query_triggers=True)]


def test_blocking_screen_ui_skips_world_ui_occlusion_and_scene_query(scene, monkeypatch):
    from Infernux.physics import Physics

    control(scene, position=(0, 0, 0))
    canvas_object = scene.create_game_object('Screen Canvas')
    canvas = UICanvas()
    canvas_object.add_py_component(canvas)
    _, button = control(scene)
    button.game_object.set_parent(canvas_object)
    button.set_rect(0.0, 0.0, button.width, button.height, 1920.0, 1080.0)
    surfaces = collect_runtime_ui_input_surfaces(scene)

    def unexpected(*_args, **_kwargs):
        pytest.fail('Front-most screen UI must skip all hidden physics queries')

    monkeypatch.setattr(Physics, 'raycast', unexpected)
    monkeypatch.setattr(Physics, 'raycast_all', unexpected)
    projection_calls = []
    view = camera(mask=5)
    view.screen_point_to_ray = lambda *_args: projection_calls.append(1) or pytest.fail(
        'Blocking screen UI must not project hidden world UI')
    positions, hit = map_runtime_ui_pointer(
        surfaces, view, 10, 10, 1920, 1080,
        include_scene_hit=True,
    )
    assert len(positions) == 2
    assert hit is None
    assert projection_calls == []


def test_world_ui_touch_contacts_share_one_physics_query_batch(scene, monkeypatch):
    from Infernux.physics import Physics

    control(scene, position=(0, 0, 0))
    surfaces = collect_runtime_ui_input_surfaces(scene)
    calls = []

    def batch(origins, directions, output, **kwargs):
        calls.append((origins.copy(), directions.copy(), kwargs))
        count = origins.shape[0]
        output['hit'][:count] = 1
        output['distance'][:count] = 1.0
        return output

    monkeypatch.setattr(Physics, 'raycast_batch', batch)
    monkeypatch.setattr(Physics, 'raycast', lambda *_args, **_kwargs: pytest.fail(
        'Batched touch mapping must not issue per-pointer raycasts'))
    mapped = map_runtime_ui_pointers(
        surfaces,
        camera(),
        tuple((index, index) for index in range(8)),
        1920,
        1080,
    )
    assert len(mapped) == 8
    assert all(math.isnan(positions[0][0]) for positions in mapped)
    assert len(calls) == 1
    origins, directions, kwargs = calls[0]
    assert origins.shape == directions.shape == (8, 3)
    assert kwargs == dict(max_distance=5.0, layer_mask=0xffffffff, query_triggers=False)


def test_touch_batch_keeps_each_pointers_own_max_distance(scene, monkeypatch):
    from Infernux.physics import Physics

    _, element = control(scene, position=(0, 0, 0))
    element.width = element.height = 2000.0
    surfaces = collect_runtime_ui_input_surfaces(scene)
    view = SimpleNamespace(
        culling_mask=0xffffffff,
        screen_point_to_ray=lambda screen_x, *_args: (
            Vector3(0, 0, 5 if screen_x == 0 else 10),
            Vector3(0, 0, -1),
        ),
    )

    def batch(origins, _directions, output, **_kwargs):
        assert origins[:, 2].tolist() == [5.0, 10.0]
        output['hit'][:2] = 1
        # The same closest Collider is behind pointer 0's 5 m UI plane but
        # in front of pointer 1's 10 m plane. The shared 10 m batch distance
        # must not leak into pointer 0's shorter query contract.
        output['distance'][:2] = 6.0
        return output

    monkeypatch.setattr(Physics, 'raycast_batch', batch)
    mapped = map_runtime_ui_pointers(
        surfaces, view, ((0, 0), (1, 0)), 1920, 1080,
    )
    assert mapped[0][0][0] == pytest.approx(1000.0)
    assert math.isnan(mapped[1][0][0])


def test_screen_ui_blocks_the_whole_touch_query_batch(scene, monkeypatch):
    from Infernux.physics import Physics

    control(scene, position=(0, 0, 0))
    canvas_object = scene.create_game_object('Screen Canvas')
    canvas = UICanvas()
    canvas_object.add_py_component(canvas)
    _, button = control(scene)
    button.game_object.set_parent(canvas_object)
    button.set_rect(0.0, 0.0, button.width, button.height, 1920.0, 1080.0)
    surfaces = collect_runtime_ui_input_surfaces(scene)
    monkeypatch.setattr(Physics, 'raycast_batch', lambda *_args, **_kwargs: pytest.fail(
        'Blocking screen UI must skip the hidden touch batch'))
    mapped = map_runtime_ui_pointers(
        surfaces, camera(), ((10, 10), (20, 20)), 1920, 1080,
    )
    assert len(mapped) == 2
    assert all(len(positions) == 2 for positions in mapped)


def test_disabled_screen_ui_does_not_block_or_recompute_touch_mapping(scene, monkeypatch):
    from Infernux.physics import Physics

    control(scene, position=(0, 0, 0))
    canvas_object = scene.create_game_object('Disabled Screen Canvas')
    canvas = UICanvas()
    canvas_object.add_py_component(canvas)
    _, button = control(scene)
    button.game_object.set_parent(canvas_object)
    button.set_rect(0.0, 0.0, button.width, button.height, 1920.0, 1080.0)
    canvas.enabled = False
    surfaces = collect_runtime_ui_input_surfaces(scene)
    calls = []

    def batch(origins, _directions, output, **_kwargs):
        calls.append(origins.shape[0])
        output['hit'][:origins.shape[0]] = 0
        return output

    monkeypatch.setattr(Physics, 'raycast_batch', batch)
    mapped = map_runtime_ui_pointers(
        surfaces, camera(), ((10, 10), (20, 20)), 1920, 1080,
    )
    assert calls == [2]
    assert all(math.isnan(positions[1][0]) for positions in mapped)


def test_ignore_raycast_solid_occludes_world_ui_but_is_not_mouse_target(scene, monkeypatch):
    from Infernux.physics import Physics
    control(scene)
    surfaces = collect_runtime_ui_input_surfaces(scene)
    blocker = SimpleNamespace(distance=1., game_object=SimpleNamespace(layer=2),
                              collider=SimpleNamespace(is_trigger=False))
    target = SimpleNamespace(distance=2., game_object=SimpleNamespace(layer=0),
                             collider=SimpleNamespace(is_trigger=True))
    calls = []
    def query(*args, **kwargs):
        calls.append(kwargs)
        return [blocker, target]
    monkeypatch.setattr(Physics, 'raycast_all', query)
    positions, hit = map_runtime_ui_pointer(surfaces, camera(mask=5), 0, 0, 1920, 1080,
                                           include_scene_hit=True)
    assert hit is target
    assert math.isnan(positions[0][0])
    assert calls == [dict(max_distance=1000.0, layer_mask=5, query_triggers=True)]


def test_parent_rotation_translation_and_scale_follow_render_pose(scene):
    import random
    rng = random.Random(41)
    parent = scene.create_game_object('Rotating parent')
    obj, element = control(scene)
    obj.set_parent(parent)
    obj.transform.local_position = Vector3(1, 2, 3)
    targets = collect_runtime_ui_input_surfaces(scene)
    for _ in range(30):
        parent.transform.position = Vector3(*(rng.uniform(-5, 5) for _ in range(3)))
        parent.transform.euler_angles = Vector3(*(rng.uniform(-170, 170) for _ in range(3)))
        parent.transform.local_scale = Vector3(2, .5, 3)
        transform = obj.transform
        center, right, up, normal = transform.position, transform.right, transform.up, transform.forward
        origin = tuple(getattr(center, a) + .3*getattr(right, a) - .2*getattr(up, a)
                       + 5*getattr(normal, a) for a in 'xyz')
        direction = tuple(-getattr(normal, a) for a in 'xyz')
        assert map_pointer(targets, camera(origin, direction))[0] == pytest.approx((130, 70, 5), abs=2e-3)
        assert pick_world_ui_object_ids(scene, origin, direction) == (obj.id,)


def test_parallel_behind_camera_and_unbounded_drag_coordinates(scene):
    _, element = control(scene)
    target = WorldUIElementTarget(element)
    assert map_world_ui_ray(target, (0, 0, 5), (1, 0, 0)) is None
    assert map_world_ui_ray(target, (0, 0, 5), (0, 0, 1)) is None
    assert map_world_ui_ray(target, (10, 20, 5), (0, 0, -1)) == pytest.approx((1100, -1950, 5))


def test_measured_text_and_authored_size_refresh_cached_projection(scene):
    _, text = control(scene, UIText)
    targets = collect_runtime_ui_input_surfaces(scene)
    assert map_pointer(targets)[0][:2] == pytest.approx((100, 50))
    text.width = 400
    assert map_pointer(targets)[0][:2] == pytest.approx((200, 50))
    text.resize_mode = TextResizeMode.AutoWidth
    text.resolve_text_layout(lambda *_: (600., 80.))
    width, height = text.get_resolved_size()
    assert map_pointer(targets)[0][:2] == pytest.approx((width*.5, height*.5))


def test_world_pointer_nearest_hit_screen_priority_and_live_policy(scene):
    far_obj, far = control(scene, position=(0, 0, 0))
    near_obj, near = control(scene, position=(0, 0, 2))
    group = UIGroup()
    near_obj.add_py_component(group)
    processor = UIEventProcessor()
    def frame():
        surfaces = collect_runtime_ui_input_surfaces(scene)
        processor.process(surfaces, map_pointer(surfaces), False, False, False, (0, 0), .01)
        return processor._pointers[(PointerType.Mouse, -1)].hover_target
    assert frame() is near
    group.blocks_raycast = False
    assert frame() is far
    group.blocks_raycast = True
    near_obj.active = False
    assert frame() is far
    near_obj.active = True
    assert frame() is near
    canvas_obj = scene.create_game_object('Screen Canvas')
    canvas = UICanvas()
    canvas_obj.add_py_component(canvas)
    _, screen = control(scene)
    screen.game_object.set_parent(canvas_obj)
    screen.set_rect(0.0, 0.0, screen.width, screen.height, 1920.0, 1080.0)
    assert frame() is screen
    canvas.enabled = False
    assert frame() is near


def test_overlapping_world_quads_only_check_policy_until_front_hit(scene, monkeypatch):
    elements = [control(scene, position=(0, 0, index*.01))[1] for index in range(100)]
    surfaces = collect_runtime_ui_input_surfaces(scene)
    calls = []
    original = UIButton.effectively_blocks_raycast
    def policy(element):
        calls.append(element)
        return original(element)
    monkeypatch.setattr(UIButton, 'effectively_blocks_raycast', policy)
    processor = UIEventProcessor()
    processor.process(surfaces, map_pointer(surfaces), False, False, False, (0, 0), .01)
    assert processor._pointers[(PointerType.Mouse, -1)].hover_target is elements[-1]
    assert calls == [elements[-1]]
    # Equal depths retain the first surface's original tie-break rule.
    elements[-2].game_object.transform.position = elements[-1].game_object.transform.position
    processor.process(surfaces, map_pointer(surfaces), False, False, False, (0, 0), .01)
    assert processor._pointers[(PointerType.Mouse, -1)].hover_target is elements[-2]


@pytest.mark.parametrize('pointer_type', [PointerType.Mouse, PointerType.Touch])
def test_captured_slider_keeps_coordinates_beyond_quad(scene, pointer_type):
    _, slider = control(scene, UISlider)
    surfaces = collect_runtime_ui_input_surfaces(scene)
    processor = UIEventProcessor()
    def frame(origin_x, **state):
        processor.process_pointers(surfaces, (UIPointerFrame(
            pointer_id=7, pointer_type=pointer_type,
            canvas_positions=map_pointer(surfaces, camera((origin_x, 0, 5))), **state),), .01)
    frame(0, down=True, held=True)
    assert slider.value == pytest.approx(.5)
    frame(3, held=True)
    assert slider.value == pytest.approx(1)
    frame(-3, held=True)
    assert slider.value == pytest.approx(0)
    frame(-3, up=True)
    assert processor._pointers.get((pointer_type, 7), SimpleNamespace(press_target=None)).press_target is None


def test_destroyed_members_are_replaced_and_scene_pick_ignores_input_policy(scene):
    obj, element = control(scene, UIText)
    before = collect_runtime_ui_input_surfaces(scene)
    element.raycast_target = False
    assert pick_world_ui_object_ids(scene, (0, 0, 5), (0, 0, -1)) == (obj.id,)
    scene.destroy_game_object(obj)
    scene.process_pending_destroys()
    new_obj, new_element = control(scene, position=(0, 0, 1))
    after = collect_runtime_ui_input_surfaces(scene)
    assert after is collect_runtime_ui_input_surfaces(scene)
    assert after != before and after[0].element is new_element
    assert map_pointer(after)[0] == pytest.approx((100, 50, 4))
    assert pick_world_ui_object_ids(scene, (0, 0, 5), (0, 0, -1)) == (new_obj.id,)
