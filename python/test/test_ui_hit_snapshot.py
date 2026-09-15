"""Cached broad-phase geometry must preserve live UI input semantics."""
import pytest

from Infernux.ui import UICanvas, UIButton, UIFrame, UIGroup, UIText, TextResizeMode
from Infernux.lib import Vector3


def screen(scene):
    root = scene.create_game_object('Canvas')
    canvas = UICanvas()
    root.add_py_component(canvas)
    return root, canvas


def control(scene, parent, kind=UIButton, x=0., y=0., width=100., height=40.):
    obj = scene.create_game_object(kind.__name__)
    obj.set_parent(parent)
    item = kind()
    item.width, item.height = width, height
    obj.add_py_component(item)
    item.x, item.y = x, y
    return item


def test_static_queries_only_read_exact_geometry_of_candidates(scene, monkeypatch):
    root, canvas = screen(scene)
    button = control(scene, root)
    assert canvas.raycast(10, 10) is button
    def unexpected(*_args, **_kwargs):
        raise AssertionError('Unchanged broad-phase bounds were rebuilt')
    monkeypatch.setattr(UIButton, 'get_visual_rect', unexpected)
    monkeypatch.setattr(UIButton, 'effectively_blocks_raycast', unexpected)
    for _ in range(30):
        assert canvas.raycast(-100, -100) is None
        assert canvas.raycast(10, 10) is button


def test_parent_motion_and_native_enable_rebuild_hit_geometry(scene):
    root, canvas = screen(scene)
    frame = control(scene, root, UIFrame, width=400, height=200)
    frame.raycast_target = False
    child = control(scene, frame.game_object)
    assert canvas.raycast(10, 10) is child
    parent_transform = frame.game_object.transform
    initial = parent_transform.local_position
    for offset in (200, 400, 600):
        parent_transform.local_position = Vector3(initial.x + offset, initial.y, initial.z)
        assert canvas.raycast(10, 10) is None
        assert canvas.raycast(offset + 10, 10) is child
    child._cpp_component.enabled = False
    assert canvas.raycast(610, 10) is None
    child._cpp_component.enabled = True
    assert canvas.raycast(610, 10) is child


def test_color_animation_does_not_rebuild_input_bounds(scene):
    root, canvas = screen(scene)
    button = control(scene, root)
    group = UIGroup()
    root.add_py_component(group)
    assert canvas.raycast(10, 10) is button
    snapshot = canvas._hit_candidates_cache
    for index in range(20):
        button.color = [index / 20., .5, .3, 1.]
        group.alpha = index / 20.
        assert canvas.raycast(10, 10) is button
        assert canvas._hit_candidates_cache is snapshot
    button.raycast_target = False
    assert canvas.raycast(10, 10) is None
    button.raycast_target = True
    assert canvas.raycast(10, 10) is button


def test_group_policy_clip_rotation_tolerance_and_draw_order(scene):
    root, canvas = screen(scene)
    underneath = control(scene, root, width=300, height=200)
    frame = control(scene, root, UIFrame, width=100, height=100)
    frame.raycast_target = False
    frame.clip_content = True
    top = control(scene, frame.game_object, x=80, y=0, width=100, height=40)
    assert canvas.raycast_all(90, 10) == [top, underneath]
    assert canvas.raycast(140, 10) is underneath
    frame.clip_content = False
    assert canvas.raycast(140, 10) is top
    group = UIGroup()
    frame.game_object.add_py_component(group)
    group.blocks_raycast = False
    assert canvas.raycast(90, 10) is underneath
    group.blocks_raycast = True
    assert canvas.raycast(90, 10) is top
    top.rotation = 90
    # Rotated 100x40 rectangle around (130,20) has x=[110,150], y=[-30,70].
    assert canvas.raycast(130, 60) is top
    assert canvas.raycast(105, 20) is underneath
    assert canvas.raycast(105, 20, tolerance=6) is top


def test_resolved_text_size_changes_invalidate_input_snapshot(scene):
    root, canvas = screen(scene)
    text = control(scene, root, UIText, width=20, height=40)
    text.raycast_target = True
    text.resize_mode = TextResizeMode.AutoWidth
    assert canvas.raycast(70, 10) is None
    text.resolve_text_layout(lambda *_: (140., 30.))
    # AutoWidth retains the Transform center, so the expanded box reaches 70.
    assert canvas.raycast(70, 10) is text
    text.resize_mode = TextResizeMode.FixedSize
    assert canvas.raycast(70, 10) is None
    text.resize_mode = TextResizeMode.AutoWidth
    assert canvas.raycast(70, 10) is text


def test_stationary_mouse_refreshes_hover_after_motion_and_disable(scene):
    from Infernux.ui.ui_event_system import UIEventProcessor
    root, canvas = screen(scene)
    button = control(scene, root)
    processor = UIEventProcessor()
    def tick():
        processor.process([canvas], [(10., 10.)], False, False, False, (0., 0.), .016)
    tick()
    pointer = next(iter(processor._pointers.values()))
    assert pointer.hover_target is button
    button.x = 200
    tick()
    assert pointer.hover_target is None
    button.x = 0
    tick()
    assert pointer.hover_target is button
    button.enabled = False
    tick()
    assert pointer.hover_target is None


def test_world_rejects_outside_bounds_before_policy_traversal(scene, monkeypatch):
    from Infernux.engine.runtime_screen_ui import WorldUIElementTarget
    item = control(scene, scene.create_game_object('World root'))
    def unexpected():
        raise AssertionError('Miss queried ancestor policy')
    monkeypatch.setattr(item, 'effectively_blocks_raycast', unexpected)
    target = WorldUIElementTarget(item)
    assert target.raycast(-10, -10) is None


def test_destroy_and_new_member_does_not_reuse_hit_records(scene):
    root, canvas = screen(scene)
    first = control(scene, root)
    assert canvas.raycast(10, 10) is first
    scene.destroy_game_object(first.game_object)
    scene.process_pending_destroys()
    second = control(scene, root)
    assert canvas.raycast(10, 10) is second


def test_render_dependencies_refresh_child_layout_without_an_input_query(scene):
    from Infernux.ui.ui_render_dispatch import runtime_ui_revision
    root, canvas = screen(scene)
    parent = control(scene, root, UIFrame, width=400, height=200)
    child = control(scene, parent.game_object)
    runtime_ui_revision(scene, (canvas,), 1920, 1080)
    assert child.get_rect(1920, 1080)[0] == pytest.approx(0.)
    transform = parent.game_object.transform
    start = transform.local_position
    transform.local_position = Vector3(start.x + 200, start.y, start.z)
    runtime_ui_revision(scene, (canvas,), 1920, 1080)
    assert child.get_rect(1920, 1080)[0] == pytest.approx(200.)
