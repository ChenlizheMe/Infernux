"""Screen Canvas is a layout boundary, not an inherited world pose."""

import pytest

from Infernux.lib import Vector3
from Infernux.ui import UICanvas, UIButton, UIFrame, RenderMode
from Infernux.ui.inx_ui_screen_component import clear_rect_cache
from Infernux.ui.ui_render_dispatch import runtime_ui_revision


def _canvas(scene, parent=None):
    owner = scene.create_game_object("Screen Canvas")
    if parent is not None:
        owner.set_parent(parent)
    canvas = UICanvas()
    canvas.reference_width, canvas.reference_height = 800, 600
    owner.add_py_component(canvas)
    return owner, canvas


def _button(scene, parent):
    owner = scene.create_game_object("Button")
    owner.set_parent(parent)
    button = UIButton()
    owner.add_py_component(button)
    button.width, button.height = 100, 40
    button.x, button.y = 20, 30
    return button


@pytest.mark.parametrize("mode", [RenderMode.ScreenOverlay, RenderMode.CameraOverlay])
def test_screen_canvas_pose_does_not_move_geometry_or_hit_target(scene, mode):
    ancestor = scene.create_game_object("Scene parent")
    owner, canvas = _canvas(scene, ancestor)
    canvas.render_mode = mode
    button = _button(scene, owner)
    before = button.get_rotated_corners(800, 600)
    revision = runtime_ui_revision(scene, (canvas,), 800, 600)
    assert canvas.raycast(30, 40) is button
    for obj in (owner, ancestor):
        obj.transform.local_position = Vector3(800, -400, 75)
        obj.transform.local_euler_angles = Vector3(30, 45, 90)
        obj.transform.local_scale = Vector3(3, 2, 4)
        clear_rect_cache(object())
        assert button.get_rotated_corners(800, 600) == before
        assert canvas.raycast(30, 40) is button
        assert runtime_ui_revision(scene, (canvas,), 800, 600) == revision
    initial = button.game_object.transform.local_position
    button.game_object.transform.local_position = Vector3(initial.x + 200, initial.y, initial.z)
    assert canvas.raycast(30, 40) is None
    assert canvas.raycast(230, 40) is button


def test_canvas_stops_ancestor_layout_and_clipping(scene):
    outer_owner, outer_canvas = _canvas(scene)
    frame_owner = scene.create_game_object("Outer Frame")
    frame_owner.set_parent(outer_owner)
    frame = UIFrame()
    frame_owner.add_py_component(frame)
    frame.width, frame.height = 10, 10
    frame.x, frame.y = 500, 300
    frame.clip_content = True
    inner_owner, inner_canvas = _canvas(scene, frame_owner)
    button = _button(scene, inner_owner)
    assert button._get_parent_ui_component() is None
    assert button.get_rect(800, 600) == pytest.approx((20, 30, 100, 40))
    assert button.get_effective_clip_rect(800, 600) is None
    assert inner_canvas.raycast(30, 40) is button
    assert button not in tuple(outer_canvas.iter_ui_elements())


def test_world_ui_still_inherits_parent_pose(scene):
    parent = scene.create_game_object("World parent")
    button = _button(scene, parent)
    assert button.is_world_space()
    before = button.world_ui_matrix()
    parent.transform.local_position = Vector3(4, 5, 6)
    parent.transform.local_euler_angles = Vector3(0, 0, 45)
    assert button.world_ui_matrix() != before
