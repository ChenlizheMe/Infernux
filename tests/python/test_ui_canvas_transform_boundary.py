"""Screen Canvas is a layout boundary, not an inherited world pose."""

import pytest

from infernux.lib import Vector3
from infernux.ui import UICanvas, UIButton, UIFrame, RenderMode
from infernux.ui.inx_ui_screen_component import clear_rect_cache
from infernux.ui.ui_render_dispatch import runtime_ui_revision


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
    if not button.is_world_space():
        button.set_rect(20, 30, 100, 40, 800, 600)
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


def test_canvas_anchors_follow_viewport_and_live_rect_edits_roundtrip(scene):
    from infernux.ui.enums import ScreenAlignH, ScreenAlignV

    owner, _ = _canvas(scene)
    for horizontal, dx in ((ScreenAlignH.Left, 0), (ScreenAlignH.Center, 100), (ScreenAlignH.Right, 200)):
        for vertical, dy in ((ScreenAlignV.Top, 0), (ScreenAlignV.Center, 150), (ScreenAlignV.Bottom, 300)):
            button = _button(scene, owner)
            button.align_h, button.align_v = horizontal, vertical
            assert button.get_rect(800, 600) == pytest.approx((20, 30, 100, 40))
            assert button.get_rect(1000, 900) == pytest.approx((20 + dx, 30 + dy, 100, 40))
            button.set_rect(55, 66, 100, 40, 1000, 900)
            assert button.get_rect(1000, 900) == pytest.approx((55, 66, 100, 40))


def test_canvas_stops_ancestor_layout_and_clipping(scene):
    outer_owner, outer_canvas = _canvas(scene)
    frame_owner = scene.create_game_object("Outer Frame")
    frame_owner.set_parent(outer_owner)
    frame = UIFrame()
    frame_owner.add_py_component(frame)
    frame.width, frame.height = 10, 10
    frame.set_rect(500, 300, 10, 10, 800, 600)
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


def test_host_rect_edit_uses_canvas_layout_and_one_replayable_transaction(scene, monkeypatch):
    from types import SimpleNamespace

    from infernux.engine.interaction import ComponentCommandService
    from infernux.engine.bootstrap import EditorBootstrap
    from infernux.engine.undo import UndoManager
    from infernux.host import EditorAutomationHost, MainThreadCommandQueue, OperationError, OperationRegistry
    from infernux.host.scene_operations import build_scene_operations
    from infernux.ui import UIImage, UILayoutDirection, UILayoutPosition, UILayoutSizing

    canvas_owner, canvas = _canvas(scene)
    frame_owner = scene.create_game_object("Flow Frame")
    frame_owner.set_parent(canvas_owner)
    frame = frame_owner.add_component(UIFrame)
    frame.layout_direction = UILayoutDirection.Horizontal
    frame.clip_content = True
    frame.set_rect(80, 60, 320, 180, 800, 600)
    owner = scene.create_game_object("Flow Image")
    owner.set_parent(frame_owner)
    image = owner.add_component(UIImage)
    image.width, image.height = 80, 40
    image.width_sizing = UILayoutSizing.Fill
    owner.transform.local_position = Vector3(0, 0, 7)
    owner.transform.local_euler_angles = Vector3(0, 0, 90)
    initial_position = tuple(owner.transform.local_position)
    initial_rect = image.get_rect(800, 600)

    monkeypatch.setattr(UndoManager, "_instance", None)
    manager = UndoManager()
    monkeypatch.setattr(ComponentCommandService, "_instance", None)
    service = ComponentCommandService()
    host = EditorAutomationHost()
    monkeypatch.setattr(EditorAutomationHost, "_provider", host)
    monkeypatch.setattr(host, "interaction_core", lambda: SimpleNamespace(components=service))
    monkeypatch.setattr(EditorBootstrap, "_instance", None)
    monkeypatch.setattr(MainThreadCommandQueue, "_instance", None)
    MainThreadCommandQueue.instance().drain()
    registry = OperationRegistry()
    for operation in build_scene_operations():
        registry.register(operation)

    def execute(name, **arguments):
        return registry.execute("infernux.scene." + name, arguments, capabilities=("scene.*",))

    ids = {"object_id": owner.id, "component_id": image.component_id}
    before = execute("ui.layout.get", **ids)
    assert before["coordinates"]["basis"] == "design_reference"
    assert before["canvas"]["logical_size"] == [800, 600]
    assert before["rect"] == pytest.approx(initial_rect)
    assert before["flow"]["object_id"] == frame_owner.id
    assert execute("component.schema", **ids)["ui_layout"] == before
    tree = execute("hierarchy.get", include_ui_layout=True)
    assert tree["objects"][0]["children"][0]["children"][0]["ui_layout"] == [before]

    with pytest.raises(OperationError, match="auto layout"):
        execute("ui.rect.set", **ids, rect=[125, 93, 120, 30])
    assert tuple(owner.transform.local_position) == initial_position
    assert not manager.action_journal.applied_entries()
    edited = execute("ui.rect.set", **ids, rect=[125, 93, 120, 30], detach_from_layout=True)
    assert edited["rect"] == pytest.approx((125, 93, 120, 30))
    assert edited["visual_rect"][2:] == pytest.approx((30, 120))
    assert edited["clip_bounds"] == pytest.approx((80, 60, 400, 240))
    assert edited["flow"] is None
    assert edited["local_position"][2] == 7
    assert edited["rotation_degrees"] == 90
    assert image.layout_position == UILayoutPosition.Absolute
    assert len(manager.action_journal.applied_entries()) == 1
    manager.undo()
    assert tuple(owner.transform.local_position) == initial_position
    assert image.width_sizing == UILayoutSizing.Fill
    assert image.layout_position == UILayoutPosition.Flow
    assert image.get_rect(800, 600) == pytest.approx(initial_rect)
    manager.redo()
    assert image.get_rect(800, 600) == pytest.approx((125, 93, 120, 30))
    execute("ui.rect.set", **ids, rect=[125, 93, 120, 30])
    assert len(manager.action_journal.applied_entries()) == 1

    # A live wide viewport has a different logical extent from design pixels.
    view = SimpleNamespace(render_target_size=(1600, 600))
    monkeypatch.setattr(EditorBootstrap, "_instance", SimpleNamespace(
        window_manager=SimpleNamespace(get_window_instance=lambda _id: view),
    ))
    live = execute("ui.layout.get", **ids)
    assert live["coordinates"]["basis"] == "game_render_target"
    assert live["canvas"]["reference_size"] == [800, 600]
    assert live["canvas"]["logical_size"] == pytest.approx(canvas.compute_logical_size(1600, 600))
    execute("ui.rect.set", **ids, rect=[340, 120, 120, 30])
    assert image.get_rect(*live["canvas"]["logical_size"]) == pytest.approx((340, 120, 120, 30))
    canvas.pixel_perfect = True
    live = execute("ui.layout.get", **ids)
    assert live["render_scale"] == pytest.approx(canvas.compute_scale(1600, 600)[:2])
    assert live["render_rect"] == list(canvas.pixel_adjust_rect(live["rect"], *live["render_scale"]))
    assert all(value == round(value) for value in live["render_rect"])
    assert live["rect"] == pytest.approx((340, 120, 120, 30))
    assert live["rotation_degrees"] == 90

    # Detaching a child changes its Hug parent's width before position resolves.
    monkeypatch.setattr(EditorBootstrap, "_instance", None)
    image.layout_position = UILayoutPosition.Flow
    image.width_sizing = UILayoutSizing.Fixed
    frame.width_sizing = UILayoutSizing.Hug
    assert frame.get_rect(800, 600)[2] == 120
    execute("ui.rect.set", **ids, rect=[50, 70, 120, 30], detach_from_layout=True)
    assert frame.get_rect(800, 600)[2] == 0
    assert image.get_rect(800, 600) == pytest.approx((50, 70, 120, 30))
    manager.undo()
    assert image.layout_position == UILayoutPosition.Flow
    assert frame.get_rect(800, 600)[2] == 120

    # Moving a Hug frame does not turn its derived dimensions into Fixed.
    frame_ids = {"object_id": frame_owner.id, "component_id": frame.component_id}
    execute("ui.rect.set", **frame_ids, rect=[300, 200, 120, 180])
    assert frame.width_sizing == UILayoutSizing.Hug
    assert frame.get_rect(800, 600) == pytest.approx((300, 200, 120, 180))
    world_image = scene.create_game_object("World Image").add_component(UIImage)
    world_ids = {"object_id": world_image.game_object.id, "component_id": world_image.component_id}
    assert execute("ui.layout.get", **world_ids)["coordinates"]["basis"] == "world_geometry"
    with pytest.raises(OperationError, match="world Transform"):
        execute("ui.rect.set", **world_ids, rect=[0, 0, 100, 40])
    manager.clear()
