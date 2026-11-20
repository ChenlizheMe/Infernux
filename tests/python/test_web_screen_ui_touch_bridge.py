"""Browser touch contacts must reach the shared screen UI event processor."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import infernux.engine.runtime_screen_ui as runtime_screen_ui
import infernux.engine.ui.runtime_canvas_snapshot as canvas_snapshot
import infernux.input as input_module
from infernux.input import TouchPhase
from infernux.ui.enums import RenderMode
from infernux.ui.ui_event_data import PointerButton, PointerType


BOOTSTRAP = (
    Path(__file__).resolve().parents[2]
    / "external/plugins/infernux_web/native/bootstrap.py"
)


def test_web_screen_ui_forwards_same_frame_touch_and_mouse_buttons(monkeypatch):
    source = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    function = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_process_screen_ui_events"
    )
    bootstrap = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(BOOTSTRAP), "exec"), bootstrap)

    canvas = SimpleNamespace(
        render_mode=RenderMode.ScreenOverlay,
        reference_width=1280,
        reference_height=720,
        set_input_logical_size=lambda *_: None,
        raycast=lambda *_: None,
    )
    # The event processor queries scene roots while collecting world-space
    # canvases.  Use the smallest scene contract rather than an opaque object
    # so this fixture exercises the same path as the runtime.
    scene = SimpleNamespace(
        get_root_objects=lambda: (),
        effective_game_camera=None,
        world_id=17,
        temporal_discontinuity_revision=3,
    )
    bootstrap["_player_scene_manager"] = SimpleNamespace(
        get_active_scene=lambda: scene,
        get_runtime_persistent_scene=lambda: None,
        scene_count=1,
        get_scene_at=lambda index: scene if index == 0 else None,
    )
    bootstrap["_screen_width"] = 1280
    bootstrap["_screen_height"] = 720
    captured = []
    bootstrap["_screen_ui_event_processor"] = SimpleNamespace(
        process_pointers=lambda *args: captured.append(args),
        reset=lambda: None,
        discard=lambda: None,
    )
    scene_pointer_frames = []
    bootstrap["_mouse_event_dispatcher"] = SimpleNamespace(
        needs_scene_hit=lambda: True,
        process=lambda *args, **kwargs: scene_pointer_frames.append((args, kwargs)),
        discard=lambda: None,
    )
    bootstrap["_input_scene_token"] = None
    monkeypatch.setattr(
        canvas_snapshot,
        "collect_sorted_runtime_canvas_snapshot",
        lambda *scenes: (canvas,),
    )
    monkeypatch.setattr(
        runtime_screen_ui, "_canvas_metrics", lambda *_: (2.0, 2.0, 0.0, 640, 360)
    )
    monkeypatch.setattr(
        runtime_screen_ui,
        "collect_runtime_ui_input_surfaces",
        lambda *scenes: (canvas,),
    )

    touch = SimpleNamespace(
        finger_id=5,
        phase=TouchPhase.ENDED,
        normalized_position=(0.5, 0.75),
        began_this_frame=True,
        begin_normalized_position=(0.25, 0.5),
    )
    monkeypatch.setattr(
        input_module,
        "Input",
        SimpleNamespace(
            get_game_mouse_frame_state=lambda button: (200, 100, 0, 0, button == 1, button == 1, button == 2),
            touches=(touch,),
        ),
    )

    bootstrap["_process_screen_ui_events"](0.016)

    canvases, pointers, dt = captured.pop()
    assert canvases == (canvas,)
    assert dt == 0.016
    assert len(pointers) == 4
    assert pointers[0].pointer_type is PointerType.Mouse
    assert pointers[0].canvas_positions == ((100.0, 50.0),)
    assert [p.button for p in pointers[:3]] == [PointerButton.Left, PointerButton.Right, PointerButton.Middle]
    assert pointers[1].down and pointers[1].held and pointers[2].up
    assert pointers[3].pointer_type is PointerType.Touch
    assert pointers[3].pointer_id == 5
    assert pointers[3].canvas_positions == ((320.0, 90.0),)
    assert pointers[3].press_canvas_positions == ((160.0, 180.0),)
    assert pointers[3].down and pointers[3].up and not pointers[3].canceled
    assert len(scene_pointer_frames) == 1
    scene_args, scene_kwargs = scene_pointer_frames[0]
    assert scene_args[1:] == ((200, 100), (1280.0, 720.0))
    assert scene_kwargs["button_state"] == (False, False, False)
