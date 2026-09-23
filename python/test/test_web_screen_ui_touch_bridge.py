"""Browser touch contacts must reach the shared screen UI event processor."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import Infernux.engine.runtime_screen_ui as runtime_screen_ui
import Infernux.engine.ui.runtime_canvas_snapshot as canvas_snapshot
import Infernux.input as input_module
from Infernux.input import TouchPhase
from Infernux.ui.ui_event_data import PointerType


BOOTSTRAP = (
    Path(__file__).resolve().parents[2]
    / "external/plugins/infernux_web/native/bootstrap.py"
)


def test_web_screen_ui_forwards_same_frame_touch_and_mouse(monkeypatch):
    source = ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    function = next(
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name == "_process_screen_ui_events"
    )
    bootstrap = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(BOOTSTRAP), "exec"), bootstrap)

    canvas = SimpleNamespace(
        reference_width=1280,
        reference_height=720,
        set_input_logical_size=lambda *_: None,
    )
    scene = object()
    bootstrap["_player_scene_manager"] = SimpleNamespace(
        get_active_scene=lambda: scene,
        get_runtime_persistent_scene=lambda: None,
    )
    bootstrap["_screen_width"] = 1280
    bootstrap["_screen_height"] = 720
    captured = []
    bootstrap["_screen_ui_event_processor"] = SimpleNamespace(
        process_pointers=lambda *args: captured.append(args)
    )
    monkeypatch.setattr(
        canvas_snapshot,
        "collect_sorted_runtime_canvas_snapshot",
        lambda active, persistent: (canvas,),
    )
    monkeypatch.setattr(
        runtime_screen_ui, "_canvas_metrics", lambda *_: (2.0, 2.0, 0.0, 640, 360)
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
            get_game_mouse_frame_state=lambda _button: (200, 100, 0, 0, False, False, False),
            touches=(touch,),
        ),
    )

    bootstrap["_process_screen_ui_events"](0.016)

    canvases, pointers, dt = captured.pop()
    assert canvases == [canvas]
    assert dt == 0.016
    assert len(pointers) == 2
    assert pointers[0].pointer_type is PointerType.Mouse
    assert pointers[0].canvas_positions == ((100.0, 50.0),)
    assert pointers[1].pointer_type is PointerType.Touch
    assert pointers[1].pointer_id == 5
    assert pointers[1].canvas_positions == ((320.0, 90.0),)
    assert pointers[1].press_canvas_positions == ((160.0, 180.0),)
    assert pointers[1].down and pointers[1].up and not pointers[1].canceled
