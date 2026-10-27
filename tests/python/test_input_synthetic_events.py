from __future__ import annotations

import threading
import time

import pytest

from infernux.input import Input
from infernux.lib import (
    InputManager,
    InxGUIRenderable,
    RenderPipelineCallback,
    get_gui_semantic_snapshot,
    request_gui_semantic_snapshot,
    set_gui_semantic_capture_enabled,
)


@pytest.fixture(autouse=True)
def _gui_only_pipeline(engine):
    # These tests render real editor controls, without authoring a Scene graph.
    # Install that explicit pipeline instead of running with a missing pipeline
    # and emitting a renderer configuration error on every frame.
    class GuiOnlyPipeline(RenderPipelineCallback):
        def render(self, context, camera):
            pass

    engine.set_render_pipeline(GuiOnlyPipeline())
    try:
        yield
    finally:
        engine.set_render_pipeline(None)


@pytest.fixture(autouse=True)
def _reset_semantic_capture_between_tests():
    # Native semantic capture is process-global. Keep this module from
    # leaking a continuous-capture session into later MCP input tests.
    set_gui_semantic_capture_enabled(False)
    yield
    set_gui_semantic_capture_enabled(False)


def test_synthetic_events_follow_the_graphical_input_path(engine):
    """Automation input must reach InputManager through the graphical loop."""
    manager = InputManager.instance()
    manager.set_editor_game_viewport((500, 500, 20, 20), (500, 500, 20, 20))
    Input.set_game_focused(False)
    observed: dict[str, object] = {}
    frame = [0]
    first_sequence = int(engine.last_processed_synthetic_input_sequence) + 1

    def on_update(_delta_time: float) -> None:
        frame[0] += 1
        if frame[0] == 1:
            observed["down"] = {
                "key": manager.get_key(26),
                "game_key": Input.get_key(26),
                "game_focused": Input.is_game_focused(),
                "key_down": manager.get_key_down(26),
                "mouse": manager.get_mouse_button(0),
                "mouse_down": manager.get_mouse_button_down(0),
                "game_mouse": Input.get_mouse_button(0),
                "game_mouse_down": Input.get_mouse_button_down(0),
                "game_mouse_frame": Input.get_mouse_frame_state(0)[4:],
                "mouse_position": (manager.mouse_position_x, manager.mouse_position_y),
                "mouse_delta": Input.get_mouse_delta(),
                "mouse_delta_again": Input.get_mouse_delta(),
                "mouse_axes": (Input.get_axis("Mouse X"), Input.get_axis("Mouse Y")),
                "text": manager.input_string,
            }
            observed["release_sequences"] = (
                engine.queue_synthetic_key_input(26, False),
                engine.queue_synthetic_mouse_button_input(0, False, 37.0, 41.0),
            )
        else:
            observed["up"] = {
                "key": manager.get_key(26),
                "game_key": Input.get_key(26),
                "game_focused": Input.is_game_focused(),
                "key_up": manager.get_key_up(26),
                "mouse": manager.get_mouse_button(0),
                "mouse_up": manager.get_mouse_button_up(0),
                "game_mouse_up": Input.get_mouse_button_up(0),
                "game_mouse_frame": Input.get_mouse_frame_state(0)[4:],
                "last_sequence": engine.last_processed_synthetic_input_sequence,
                "pending": engine.pending_synthetic_input_count,
                "mouse_delta": Input.get_mouse_delta(),
            }
            engine.exit()

    try:
        engine.set_pre_scene_update_callback(on_update)
        sequences = (
            engine.queue_synthetic_key_input(26, True),
            engine.queue_synthetic_mouse_motion_input(37.0, 41.0, 5.0, -2.0),
            engine.queue_synthetic_mouse_button_input(0, True, 37.0, 41.0),
            engine.queue_synthetic_text_input("agent-input"),
        )
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        Input.set_game_focused(True)

    assert sequences == (first_sequence, first_sequence + 1, first_sequence + 2, first_sequence + 3)
    assert observed["down"] == {
        "key": True,
        "game_key": True,
        "game_focused": False,
        "key_down": True,
        "mouse": True,
        "mouse_down": True,
        "game_mouse": True,
        "game_mouse_down": True,
        "game_mouse_frame": (True, True, False),
        "mouse_position": (37.0, 41.0),
        "mouse_delta": (5.0, -2.0),
        "mouse_delta_again": (5.0, -2.0),
        "mouse_axes": (0.5, -0.2),
        "text": "agent-input",
    }
    assert observed["release_sequences"] == (first_sequence + 4, first_sequence + 5)
    assert observed["up"] == {
        "key": False,
        "game_key": False,
        "game_focused": False,
        "key_up": True,
        "mouse": False,
        "mouse_up": True,
        "game_mouse_up": True,
        "game_mouse_frame": (False, False, True),
        "last_sequence": first_sequence + 5,
        "pending": 0,
        "mouse_delta": (0.0, 0.0),
    }


@pytest.mark.parametrize("button,occluded", [(index, False) for index in range(5)] + [(0, True)])
def test_game_capture_is_resolved_before_component_update(engine, scene, button, occluded, editor_project):
    from infernux.components import InxComponent
    from infernux.engine.ui.game_view_panel import GameViewPanel
    from infernux.lib import SceneManager

    manager = InputManager.instance()
    state = {"frame": 0}
    down_samples = []

    class GameCaptureView(GameViewPanel):
        def __init__(self):
            super().__init__()
            self.point = None

        def _is_playing(self):
            return SceneManager.instance().is_playing()

        def on_render(self, ctx):
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            ctx.set_next_window_size(64.0, 64.0, 0)
            if ctx.begin_window("Game Capture###game_capture", True, 0):
                ctx.invisible_button("##GameViewportInput", 32.0, 24.0)
                rect = tuple(ctx.get_item_visible_rect())
                manager.set_editor_game_viewport(rect, rect)
                self.point = (rect[0] + rect[2] * 0.5, rect[1] + rect[3] * 0.5)
                self._route_game_input(
                    ctx, 64, 64,
                    any(ctx.is_item_clicked(index) for index in range(5)),
                )
            ctx.end_window()
            if occluded:
                ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
                ctx.set_next_window_size(64.0, 64.0, 0)
                ctx.set_next_window_focus()
                if ctx.begin_window("Floating Panel###capture_blocker", True, 0):
                    ctx.button("Covered")
                ctx.end_window()

    class ClickConsumer(InxComponent):
        def start(self):
            Input.set_cursor_locked(True)

        def update(self, _delta_time):
            if manager.get_mouse_button_down(button):
                down_samples.append((Input.get_mouse_button_down(button), Input.is_cursor_locked()))

    view = GameCaptureView()
    scene.create_game_object("Click consumer").add_component(ClickConsumer)
    engine.register_gui_renderable("test.game_capture", view)
    set_gui_semantic_capture_enabled(True)

    def on_update(_delta_time):
        state["frame"] += 1
        if state["frame"] == 3:
            engine.queue_synthetic_mouse_motion_input(*view.point, 0.0, 0.0)
        elif state["frame"] == 5:
            engine.queue_synthetic_mouse_button_input(button, True, *view.point)
        elif state["frame"] == 7:
            engine.queue_synthetic_mouse_button_input(button, False, *view.point)
        elif state["frame"] == 10:
            engine.exit()

    scenes = SceneManager.instance()
    try:
        scenes.play()
        manager.set_editor_cursor_capture_allowed(False)
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.game_capture")
        scenes.stop()
        Input.set_cursor_locked(False)
        manager.set_editor_cursor_capture_allowed(True)
        Input.set_game_focused(True)

    assert down_samples == [(True, not occluded)]


@pytest.mark.parametrize("button", [1, 2], ids=["look", "pan"])
def test_scene_camera_keeps_captured_motion_on_gui_frames(engine, button):
    from infernux.engine.ui.scene_view_panel import SceneViewPanel

    manager = InputManager.instance()
    panel = SceneViewPanel(engine=engine)
    camera = engine.editor_camera
    initial_position = tuple(camera.position)
    initial_rotation = tuple(camera.rotation)
    initial_focus = tuple(camera.focus_point)
    initial_distance = float(camera.focus_distance)
    state = {"frame": 0, "gui_motion": False}

    class CameraInputView(InxGUIRenderable):
        def on_render(self, ctx):
            if manager.is_editor_mouse_capture_active:
                state["gui_motion"] |= Input.get_mouse_delta() == (20.0, 5.0)
                panel._process_camera_input(ctx, 1.0 / 60.0)

    view = CameraInputView()

    def on_update(_delta_time):
        state["frame"] += 1
        if state["frame"] == 1:
            engine.queue_synthetic_mouse_button_input(button, True, 32.0, 32.0)
        elif state["frame"] == 3:
            manager.set_editor_mouse_capture(True)
            panel._camera_capture_active = True
            panel._was_right_down = button == 1
            panel._was_middle_down = button == 2
            engine.queue_synthetic_mouse_motion_input(32.0, 32.0, 20.0, 5.0)
        elif state["frame"] == 5:
            state["position"] = tuple(camera.position)
            state["rotation"] = tuple(camera.rotation)
            engine.queue_synthetic_mouse_button_input(button, False, 32.0, 32.0)
        elif state["frame"] == 7:
            engine.exit()

    engine.register_gui_renderable("test.scene_camera_input", view)
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.scene_camera_input")
        manager.set_editor_mouse_capture(False)
        panel._force_camera_input_release()
        camera.restore_state(
            *initial_position, *initial_focus, initial_distance, *initial_rotation,
        )

    assert state["gui_motion"], "Motion must share a frame with the editor GUI"
    if button == 1:
        assert state["rotation"] != pytest.approx(initial_rotation)
        assert state["position"] == pytest.approx(initial_position)
    else:
        assert state["position"] != pytest.approx(initial_position)
        assert state["rotation"] == pytest.approx(initial_rotation)


@pytest.mark.parametrize("playing,focus", [(True, "Game"), (True, "Editor"), (False, "Game")])
def test_game_keyboard_does_not_navigate_editor_controls(engine, playing, focus):
    from infernux.engine.ui.imgui_keys import KEY_ENTER, KEY_F2, KEY_LEFT_CTRL, KEY_S, KEY_TAB

    manager = InputManager.instance()
    keys = {43: KEY_TAB, 59: KEY_F2, 224: KEY_LEFT_CTRL, 22: KEY_S, 40: KEY_ENTER}
    observed = {"gui": set(), "game": set(), "text": "", "frame": 0, "renders": 0}
    sent_at = None
    deadline = time.monotonic() + 5.0

    class KeyboardView(InxGUIRenderable):
        def on_render(self, ctx):
            for name in ("Game", "Editor"):
                ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
                ctx.set_next_window_size(64.0, 64.0, 0)
                if name == focus:
                    ctx.set_next_window_focus()
                if ctx.begin_window(name + "###keyboard_" + name, True, 1 << 12):
                    if name == "Game":
                        manager.set_editor_game_viewport((0, 0, 64, 64), (0, 0, 64, 64))
                    ctx.button(name + " control")
                ctx.end_window()
            observed["gui"].update(key for key in keys if ctx.is_key_pressed(keys[key]))
            observed["renders"] += 1

    view = KeyboardView()

    def on_update(_delta_time):
        nonlocal sent_at
        observed["frame"] += 1
        if sent_at is None and observed["renders"] >= 3:
            sent_at = observed["frame"]
            for key in keys:
                engine.queue_synthetic_key_input(key, True)
            engine.queue_synthetic_text_input("game-text")
        elif sent_at is not None and observed["frame"] == sent_at + 2:
            for key in keys:
                engine.queue_synthetic_key_input(key, False)
        elif observed["renders"] >= 7 or time.monotonic() >= deadline:
            engine.exit()
        observed["game"].update(key for key in keys if manager.get_game_key_down(key))
        observed["text"] += manager.input_string

    engine.register_gui_renderable("test.keyboard_ownership", view)
    try:
        manager.set_editor_cursor_capture_allowed(True)
        engine.set_play_mode_rendering(playing)
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_play_mode_rendering(False)
        engine.unregister_gui_renderable("test.keyboard_ownership")
        manager.set_editor_game_viewport((0, 0, 0, 0), (0, 0, 0, 0))
        manager.reset_all()

    assert observed["game"] == set(keys)
    assert observed["text"] == "game-text"
    assert observed["gui"] == (set() if playing and focus == "Game" else set(keys))


def test_semantic_snapshot_request_captures_exactly_one_rendered_frame(engine):
    class _SemanticProbe(InxGUIRenderable):
        def on_render(self, ctx) -> None:
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            ctx.set_next_window_size(64.0, 64.0, 0)
            if ctx.begin_window("One Shot Probe###one_shot_probe", True, 0):
                ctx.button("Probe##one_shot")
                ctx.record_semantic_item("button", "Probe", True, "test.semantic.one_shot")
            ctx.end_window()

    probe = _SemanticProbe()
    state = {"update": 0, "first_frame": 0, "last_frame": 0}
    set_gui_semantic_capture_enabled(False)
    request_gui_semantic_snapshot()

    def on_update(_delta_time: float) -> None:
        state["update"] += 1
        snapshot = get_gui_semantic_snapshot()
        if state["update"] == 3:
            state["first_frame"] = int(snapshot.get("frame", 0) or 0)
        elif state["update"] == 7:
            state["last_frame"] = int(snapshot.get("frame", 0) or 0)
            engine.exit()

    engine.register_gui_renderable("test.semantic_one_shot", probe)
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.semantic_one_shot")
        set_gui_semantic_capture_enabled(False)

    assert state["first_frame"] > 0
    assert state["last_frame"] == state["first_frame"]


@pytest.mark.parametrize('close_timing', ['none', 'before_batch', 'within_batch'])
def test_synthetic_pointer_click_invokes_python_button_callback(engine, close_timing):
    """A replayed release must retain the synthetic pointer position for Python UI callbacks."""

    class _ButtonProbe(InxGUIRenderable):
        def __init__(self) -> None:
            super().__init__()
            self.clicks = 0

        def _on_click(self) -> None:
            self.clicks += 1

        def on_render(self, ctx) -> None:
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            ctx.set_next_window_size(64.0, 64.0, 0)
            if ctx.begin_window("Synthetic Button Probe###synthetic_button_probe", True, 0):
                ctx.button("Activate##synthetic_callback", self._on_click)
                ctx.record_semantic_item("button", "Activate", True, "test.synthetic.button.callback")
            ctx.end_window()

    probe = _ButtonProbe()
    state = {"frame": 0, "queued": False, "target": None}
    set_gui_semantic_capture_enabled(True)

    def _find_target():
        snapshot = get_gui_semantic_snapshot()
        for target in snapshot.get("targets", []):
            if target.get("semantic_id") == "test.synthetic.button.callback":
                return target
        return None

    def on_update(_delta_time: float) -> None:
        state["frame"] += 1
        if close_timing == 'before_batch' and 'close_sequence' not in state:
            state['close_sequence'] = engine.queue_synthetic_close_request()
            return
        target = _find_target()
        if not state["queued"] and target:
            if close_timing == 'before_batch':
                assert engine.is_close_requested()
            rect = target["rect"]
            x = float(rect[0]) + float(rect[2]) * 0.5
            y = float(rect[1]) + float(rect[3]) * 0.5
            if close_timing == 'within_batch':
                state['close_sequence'] = engine.queue_synthetic_close_request()
            engine.queue_synthetic_mouse_motion_input(x, y, 0.0, 0.0)
            engine.queue_synthetic_mouse_button_input(0, True, x, y)
            state['release_sequence'] = engine.queue_synthetic_mouse_button_input(0, False, x, y)
            state["queued"] = True
            state["target"] = target
        elif probe.clicks:
            engine.exit()
        elif state["frame"] >= 32:
            engine.exit()

    engine.register_gui_renderable("test.synthetic_button_callback", probe)
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.synthetic_button_callback")
        engine.cancel_close()

    assert state["target"] is not None
    assert probe.clicks == 1
    assert engine.last_processed_synthetic_input_sequence == state['release_sequence']
    assert engine.pending_synthetic_input_count == 0


def test_synthetic_pointer_click_invokes_callback_when_events_span_frames(engine):
    """MCP waits for each input transition, so the click must survive three frames."""

    class _ButtonProbe(InxGUIRenderable):
        def __init__(self) -> None:
            super().__init__()
            self.clicks = 0

        def on_render(self, ctx) -> None:
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            ctx.set_next_window_size(64.0, 64.0, 0)
            if ctx.begin_window("Synthetic Split-Frame Button Probe###synthetic_split_frame_button_probe", True, 0):
                if ctx.button("Activate##synthetic_split_frame_callback"):
                    self.clicks += 1
                ctx.record_semantic_item("button", "Activate", True, "test.synthetic.button.split_frame_callback")
            ctx.end_window()

    probe = _ButtonProbe()
    state = {
        "frame": 0,
        "phase": "find",
        "target": None,
        "semantic_frames": [],
        "semantic_input_sequences": [],
        "semantic_mouse_positions": [],
        "queued_sequences": [],
    }
    set_gui_semantic_capture_enabled(True)

    def _find_target():
        snapshot = get_gui_semantic_snapshot()
        for target in snapshot.get("targets", []):
            if target.get("semantic_id") == "test.synthetic.button.split_frame_callback":
                return target
        return None

    def _record_semantic_snapshot() -> None:
        snapshot = get_gui_semantic_snapshot()
        state["semantic_frames"].append(int(snapshot.get("frame", 0) or 0))
        state["semantic_input_sequences"].append(int(snapshot.get("input_sequence", 0) or 0))
        state["semantic_mouse_positions"].append(tuple(snapshot.get("mouse", ())))

    def on_update(_delta_time: float) -> None:
        state["frame"] += 1
        target = _find_target()
        if state["phase"] == "find" and target:
            rect = target["rect"]
            x = float(rect[0]) + float(rect[2]) * 0.5
            y = float(rect[1]) + float(rect[3]) * 0.5
            state["target"] = target
            state["point"] = (x, y)
            set_gui_semantic_capture_enabled(False)
            state["queued_sequences"].append(engine.queue_synthetic_mouse_motion_input(x, y, 0.0, 0.0))
            state["phase"] = "moved"
        elif state["phase"] == "moved":
            x, y = state["point"]
            state["queued_sequences"].append(engine.queue_synthetic_mouse_button_input(0, True, x, y))
            state["phase"] = "pressed"
        elif state["phase"] == "pressed":
            _record_semantic_snapshot()
            x, y = state["point"]
            state["queued_sequences"].append(engine.queue_synthetic_mouse_button_input(0, False, x, y))
            state["phase"] = "released"
        elif state["phase"] == "released":
            _record_semantic_snapshot()
            state["phase"] = "await_click"
        elif probe.clicks:
            _record_semantic_snapshot()
            engine.exit()
        elif state["frame"] >= 48:
            engine.exit()

    engine.register_gui_renderable("test.synthetic_split_frame_button_callback", probe)
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.synthetic_split_frame_button_callback")

    assert state["target"] is not None
    assert probe.clicks == 1
    assert len(state["semantic_frames"]) == 3
    assert state["semantic_frames"][0] > 0
    assert state["semantic_frames"] == sorted(set(state["semantic_frames"]))
    assert state["semantic_input_sequences"] == state["queued_sequences"]
    assert len(state["semantic_mouse_positions"]) == 3
    expected_x, expected_y = state["point"]
    assert all(
        len(position) == 2
        and abs(float(position[0]) - expected_x) < 1.0
        and abs(float(position[1]) - expected_y) < 1.0
        for position in state["semantic_mouse_positions"]
    ), (state["point"], state["semantic_mouse_positions"])


def test_combo_popup_exposes_clickable_options_and_preserves_trigger_alias(engine):
    """Regular combos must expose their popup through the same semantic input path as a user."""

    class _ComboProbe(InxGUIRenderable):
        def __init__(self) -> None:
            super().__init__()
            self.value = 0

        def on_render(self, ctx) -> None:
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            ctx.set_next_window_size(64.0, 64.0, 0)
            if ctx.begin_window("Synthetic Combo Probe###synthetic_combo_probe", True, 0):
                self.value = ctx.combo("##mode", self.value, ["2D", "3D", "Timeline"], 3)
                ctx.record_semantic_item("combo", "Mode", True, "test.synthetic.combo.mode")
            ctx.end_window()

    probe = _ComboProbe()
    state = {"frame": 0, "phase": "find_trigger", "trigger": None, "option": None, "alias_preserved": False}
    set_gui_semantic_capture_enabled(True)

    def _targets():
        return get_gui_semantic_snapshot().get("targets", [])

    def _find(semantic_id: str):
        return next((target for target in _targets() if target.get("semantic_id") == semantic_id), None)

    def _point(target) -> tuple[float, float]:
        return float(target["click_point"][0]), float(target["click_point"][1])

    def on_update(_delta_time: float) -> None:
        state["frame"] += 1
        if state["phase"] == "find_trigger":
            trigger = _find("test.synthetic.combo.mode")
            if trigger and trigger.get("has_click_point"):
                state["trigger"] = trigger
                state["point"] = _point(trigger)
                engine.queue_synthetic_mouse_motion_input(*state["point"], 0.0, 0.0)
                state["phase"] = "trigger_moved"
        elif state["phase"] == "trigger_moved":
            engine.queue_synthetic_mouse_button_input(0, True, *state["point"])
            state["phase"] = "trigger_pressed"
        elif state["phase"] == "trigger_pressed":
            engine.queue_synthetic_mouse_button_input(0, False, *state["point"])
            state["phase"] = "find_option"
        elif state["phase"] == "find_option":
            alias = _find("test.synthetic.combo.mode")
            option = _find("mode:option:2")
            if alias and option and option.get("has_click_point"):
                state["alias_preserved"] = alias.get("kind") == "combo" and alias.get("label") == "Mode"
                state["option"] = option
                state["point"] = _point(option)
                engine.queue_synthetic_mouse_motion_input(*state["point"], 0.0, 0.0)
                state["phase"] = "option_moved"
        elif state["phase"] == "option_moved":
            engine.queue_synthetic_mouse_button_input(0, True, *state["point"])
            state["phase"] = "option_pressed"
        elif state["phase"] == "option_pressed":
            engine.queue_synthetic_mouse_button_input(0, False, *state["point"])
            state["phase"] = "selected"
        elif state["phase"] == "selected" and probe.value == 2:
            engine.exit()
        elif state["frame"] >= 48:
            engine.exit()

    engine.register_gui_renderable("test.synthetic_combo", probe)
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.synthetic_combo")

    assert state["trigger"] is not None
    assert state["option"] is not None
    assert state["alias_preserved"] is True
    assert probe.value == 2


def test_synthetic_close_request_follows_graphical_close_path(engine):
    """Remote close requests must enter the same intercepted Editor path."""

    observed: dict[str, object] = {}
    frame = [0]

    def on_update(_delta_time: float) -> None:
        frame[0] += 1
        if frame[0] == 1:
            observed["sequence"] = engine.queue_synthetic_close_request()
        elif engine.is_close_requested():
            observed["close_requested"] = True
            engine.cancel_close()
            engine.exit()
        elif frame[0] >= 16:
            engine.exit()

    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)

    assert int(observed["sequence"]) > 0
    assert observed["close_requested"] is True


def test_floating_window_is_recovered_and_close_button_is_semantic(engine):
    """Off-viewport floating panels must recover with their native close control reachable."""

    class _ClosableProbe(InxGUIRenderable):
        def __init__(self) -> None:
            super().__init__()
            self.open = True
            self.viewport = (0.0, 0.0, 0.0, 0.0)
            self.initial_layout_submitted = False

        def on_render(self, ctx) -> None:
            self.viewport = tuple(float(v) for v in ctx.get_main_viewport_bounds())
            vx, vy, vw, vh = self.viewport
            if not self.initial_layout_submitted:
                # Seed an oversized, off-viewport floating layout. The next
                # frame must recover it before the window is published.
                ctx.set_next_window_pos(vx + vw - 8.0, vy + vh - 8.0, 1, 0.0, 0.0)
                ctx.set_next_window_size(vw + 32.0, vh + 32.0, 1)
                self.initial_layout_submitted = True
            visible, self.open = ctx.begin_window_closable(
                "Synthetic Closable Probe###synthetic_closable_probe", self.open, 1 << 19
            )
            if visible:
                ctx.label("Content")
            ctx.end_window()

    probe = _ClosableProbe()
    state = {
        "frame": 0,
        "semantic_frame": 0,
        "semantic_frames_seen": 0,
        "phase": "find",
        "window": None,
        "close": None,
        "last_targets": [],
    }
    deadline = time.monotonic() + 10.0
    set_gui_semantic_capture_enabled(True)

    def _targets():
        snapshot = get_gui_semantic_snapshot()
        semantic_frame = int(snapshot.get("frame", 0) or 0)
        if semantic_frame and semantic_frame != state["semantic_frame"]:
            state["semantic_frame"] = semantic_frame
            state["semantic_frames_seen"] += 1
        state["last_targets"] = [
            {
                "semantic_id": target.get("semantic_id"),
                "kind": target.get("kind"),
                "rect": target.get("rect"),
                "visible": target.get("visible"),
                "enabled": target.get("enabled"),
            }
            for target in snapshot.get("targets", [])
        ]
        window = None
        close = None
        for target in snapshot.get("targets", []):
            if target.get("semantic_id") == "synthetic_closable_probe":
                window = target
            elif target.get("semantic_id") == "synthetic_closable_probe.close":
                close = target
        return window, close

    def on_update(_delta_time: float) -> None:
        state["frame"] += 1
        window, close = _targets()
        if (
            state["phase"] == "find"
            and window
            and close
            and window.get("visible")
            and close.get("visible")
        ):
            state["window"] = window
            state["close"] = close
            x, y = (float(v) for v in close["click_point"])
            state["point"] = (x, y)
            engine.queue_synthetic_mouse_motion_input(x, y, 0.0, 0.0)
            state["phase"] = "moved"
        elif state["phase"] == "moved":
            x, y = state["point"]
            engine.queue_synthetic_mouse_button_input(0, True, x, y)
            state["phase"] = "pressed"
        elif state["phase"] == "pressed":
            x, y = state["point"]
            engine.queue_synthetic_mouse_button_input(0, False, x, y)
            state["phase"] = "released"
        elif state["phase"] == "released" and not probe.open:
            engine.exit()
        elif state["semantic_frames_seen"] >= 8 or time.monotonic() >= deadline:
            engine.exit()

    engine.register_gui_renderable("test.synthetic_closable_probe", probe)
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.synthetic_closable_probe")

    assert state["window"] is not None, state["last_targets"]
    assert state["close"] is not None
    assert state["close"]["visible"] is True
    wx, wy, ww, wh = (float(v) for v in state["window"]["rect"])
    vx, vy, vw, vh = probe.viewport
    assert wx >= vx and wy >= vy
    assert wx + ww <= vx + vw
    assert wy + wh <= vy + vh
    assert probe.open is False


def test_docked_window_close_button_is_recorded(engine):
    """A dock tab must record the close control that ImGui actually renders."""

    class _DockedClosableProbe(InxGUIRenderable):
        def __init__(self) -> None:
            super().__init__()
            self.open = True

        def on_render(self, ctx) -> None:
            visible, self.open = ctx.begin_window_closable(
                "Synthetic Docked Probe###synthetic_docked_probe", self.open, 0
            )
            if visible:
                ctx.label("Content")
            ctx.end_window()

    probe = _DockedClosableProbe()
    state = {"frame": 0, "close": None}
    set_gui_semantic_capture_enabled(True)

    def on_update(_delta_time: float) -> None:
        state["frame"] += 1
        snapshot = get_gui_semantic_snapshot()
        state["close"] = next(
            (
                target
                for target in snapshot.get("targets", [])
                if target.get("semantic_id") == "synthetic_docked_probe.close"
            ),
            None,
        )
        if state["close"] is not None or state["frame"] >= 16:
            engine.exit()

    engine.register_gui_renderable("test.synthetic_docked_probe", probe)
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.synthetic_docked_probe")

    assert state["close"] is not None
    assert state["close"]["kind"] == "window_close"
    assert all(float(value) > 0.0 for value in state["close"]["rect"][2:])


def test_close_request_keeps_synthetic_input_operable_until_resolved(engine):
    """An intercepted close must not strand remote input at the confirmation UI."""

    manager = InputManager.instance()
    observed: dict[str, object] = {}
    frame = [0]

    def on_update(_delta_time: float) -> None:
        frame[0] += 1
        if frame[0] == 1:
            observed["close_sequence"] = engine.queue_synthetic_close_request()
        elif engine.is_close_requested() and "key_sequence" not in observed:
            observed["key_sequence"] = engine.queue_synthetic_key_input(26, True)
        elif "key_sequence" in observed and manager.get_key(26):
            observed["key_delivered"] = True
            engine.queue_synthetic_key_input(26, False)
            engine.cancel_close()
            engine.exit()
        elif frame[0] >= 32:
            engine.cancel_close()
            engine.exit()

    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)

    assert int(observed["close_sequence"]) > 0
    assert int(observed["key_sequence"]) > int(observed["close_sequence"])
    assert observed["key_delivered"] is True


def test_graphical_run_releases_gil_for_background_python_work(engine):
    """A long-running Editor loop must not starve the embedded MCP server."""
    first_frame = threading.Event()
    worker_finished = threading.Event()
    observed: dict[str, object] = {}
    frames = [0]

    def worker() -> None:
        if not first_frame.wait(timeout=2.0):
            observed["worker_timeout"] = True
            return
        observed["worker_ran"] = True
        worker_finished.set()
        engine.exit()

    def on_update(_delta_time: float) -> None:
        frames[0] += 1
        first_frame.set()
        # A regression that keeps the GIL inside engine.run() cannot wake the
        # worker. Keep the test finite instead of hanging the test process.
        if frames[0] >= 240:
            observed["fallback_exit"] = True
            engine.exit()

    thread = threading.Thread(target=worker, name="infernux-gil-probe")
    thread.start()
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        thread.join(timeout=2.0)

    assert observed.get("worker_timeout") is not True
    assert observed.get("worker_ran") is True
    assert worker_finished.is_set()
    assert observed.get("fallback_exit") is not True


def test_synthetic_ctrl_a_replaces_focused_imgui_text(engine):
    """Synthetic modifier state must reach ImGui shortcuts, not just InputManager."""
    from infernux.resources import engine_font_path

    engine.set_gui_font(engine_font_path, 18)

    class _TextProbe(InxGUIRenderable):
        def __init__(self) -> None:
            super().__init__()
            self.value = "UntitledScene"
            self.render_count = 0

        def on_render(self, ctx) -> None:
            self.render_count += 1
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            ctx.set_next_window_size(64.0, 64.0, 0)
            if ctx.begin_window("Synthetic Input Probe###synthetic_input_probe", True, 0):
                self.value = ctx.text_input("Name##synthetic_input_name", self.value, 128)
                ctx.record_semantic_item("text_input", "Name", True, "test.synthetic.ctrl_a.name")
            ctx.end_window()

    probe = _TextProbe()
    state = {"frame": 0, "clicked": False, "focused": False, "typed": False, "target": None}
    deadline = time.monotonic() + 5.0
    set_gui_semantic_capture_enabled(True)

    def _find_target():
        snapshot = get_gui_semantic_snapshot()
        for target in snapshot.get("targets", []):
            if target.get("semantic_id") == "test.synthetic.ctrl_a.name":
                return target
        return None

    def on_update(_delta_time: float) -> None:
        state["frame"] += 1
        target = _find_target()
        if (not state["clicked"] and target and target.get("visible")
                and target.get("enabled") and target.get("has_click_point")):
            x, y = map(float, target["click_point"])
            engine.queue_synthetic_mouse_motion_input(x, y, 0.0, 0.0)
            engine.queue_synthetic_mouse_button_input(0, True, x, y)
            engine.queue_synthetic_mouse_button_input(0, False, x, y)
            state["clicked"] = True
            state["target"] = target
        elif state["clicked"] and target and target.get("active") and not state["typed"]:
            engine.queue_synthetic_key_input(224, True)  # SDL_SCANCODE_LCTRL
            engine.queue_synthetic_key_input(4, True)    # SDL_SCANCODE_A
            engine.queue_synthetic_key_input(4, False)
            engine.queue_synthetic_key_input(224, False)
            engine.queue_synthetic_text_input("RacingEntry")
            state["focused"] = True
            state["typed"] = True
        elif state["typed"] and probe.value == "RacingEntry":
            engine.exit()
        # GUI updates have their own cadence, independent of the simulation loop.
        elif probe.render_count >= 16 or time.monotonic() >= deadline:
            engine.exit()

    engine.register_gui_renderable("test.synthetic_ctrl_a", probe)
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.synthetic_ctrl_a")

    assert state["target"] is not None
    assert state["focused"] is True, state
    assert state["typed"] is True, state
    assert probe.value == "RacingEntry", {"engine_frames": state["frame"], "gui_frames": probe.render_count}


def test_synthetic_ctrl_click_edits_semantic_vector_axis(engine):
    """Vector-axis semantics must expose the real ImGui numeric text path."""

    class _VectorProbe(InxGUIRenderable):
        def __init__(self) -> None:
            super().__init__()
            self.value = (0.0, 0.0, 0.0)

        def on_render(self, ctx) -> None:
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            # The shared native test engine has a 64x64 SDL viewport. Keep
            # every generated axis inside it so this exercises real hit
            # testing instead of an ImGui item clipped beyond the window.
            ctx.set_next_window_size(64.0, 64.0, 0)
            if ctx.begin_window("Synthetic Vector Probe###synthetic_vector_probe", True, 0):
                self.value = ctx.vector3(
                    "P",
                    *self.value,
                    label_width=8.0,
                    semantic_id="test.vector.position",
                )
            ctx.end_window()

    probe = _VectorProbe()
    state = {"frame": 0, "activated": False, "typed": False, "axis": None}
    set_gui_semantic_capture_enabled(True)

    def _find_axis():
        snapshot = get_gui_semantic_snapshot()
        for target in snapshot.get("targets", []):
            if target.get("semantic_id") == "test.vector.position.y":
                return target
        return None

    def on_update(_delta_time: float) -> None:
        state["frame"] += 1
        axis = _find_axis()
        if not state["activated"] and axis:
            rect = axis["rect"]
            x = float(rect[0]) + float(rect[2]) * 0.5
            y = float(rect[1]) + float(rect[3]) * 0.5
            engine.queue_synthetic_key_input(224, True)  # SDL_SCANCODE_LCTRL
            engine.queue_synthetic_mouse_motion_input(x, y, 0.0, 0.0)
            engine.queue_synthetic_mouse_button_input(0, True, x, y)
            engine.queue_synthetic_mouse_button_input(0, False, x, y)
            state["activated"] = True
            state["axis"] = axis
        elif state["activated"] and axis and axis.get("focused") and not state["typed"]:
            engine.queue_synthetic_key_input(224, False)
            engine.queue_synthetic_key_input(224, True)  # SDL_SCANCODE_LCTRL
            engine.queue_synthetic_key_input(4, True)    # SDL_SCANCODE_A
            engine.queue_synthetic_key_input(4, False)
            engine.queue_synthetic_key_input(224, False)
            engine.queue_synthetic_text_input("3.5")
            state["typed"] = True
        elif state["typed"] and abs(probe.value[1] - 3.5) < 1e-6:
            engine.exit()
        elif state["frame"] >= 64:
            engine.exit()

    engine.register_gui_renderable("test.synthetic_vector_axis", probe)
    try:
        engine.set_pre_scene_update_callback(on_update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.synthetic_vector_axis")

    assert state["axis"] is not None
    assert state["activated"] is True
    assert state["typed"] is True
    assert abs(probe.value[1] - 3.5) < 1e-6
