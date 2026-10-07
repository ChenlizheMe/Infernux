"""Real SDL -> ImGui -> Python drag/drop preserves the authored value type."""
import pytest

from infernux import lib
from infernux.engine.ui.theme import ImGuiWindowFlags


@pytest.mark.parametrize("value", ["choiceA", "中文a", "", "a\0bcdef", 0, 2**64 - 1])
@pytest.mark.parametrize("any_type", [False, True])
def test_native_drag_drop_round_trip_through_python(engine, value, any_type):
    state = {"frame": 0, "source": None, "target": None, "start": None,
             "delivered": [], "errors": [], "source_frames": 0}
    payload_type = "PYTHON_DRAG_VALUE"

    class Probe(lib.InxGUIRenderable):
        def on_render(self, ctx):
            ctx.set_next_window_pos(0., 0., 0, 0., 0.)
            ctx.set_next_window_size(64., 64., 0)
            ctx.set_next_window_focus()
            flags = ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoScrollbar
            visible = ctx.begin_window("Payload###python_payload_contract", True, flags)
            try:
                if not visible:
                    return
                for name, y in (("source", 6.), ("target", 36.)):
                    ctx.set_cursor_pos_x(8.)
                    ctx.set_cursor_pos_y(y)
                    ctx.button(name[0].upper(), None, 30., 18.)
                    state[name] = ((ctx.get_item_rect_min_x() + ctx.get_item_rect_max_x()) / 2,
                                   (ctx.get_item_rect_min_y() + ctx.get_item_rect_max_y()) / 2)
                    if name == "source" and ctx.begin_drag_drop_source():
                        try:
                            setter = ctx.set_drag_drop_payload_str if isinstance(value, str) else ctx.set_drag_drop_payload
                            setter(payload_type, value)
                            ctx.label("Value")
                            state["source_frames"] += 1
                        finally:
                            ctx.end_drag_drop_source()
                    if name == "target" and ctx.begin_drag_drop_target():
                        try:
                            assert ctx.accept_drag_drop_payload("WRONG_TYPE") is None
                            result = (ctx.accept_any_drag_drop_payload() if any_type
                                      else ctx.accept_drag_drop_payload(payload_type))
                            if result is not None:
                                state["delivered"].append(result)
                        finally:
                            ctx.end_drag_drop_target()
            except Exception as error:
                state["errors"].append(error)
            finally:
                ctx.end_window()

    def update(_delta):
        state["frame"] += 1
        if state["frame"] > 40 or state["errors"]:
            engine.exit()
            return
        if state["source"] is None or state["target"] is None:
            return
        if state["start"] is None:
            state["start"] = state["frame"]
        step = state["frame"] - state["start"]
        source, target = state["source"], state["target"]
        if step == 0:
            engine.queue_synthetic_mouse_motion_input(*source, 0., 0.)
        elif step == 2:
            engine.queue_synthetic_mouse_button_input(0, True, *source)
        elif step == 4:
            engine.queue_synthetic_mouse_motion_input(*target, 0., target[1] - source[1])
        elif step == 8:
            engine.queue_synthetic_mouse_button_input(0, False, *target)
        elif step >= 12:
            engine.exit()

    probe = Probe()
    engine.register_gui_renderable("test.python_drag_drop", probe)
    try:
        engine.set_pre_scene_update_callback(update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.python_drag_drop")
    assert not state["errors"], repr(state["errors"])
    assert state["source_frames"] > 0, state
    expected = (payload_type, value) if any_type else value
    assert state["delivered"] == [expected], state
    observed = state["delivered"][0][1] if any_type else state["delivered"][0]
    assert type(observed) is type(value)
