"""Selectable visual feedback aggregates independent pointer identities."""
import pytest

from infernux.engine import runtime_event_queue
from infernux.ui import UIButton, UICanvas, PointerEventData, PointerType, UIPointerFrame
from infernux.ui.ui_event_system import UIEventProcessor
from infernux.ui.ui_render_revision import get_runtime_ui_revision
from infernux.ui.ui_selectable import SelectionState


@pytest.fixture
def controls(scene):
    runtime_event_queue.clear()
    root = scene.create_game_object("Two pointer canvas")
    canvas = root.add_py_component(UICanvas())
    canvas.reference_width, canvas.reference_height = 800, 600
    canvas.set_input_logical_size(800, 600)
    owner = scene.create_game_object("Two pointer button")
    owner.set_parent(root, False)
    button = owner.add_py_component(UIButton())
    button.set_rect(0, 0, 200, 40, 800, 600)
    button.awake()
    processor = UIEventProcessor()
    yield canvas, button, processor
    processor.reset()
    runtime_event_queue.drain()
    runtime_event_queue.clear()


def frame(processor, canvas, pointers):
    processor.process_pointers([canvas], pointers, .016)
    runtime_event_queue.drain()


def pointer(kind, identifier, x=40, **flags):
    return UIPointerFrame(identifier, kind, ((x, 20),), **flags)


@pytest.mark.parametrize("other_kind,other_id", [(PointerType.Touch, 2), (PointerType.Mouse, 1)])
@pytest.mark.parametrize("first_end", ["up", "cancel", "exit", "disappear"])
def test_remaining_pointer_keeps_pressed_without_rebuilding_draw_state(controls, other_kind, other_id, first_end):
    canvas, button, processor = controls
    first = pointer(PointerType.Touch, 1, down=True)
    other = pointer(other_kind, other_id, x=70, down=True)
    frame(processor, canvas, [first, other])
    assert button.current_selection_state == SelectionState.Pressed
    revision = get_runtime_ui_revision()
    remaining = [pointer(other_kind, other_id, x=70, held=True)]
    if first_end != "disappear":
        remaining.insert(0, pointer(PointerType.Touch, 1, x=400 if first_end == "exit" else 40,
                                    up=first_end == "up", canceled=first_end == "cancel",
                                    held=first_end == "exit"))
    frame(processor, canvas, remaining)
    assert button.current_selection_state == SelectionState.Pressed
    assert button.get_current_tint() == button.pressed_color
    assert get_runtime_ui_revision() == revision, "unchanged aggregate state invalidated draw packets"
    assert processor._pointers[(other_kind, other_id)].press_target is button
    processor.reset()
    runtime_event_queue.drain()
    assert button.current_selection_state == SelectionState.Normal


@pytest.mark.parametrize("first_kind", [PointerType.Mouse, PointerType.Touch])
def test_hover_exit_only_clears_its_own_pointer(controls, first_kind):
    canvas, button, processor = controls
    other_kind = PointerType.Touch if first_kind == PointerType.Mouse else PointerType.Mouse
    frame(processor, canvas, [pointer(first_kind, 4), pointer(other_kind, 4)])
    assert button.current_selection_state == SelectionState.Highlighted
    frame(processor, canvas, [pointer(first_kind, 4, x=400), pointer(other_kind, 4)])
    assert button.current_selection_state == SelectionState.Highlighted


@pytest.mark.parametrize("hook", ["on_disable", "on_destroy"])
def test_lifecycle_clears_all_transient_pointer_visual_state(controls, hook):
    canvas, button, processor = controls
    frame(processor, canvas, [pointer(PointerType.Touch, 1, down=True), pointer(PointerType.Touch, 2, down=True)])
    getattr(button, hook)()
    assert button.current_selection_state == SelectionState.Normal
    assert button.get_current_tint() == button.normal_color


def test_repeated_same_pointer_events_are_idempotent(controls):
    _, button, _ = controls
    event = PointerEventData()
    event.pointer_type, event.pointer_id = PointerType.Touch, 42
    for _ in range(4):
        button.on_pointer_enter(event)
        button.on_pointer_down(event)
    assert button.current_selection_state == SelectionState.Pressed
    button.on_pointer_up(event)
    assert button.current_selection_state == SelectionState.Highlighted
    button.on_pointer_exit(event)
    assert button.current_selection_state == SelectionState.Normal


def test_native_owner_destruction_clears_pointer_visual_state(scene, controls):
    canvas, button, processor = controls
    frame(processor, canvas, [pointer(PointerType.Touch, 1, down=True), pointer(PointerType.Touch, 2, down=True)])
    scene.destroy_game_object(button.game_object)
    scene.process_pending_destroys()
    assert not button.is_valid
    assert button.current_selection_state == SelectionState.Normal
