"""Tutorial callback binding mistakes, using real scene components."""
import infernux as inx
import pytest

from infernux.engine import runtime_event_queue
from infernux.ui.ui_event_system import UIEventProcessor
from infernux.ui.ui_event_entry import UIEventEntry, get_callable_methods


class TutorialActionProbe(inx.InxComponent):
    calls: int = inx.serialized_field(default=0)

    def action(self) -> None:
        self.calls += 1

    def _private_action(self) -> None:
        self.calls += 100

    def update(self, delta_time: float) -> None:
        pass


def test_persistent_and_runtime_subscriptions_are_two_independent_invocations(scene):
    target = scene.create_game_object("TutorialActionTarget")
    probe = target.add_component(TutorialActionProbe)
    button = scene.create_game_object("TutorialActionButton").add_component(inx.UIButton)
    entry = UIEventEntry()
    entry.target = target
    entry.component_name = "TutorialActionProbe"
    entry.method_name = "action"
    button.on_click_entries = [entry]
    button.on_click.add_listener(probe.action)

    button.on_pointer_click(None)
    assert probe.calls == 2
    assert button._last_persistent_dispatch[0]["status"] == "invoked"

    button.on_click.remove_listener(probe.action)
    button.on_pointer_click(None)
    assert probe.calls == 3


def test_callback_picker_excludes_private_and_lifecycle_methods(scene):
    probe = scene.create_game_object("TutorialMethodTarget").add_component(TutorialActionProbe)
    methods = get_callable_methods(probe)
    assert "action" in methods
    assert "_private_action" not in methods
    assert "update" not in methods


@pytest.mark.parametrize(
    "blocked_by",
    ["canvas_disabled", "button_disabled", "not_interactable", "covered", "outside_view", "raycast_disabled"],
)
def test_documented_pointer_blockers_stop_clicks_and_recover(scene, blocked_by):
    root = scene.create_game_object("TutorialCanvas")
    canvas = root.add_component(inx.ui.UICanvas)
    canvas.reference_width, canvas.reference_height = 800, 600
    canvas.set_input_logical_size(800, 600)
    owner = scene.create_game_object("TutorialButton")
    owner.set_parent(root, False)
    button = owner.add_component(inx.ui.UIButton)
    button.set_rect(0, 0, 200, 40, 800, 600)
    calls = []
    button.on_click.add_listener(lambda: calls.append("click"))
    processor = UIEventProcessor()

    def click():
        processor.process([canvas], [(20, 20)], True, False, True, (0, 0), .016)
        runtime_event_queue.drain()
        processor.process([canvas], [(20, 20)], False, True, False, (0, 0), .016)
        runtime_event_queue.drain()

    runtime_event_queue.clear()
    try:
        click()
        assert calls == ["click"]

        cover = None
        if blocked_by == "canvas_disabled":
            canvas.enabled = False
        elif blocked_by == "button_disabled":
            button.enabled = False
        elif blocked_by == "not_interactable":
            button.interactable = False
        elif blocked_by == "covered":
            cover_owner = scene.create_game_object("FrontmostRaycastTarget")
            cover_owner.set_parent(root, False)
            cover = cover_owner.add_component(inx.ui.UIFrame)
            cover.raycast_target = True
            cover.set_rect(0, 0, 200, 40, 800, 600)
        elif blocked_by == "outside_view":
            button.set_rect(900, 0, 200, 40, 800, 600)
        else:
            button.raycast_target = False

        click()
        assert calls == ["click"]

        canvas.enabled = True
        button.enabled = True
        button.interactable = True
        button.raycast_target = True
        button.set_rect(0, 0, 200, 40, 800, 600)
        if cover is not None:
            cover.raycast_target = False
        click()
        assert calls == ["click", "click"]
    finally:
        processor.reset()
        runtime_event_queue.clear()
