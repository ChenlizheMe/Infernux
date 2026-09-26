"""Mouse callbacks obey the same publication boundary as UI and physics."""

from types import SimpleNamespace

import pytest

from Infernux.components import InxComponent
from Infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch
from Infernux.engine.runtime_mouse_events import MouseEventDispatcher


class MouseProbe(InxComponent):
    def on_mouse_enter(self):
        self.events.append("enter")

    def on_mouse_over(self):
        self.events.append("old-over")

    def on_mouse_exit(self):
        self.events.append("exit")

    def on_mouse_down(self):
        self.events.append("down")

    def on_mouse_up(self):
        self.events.append("up")

    def on_mouse_up_as_button(self):
        self.events.append("button")

    def on_mouse_drag(self):
        self.events.append("drag")


@pytest.fixture
def mouse(monkeypatch):
    probe = MouseProbe()
    probe.events = []
    publication = publish_runtime_dispatch_epoch((MouseProbe,))
    publication.commit()
    queries = []
    target = SimpleNamespace(id=991, get_py_components=lambda: queries.append(1) or (probe,))
    dispatcher = MouseEventDispatcher()
    hit = SimpleNamespace(game_object=target)
    state = [False, False, False]  # held, down, up
    monkeypatch.setattr("Infernux.engine.runtime_mouse_events.Input.get_game_mouse_frame_state",
                        lambda _button: (0, 0, 0, 0, *state))
    try:
        yield probe, dispatcher, hit, state, queries
    finally:
        publication.rollback()


def run(dispatcher, hit, **kwargs):
    dispatcher.process(object(), (1, 1), (100, 100), hit=hit, **kwargs)


def test_unpublished_class_edit_cannot_change_a_mouse_callback(mouse, monkeypatch):
    probe, dispatcher, hit, _state, _queries = mouse
    monkeypatch.setattr(MouseProbe, "on_mouse_over", lambda self: self.events.append("unpublished"))
    run(dispatcher, hit)
    assert probe.events == ["enter", "old-over"]


def test_same_frame_uses_one_epoch_then_next_frame_sees_publication(mouse, monkeypatch):
    probe, dispatcher, hit, _state, _queries = mouse
    updates = []

    def enter(self):
        self.events.append("enter")
        monkeypatch.setattr(MouseProbe, "on_mouse_over", lambda self: self.events.append("new-over"))
        update = publish_runtime_dispatch_epoch((MouseProbe,))
        update.commit()
        updates.append(update)

    monkeypatch.setattr(MouseProbe, "on_mouse_enter", enter)
    before = publish_runtime_dispatch_epoch((MouseProbe,))
    before.commit()
    try:
        run(dispatcher, hit)
        run(dispatcher, hit)
        assert probe.events == ["enter", "old-over", "new-over"]
    finally:
        for update in reversed(updates):
            update.rollback()
        before.rollback()


def test_one_object_component_snapshot_per_frame(mouse):
    probe, dispatcher, hit, state, queries = mouse
    state[:] = [True, True, False]
    run(dispatcher, hit)
    assert queries == [1]
    assert probe.events == ["enter", "old-over", "down", "drag"]


def test_reset_retires_capture_before_failed_exit(mouse, monkeypatch):
    probe, dispatcher, hit, state, _queries = mouse
    state[:] = [True, True, False]
    run(dispatcher, hit)

    def fail(self):
        self.events.append("failed-exit")
        raise ValueError("author failure")

    monkeypatch.setattr(MouseProbe, "on_mouse_exit", fail)
    update = publish_runtime_dispatch_epoch((MouseProbe,))
    update.commit()
    try:
        with pytest.raises(ValueError, match="author failure"):
            dispatcher.reset()
        dispatcher.reset()
        assert dispatcher._hover_object is None
        assert dispatcher._pressed_object is None
        assert probe.events.count("failed-exit") == 1
    finally:
        update.rollback()


def test_input_snapshot_is_read_once_and_explicit_snapshot_is_reused(mouse, monkeypatch):
    probe, dispatcher, hit, state, _queries = mouse
    reads = []
    monkeypatch.setattr("Infernux.engine.runtime_mouse_events.Input.get_game_mouse_frame_state",
                        lambda button: reads.append(button) or (0, 0, 0, 0, *state))
    run(dispatcher, hit)
    assert reads == [0]
    run(dispatcher, hit, button_state=(True, True, False))
    assert reads == [0]
    assert probe.events[-2:] == ["down", "drag"]


def test_release_over_another_object_does_not_click_it(mouse):
    probe, dispatcher, hit, state, _queries = mouse
    state[:] = [True, True, False]
    run(dispatcher, hit)
    state[:] = [False, False, True]
    other = SimpleNamespace(game_object=SimpleNamespace(id=992, get_py_components=lambda: ()))
    run(dispatcher, other)
    assert probe.events[-2:] == ["exit", "up"]
    assert "button" not in probe.events
    assert dispatcher._pressed_object is None


def test_release_failure_cannot_retain_or_replay_pressed_transaction(mouse, monkeypatch):
    probe, dispatcher, hit, state, _queries = mouse
    state[:] = [True, True, False]
    run(dispatcher, hit)

    def fail(self):
        self.events.append("failed-up")
        raise ValueError("release failed")

    monkeypatch.setattr(MouseProbe, "on_mouse_up", fail)
    publication = publish_runtime_dispatch_epoch((MouseProbe,))
    publication.commit()
    try:
        state[:] = [False, False, True]
        with pytest.raises(ValueError, match="release failed"):
            run(dispatcher, hit)
        assert dispatcher._pressed_object is None
        run(dispatcher, hit)
        assert probe.events.count("failed-up") == 1
        assert "button" not in probe.events
    finally:
        publication.rollback()


def test_callback_reset_cancels_remaining_events_in_current_dispatch(mouse, monkeypatch):
    probe, dispatcher, hit, state, _queries = mouse
    monkeypatch.setattr(MouseProbe, "on_mouse_enter", lambda self: dispatcher.reset())
    publication = publish_runtime_dispatch_epoch((MouseProbe,))
    publication.commit()
    try:
        state[:] = [True, True, False]
        run(dispatcher, hit)
        assert probe.events == ["exit"]
        assert dispatcher._hover_object is None and dispatcher._pressed_object is None
    finally:
        publication.rollback()


def test_destroyed_components_are_not_called_from_snapshot(mouse):
    probe, dispatcher, hit, state, _queries = mouse
    probe._is_destroyed = True
    state[:] = [True, True, False]
    run(dispatcher, hit)
    dispatcher.reset()
    assert probe.events == []


def test_component_added_during_callback_waits_until_next_input_frame(mouse, monkeypatch):
    probe, dispatcher, hit, _state, _queries = mouse
    second = MouseProbe()
    second.events = []
    components = [probe]
    hit.game_object.get_py_components = lambda: components
    monkeypatch.setattr(MouseProbe, "on_mouse_enter", lambda self: components.append(second))
    publication = publish_runtime_dispatch_epoch((MouseProbe,))
    publication.commit()
    try:
        run(dispatcher, hit)
        assert second.events == []
        run(dispatcher, hit)
        assert second.events == ["old-over"]
    finally:
        publication.rollback()


def test_missing_native_identity_does_not_merge_distinct_objects():
    assert not MouseEventDispatcher._same_object(object(), object())


def test_reset_in_first_component_stops_other_component_events(mouse, monkeypatch):
    probe, dispatcher, hit, _state, _queries = mouse
    second = MouseProbe()
    second.events = []
    hit.game_object.get_py_components = lambda: (probe, second)
    monkeypatch.setattr(MouseProbe, "on_mouse_enter", lambda self: dispatcher.reset())
    publication = publish_runtime_dispatch_epoch((MouseProbe,))
    publication.commit()
    try:
        run(dispatcher, hit)
        assert probe.events == ["exit"]
        assert second.events == ["exit"]
        assert dispatcher._hover_object is None
    finally:
        publication.rollback()
