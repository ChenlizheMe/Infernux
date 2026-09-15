"""Focused R3-B3 tests for lifecycle, physics, and UI pointer dispatch."""

from __future__ import annotations

import pytest

from Infernux.components import InxComponent
from Infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch
from Infernux.ui.ui_event_data import PointerType
from Infernux.ui.ui_event_system import UIEventProcessor, UIPointerFrame


class _PhysicsProbe(InxComponent):
    def on_collision_stay(self, collision) -> None:
        self.events.append(("old", collision))

    def on_trigger_enter(self, other) -> None:
        self.events.append(("trigger-old", other))

    def awake(self) -> None:
        self.events = []


def test_physics_callbacks_resolve_the_published_body():
    probe = _PhysicsProbe()
    probe.events = []
    initial = publish_runtime_dispatch_epoch((_PhysicsProbe,))
    initial.commit()
    old_collision = _PhysicsProbe.on_collision_stay
    old_trigger = _PhysicsProbe.on_trigger_enter
    try:
        probe._call_on_collision_stay("first")
        probe._call_on_trigger_enter("first-trigger")

        def new_collision(self, collision) -> None:
            self.events.append(("new", collision))

        def new_trigger(self, other) -> None:
            self.events.append(("trigger-new", other))

        _PhysicsProbe.on_collision_stay = new_collision
        _PhysicsProbe.on_trigger_enter = new_trigger
        publication = publish_runtime_dispatch_epoch((_PhysicsProbe,))
        publication.commit()
        try:
            probe._call_on_collision_stay("second")
            probe._call_on_trigger_enter("second-trigger")
            assert probe.events == [
                ("old", "first"),
                ("trigger-old", "first-trigger"),
                ("new", "second"),
                ("trigger-new", "second-trigger"),
            ]
        finally:
            publication.rollback()
    finally:
        _PhysicsProbe.on_collision_stay = old_collision
        _PhysicsProbe.on_trigger_enter = old_trigger
        initial.rollback()


class _PointerProbe(InxComponent):
    def awake(self) -> None:
        self.events = []

    def on_pointer_enter(self, event) -> None:
        self.events.append("enter")

    def on_pointer_exit(self, event) -> None:
        self.events.append("exit")

    def on_pointer_down(self, event) -> None:
        self.events.append("down")

    def on_pointer_up(self, event) -> None:
        self.events.append("up")

    def on_pointer_click(self, event) -> None:
        self.events.append("click")

    def on_begin_drag(self, event) -> None:
        self.events.append("begin_drag")

    def on_drag(self, event) -> None:
        self.events.append("drag")

    def on_end_drag(self, event) -> None:
        self.events.append("end_drag")

    def on_scroll(self, event) -> None:
        self.events.append("scroll")


class _Canvas:
    game_object = None
    enabled = True
    input_logical_size = (100.0, 100.0)

    def __init__(self, target) -> None:
        self.target = target
        self.hit = True

    def raycast(self, _x, _y):
        return self.target if self.hit else None


def _make_pointer_target():
    target = _PointerProbe()
    target.events = []
    target._try_get_game_object = lambda: type("DebugOwner", (), {"name": "PointerProbe"})()
    return target


def test_ui_process_routes_all_pointer_hooks_through_one_event_path():
    target = _make_pointer_target()
    canvas = _Canvas(target)
    publication = publish_runtime_dispatch_epoch((_PointerProbe,))
    publication.commit()
    try:
        processor = UIEventProcessor()
        processor.process([canvas], [(0.0, 0.0)], True, False, True, (0.0, 0.0), 0.016)
        processor.process([canvas], [(10.0, 0.0)], False, False, True, (0.0, 0.0), 0.016)
        processor.process([canvas], [(20.0, 0.0)], False, False, True, (1.0, 0.0), 0.016)
        processor.process([canvas], [(20.0, 0.0)], False, True, False, (0.0, 0.0), 0.016)
        canvas.hit = False
        processor.process([canvas], [(30.0, 0.0)], False, False, False, (0.0, 0.0), 0.016)

        assert target.events == [
            "enter",
            "down",
            "begin_drag",
            "drag",
            "scroll",
            "up",
            "click",
            "end_drag",
            "exit",
        ]
    finally:
        publication.rollback()


def test_ui_process_keeps_one_epoch_when_a_callback_publishes():
    target = _make_pointer_target()
    canvas = _Canvas(target)
    initial = publish_runtime_dispatch_epoch((_PointerProbe,))
    initial.commit()
    old_enter = _PointerProbe.on_pointer_enter
    old_down = _PointerProbe.on_pointer_down
    publication_box = []
    try:
        def replacement_down(self, event) -> None:
            self.events.append("down-new")

        def publishing_enter(self, event) -> None:
            self.events.append("enter-old")
            _PointerProbe.on_pointer_down = replacement_down
            publication = publish_runtime_dispatch_epoch((_PointerProbe,))
            publication.commit()
            publication_box.append(publication)

        _PointerProbe.on_pointer_enter = publishing_enter
        publication = publish_runtime_dispatch_epoch((_PointerProbe,))
        publication.commit()
        try:
            processor = UIEventProcessor()
            processor.process(
                [canvas],
                [(0.0, 0.0)],
                True,
                False,
                True,
                (0.0, 0.0),
                0.016,
            )
            assert target.events == ["enter-old", "down"]
        finally:
            publication.rollback()
    finally:
        for item in reversed(publication_box):
            item.rollback()
        _PointerProbe.on_pointer_enter = old_enter
        _PointerProbe.on_pointer_down = old_down
        initial.rollback()


def test_ui_pointer_exception_propagates_without_retry():
    target = _make_pointer_target()
    canvas = _Canvas(target)
    old_click = _PointerProbe.on_pointer_click

    def failing_click(self, event) -> None:
        self.events.append("click-failed")
        raise RuntimeError("expected pointer failure")

    _PointerProbe.on_pointer_click = failing_click
    publication = publish_runtime_dispatch_epoch((_PointerProbe,))
    publication.commit()
    try:
        processor = UIEventProcessor()
        processor.process([canvas], [(0.0, 0.0)], True, False, True, (0.0, 0.0), 0.016)
        with pytest.raises(RuntimeError, match="expected pointer failure"):
            processor.process([canvas], [(0.0, 0.0)], False, True, False, (0.0, 0.0), 0.016)
        assert target.events.count("click-failed") == 1
    finally:
        publication.rollback()
        _PointerProbe.on_pointer_click = old_click


def test_ui_process_keeps_simultaneous_touch_transactions_independent():
    left = _make_pointer_target()
    right = _make_pointer_target()

    class SplitCanvas:
        game_object = None
        enabled = True
        input_logical_size = (100.0, 100.0)

        @staticmethod
        def raycast(x, _y):
            return left if x < 50.0 else right

    publication = publish_runtime_dispatch_epoch((_PointerProbe,))
    publication.commit()
    try:
        processor = UIEventProcessor()
        processor.process_pointers(
            [SplitCanvas()],
            (
                UIPointerFrame(10, PointerType.Touch, ((10.0, 0.0),), down=True, held=True),
                UIPointerFrame(11, PointerType.Touch, ((90.0, 0.0),), down=True, held=True),
            ),
            0.016,
        )
        processor.process_pointers(
            [SplitCanvas()],
            (
                UIPointerFrame(10, PointerType.Touch, ((10.0, 0.0),), up=True),
                UIPointerFrame(11, PointerType.Touch, ((90.0, 0.0),), up=True),
            ),
            0.016,
        )

        assert left.events == ["enter", "down", "up", "click", "exit"]
        assert right.events == ["enter", "down", "up", "click", "exit"]
    finally:
        publication.rollback()


def test_ui_touch_cancel_releases_capture_without_click():
    target = _make_pointer_target()
    canceled = []
    old_up = _PointerProbe.on_pointer_up

    def capture_cancel(self, event) -> None:
        self.events.append("up")
        canceled.append((event.pointer_id, event.pointer_type, event.canceled))

    _PointerProbe.on_pointer_up = capture_cancel
    publication = publish_runtime_dispatch_epoch((_PointerProbe,))
    publication.commit()
    try:
        processor = UIEventProcessor()
        processor.process_pointers(
            [_Canvas(target)],
            (UIPointerFrame(27, PointerType.Touch, ((0.0, 0.0),), down=True, held=True),),
            0.016,
        )
        processor.process_pointers(
            [_Canvas(target)],
            (
                UIPointerFrame(
                    27,
                    PointerType.Touch,
                    ((0.0, 0.0),),
                    up=True,
                    canceled=True,
                ),
            ),
            0.016,
        )

        assert target.events == ["enter", "down", "up", "exit"]
        assert canceled == [(27, PointerType.Touch, True)]
    finally:
        publication.rollback()
        _PointerProbe.on_pointer_up = old_up


def test_ui_group_interactable_change_cancels_active_capture():
    target = _make_pointer_target()
    target.accepts_interaction = True
    target.is_effectively_interactable = lambda: target.accepts_interaction
    canceled = []
    old_up = _PointerProbe.on_pointer_up

    def capture_cancel(self, event) -> None:
        self.events.append("up")
        canceled.append(event.canceled)

    _PointerProbe.on_pointer_up = capture_cancel
    publication = publish_runtime_dispatch_epoch((_PointerProbe,))
    publication.commit()
    try:
        processor = UIEventProcessor()
        canvas = _Canvas(target)
        processor.process(
            [canvas], [(0.0, 0.0)], True, False, True, (0.0, 0.0), 0.016
        )
        target.accepts_interaction = False
        processor.process(
            [canvas], [(1.0, 0.0)], False, False, True, (0.0, 0.0), 0.016
        )

        assert target.events == ["enter", "down", "up", "exit"]
        assert canceled == [True]
    finally:
        publication.rollback()
        _PointerProbe.on_pointer_up = old_up


def test_ui_process_prefers_screen_canvas_over_world_element():
    world_target = _make_pointer_target()
    screen_target = _make_pointer_target()
    world = _Canvas(world_target)
    world.input_priority = lambda position, _order: (0, 0, -position[2])
    screen = _Canvas(screen_target)
    publication = publish_runtime_dispatch_epoch((_PointerProbe,))
    publication.commit()
    try:
        processor = UIEventProcessor()
        processor.process_pointers(
            (world, screen),
            (UIPointerFrame(-1, PointerType.Mouse, ((20.0, 10.0, 2.0), (20.0, 10.0)), down=True),),
            0.016,
        )

        assert world_target.events == []
        assert screen_target.events == ["enter", "down"]
    finally:
        publication.rollback()


def test_ui_drag_uses_the_captured_world_element_coordinates():
    target = _make_pointer_target()
    world = _Canvas(target)
    world.input_priority = lambda position, _order: (0, 0, -position[2])
    publication = publish_runtime_dispatch_epoch((_PointerProbe,))
    publication.commit()
    try:
        processor = UIEventProcessor()
        processor.process_pointers(
            (world,),
            (UIPointerFrame(-1, PointerType.Mouse, ((10.0, 10.0, 2.0),), down=True, held=True),),
            0.016,
        )
        processor.process_pointers(
            (world,),
            (UIPointerFrame(-1, PointerType.Mouse, ((20.0, 10.0, 1.8),), held=True),),
            0.016,
        )

        assert target.events == ["enter", "down", "begin_drag"]
    finally:
        publication.rollback()
