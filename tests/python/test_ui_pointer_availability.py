"""Live UI eligibility applies to captured and queued pointer transactions."""

import pytest

from infernux.engine import runtime_event_queue
from infernux.ui import UIButton, UICanvas, UIGroup, UISlider, UIPointerFrame, PointerButton, PointerType
from infernux.ui.ui_event_system import UIEventProcessor, _mouse_pointer_frames


class TrackedSlider(UISlider):
    def awake(self):
        super().awake()
        self.terminals = []

    def on_pointer_up(self, event):
        super().on_pointer_up(event)
        self.terminals.append(("up", event.canceled))

    def on_end_drag(self, event):
        self.terminals.append(("end", event.canceled))

    def on_pointer_exit(self, event):
        super().on_pointer_exit(event)
        self.terminals.append(("exit", event.canceled))


class TrackedButton(UIButton):
    def awake(self):
        super().awake()
        self.events = []

    def on_pointer_enter(self, event):
        super().on_pointer_enter(event)
        self.events.append(("enter", event.pointer_id, event.button, event.canceled))

    def on_pointer_down(self, event):
        super().on_pointer_down(event)
        self.events.append(("down", event.pointer_id, event.button, event.canceled))

    def on_pointer_up(self, event):
        super().on_pointer_up(event)
        self.events.append(("up", event.pointer_id, event.button, event.canceled))


@pytest.fixture(autouse=True)
def isolated_queue():
    runtime_event_queue.clear()
    yield
    runtime_event_queue.clear()


def make_canvas(scene, name="Canvas"):
    root = scene.create_game_object(name)
    canvas = root.add_py_component(UICanvas())
    canvas.reference_width, canvas.reference_height = 800, 600
    canvas.set_input_logical_size(800, 600)
    return root, canvas


def make_control(scene, root, kind, x=0):
    owner = scene.create_game_object(kind.__name__)
    owner.set_parent(root, False)
    control = owner.add_py_component(kind())
    control.set_rect(x, 0, 200, 40, 800, 600)
    control.awake()
    return control


def pointer(identifier, x=20, pointer_type=PointerType.Touch, **flags):
    return UIPointerFrame(identifier, pointer_type, ((float(x), 20.0),), **flags)


def test_mouse_buttons_keep_independent_capture_with_one_hover_and_one_button_click(scene):
    root, canvas = make_canvas(scene)
    control = make_control(scene, root, TrackedButton)
    processor, clicks = UIEventProcessor(), []
    control.on_click.add_listener(lambda: clicks.append("left"))
    for states in ([(True, True, False)] * 3, [(False, False, True)] * 3):
        frames = _mouse_pointer_frames(((20, 20),), states)
        processor.process_pointers([canvas], frames, 0.016)
        runtime_event_queue.drain()
    assert [item[0] for item in control.events].count("enter") == 1
    assert [(item[1], item[2]) for item in control.events if item[0] == "down"] == [
        (-1, PointerButton.Left), (-2, PointerButton.Right), (-3, PointerButton.Middle)]
    assert clicks == ["left"]
    right = _mouse_pointer_frames(((20, 20),), [(False, False, False), (True, True, False)])
    processor.process_pointers([canvas], right, 0.016)
    runtime_event_queue.drain()
    processor.process_pointers([canvas], _mouse_pointer_frames(((20, 20),), [(False, False, False)]), 0.016)
    runtime_event_queue.drain()
    assert control.events[-1] == ("up", -2, PointerButton.Right, True)
    assert processor._pointers[(PointerType.Mouse, -1)].hover_target is control
    assert clicks == ["left"]


def test_right_mouse_does_not_change_slider_value(scene):
    root, canvas = make_canvas(scene)
    slider = make_control(scene, root, UISlider)
    processor = UIEventProcessor()
    for x, state in ((20, (True, True, False)), (180, (True, False, False)), (190, (False, False, True))):
        processor.process_pointers([canvas], _mouse_pointer_frames(((x, 20),), [(False, False, False), state]), 0.016)
        runtime_event_queue.drain()
    assert slider.value == 0


def test_canvas_clips_new_hits_but_releases_a_drag_outside_its_rect(scene):
    root, canvas = make_canvas(scene)
    slider = make_control(scene, root, TrackedSlider, x=700)
    assert canvas.raycast(750, 20) is slider
    assert canvas.raycast(850, 20) is None
    processor = UIEventProcessor()
    for x, flags in ((850, dict(down=True, up=True)), (750, dict(down=True)),
                     (850, dict(held=True)), (850, dict(up=True))):
        processor.process_pointers([canvas], [pointer(1, x, **flags)], 0.016)
        runtime_event_queue.drain()
    assert slider.value == pytest.approx(0.75)
    assert ("up", False) in slider.terminals
    assert ("end", False) in slider.terminals


MODES = (
    "enabled",
    "owner_active",
    "parent_active",
    "group",
    "interactable",
    "canvas_enabled",
    "canvas_removed",
    "reparent",
)


def invalidate(scene, root, canvas, control, mode):
    if mode == "enabled":
        control.enabled = False
    elif mode == "owner_active":
        control.game_object.active = False
    elif mode == "parent_active":
        root.active = False
    elif mode == "group":
        control.game_object.add_py_component(UIGroup()).interactable = False
    elif mode == "interactable":
        control.interactable = False
    elif mode == "canvas_enabled":
        canvas.enabled = False
    elif mode == "canvas_removed":
        root.remove_py_component(canvas)
    elif mode == "reparent":
        destination, _ = make_canvas(scene, "Different coordinate system")
        control.game_object.set_parent(destination, True)


@pytest.mark.parametrize("mode", ["enabled", "reparent"])
def test_prior_click_can_disable_a_later_queued_click(scene, mode):
    root, canvas = make_canvas(scene)
    first = make_control(scene, root, UIButton)
    second = make_control(scene, root, UIButton, x=250)
    clicks = []

    def first_click():
        clicks.append("first")
        invalidate(scene, root, canvas, second, mode)

    first.on_click.add_listener(first_click)
    second.on_click.add_listener(lambda: clicks.append("second"))
    processor = UIEventProcessor()
    processor.process_pointers([canvas], [pointer(1, 40, down=True), pointer(2, 290, down=True)], 0.016)
    runtime_event_queue.drain()
    processor.process_pointers([canvas], [pointer(1, 40, up=True), pointer(2, 290, up=True)], 0.016)
    runtime_event_queue.drain()
    assert clicks == ["first"]
    assert not processor._pointers


@pytest.mark.parametrize(
    "mode,timing,pointer_type",
    [
        ("enabled", "before_frame", PointerType.Mouse),
        ("owner_active", "before_drain", PointerType.Touch),
        ("canvas_removed", "before_frame", PointerType.Touch),
        ("reparent", "before_drain", PointerType.Mouse),
    ],
)
def test_disabled_capture_cancels_once_without_late_value_changes(scene, mode, timing, pointer_type):
    root, canvas = make_canvas(scene)
    slider = make_control(scene, root, TrackedSlider)
    changes = []
    slider.on_value_changed.add_listener(changes.append)
    processor = UIEventProcessor()
    processor.process_pointers([canvas], [pointer(1, 20, pointer_type, down=True)], 0.016)
    runtime_event_queue.drain()
    processor.process_pointers([canvas], [pointer(1, 80, pointer_type, held=True)], 0.016)
    runtime_event_queue.drain()
    before, count = slider.value, len(changes)
    if timing == "before_frame":
        invalidate(scene, root, canvas, slider, mode)
    # Removing the old surface mirrors runtime surface collection after Canvas
    # removal / reparent. The same still-live target must lose that capture.
    surfaces = [] if mode in ("canvas_removed", "reparent") and timing == "before_frame" else [canvas]
    sample = pointer(1, 180, pointer_type, held=True)
    if not surfaces:
        sample = UIPointerFrame(1, pointer_type, (), held=True)
    processor.process_pointers(surfaces, [sample], 0.016)
    if timing == "before_drain":
        invalidate(scene, root, canvas, slider, mode)
    runtime_event_queue.drain()
    assert slider.value == before and len(changes) == count
    state = processor._pointers.get((pointer_type, 1))
    assert state is None or state.press_target is None
    assert slider.terminals == [("up", True), ("end", True), ("exit", True)]
    processor.reset()
    runtime_event_queue.drain()
    assert slider.terminals == [("up", True), ("end", True), ("exit", True)]


@pytest.mark.parametrize("mode", ["normal", "canceled", "destroyed", "discarded"])
def test_click_lifecycle_controls(scene, mode):
    root, canvas = make_canvas(scene)
    button = make_control(scene, root, UIButton)
    clicks = []
    button.on_click.add_listener(lambda: clicks.append("click"))
    processor = UIEventProcessor()
    processor.process_pointers([canvas], [pointer(1, down=True, up=True, canceled=mode == "canceled")], 0.016)
    if mode == "destroyed":
        scene.destroy_game_object(button.game_object)
        scene.process_pending_destroys()
    if mode == "discarded":
        processor.discard()
    runtime_event_queue.drain()
    assert clicks == (["click"] if mode == "normal" else [])


def test_gui_queued_click_rechecks_availability_after_first_callback(scene, engine, capfd):
    from infernux.lib import InxGUIRenderable
    from infernux.renderstack import RenderStackPipeline

    root, canvas = make_canvas(scene)
    first = make_control(scene, root, UIButton)
    second = make_control(scene, root, UIButton, x=250)
    processor = UIEventProcessor()
    clicks, errors = [], []
    inside_render, phase = False, 0

    def first_click():
        assert not inside_render, "author callbacks ran in the native GUI traversal"
        clicks.append("first")
        second.enabled = False

    first.on_click.add_listener(first_click)
    second.on_click.add_listener(lambda: clicks.append("second"))

    class Panel(InxGUIRenderable):
        def on_render(self, ctx):
            nonlocal inside_render, phase
            inside_render = True
            try:
                if phase < 2:
                    flags = {"down": True} if phase == 0 else {"up": True}
                    processor.process_pointers(
                        [canvas],
                        [pointer(1, 40, **flags), pointer(2, 290, **flags)],
                        0.016,
                    )
                    phase += 1
            except BaseException as error:
                errors.append(error)
                engine.exit()
            finally:
                inside_render = False

    def after_draw():
        try:
            runtime_event_queue.drain()
            if phase == 2:
                assert clicks == ["first"]
                engine.exit()
        except BaseException as error:
            errors.append(error)
            engine.exit()

    panel = Panel()
    pipeline = RenderStackPipeline()
    try:
        engine.set_render_pipeline(pipeline)
        engine.register_gui_renderable("pointer-availability-test", panel, 0)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        assert not errors, repr(errors)
        assert phase == 2 and clicks == ["first"]
    finally:
        engine.set_post_draw_callback(None)
        engine.unregister_gui_renderable("pointer-availability-test")
        engine.set_render_pipeline(None)
    captured = capfd.readouterr()
    for diagnostic in ("VUID-", "SYNC-HAZARD", "Validation Error"):
        assert diagnostic not in captured.out + captured.err


@pytest.mark.parametrize("cancel", [False, True])
def test_logical_ui_hooks_finish_after_traversal_before_fixed_with_owner_state(scene, monkeypatch, cancel):
    from types import SimpleNamespace
    import infernux.lib as native_module
    import infernux.engine.runtime_input_state as state_module
    from infernux.engine.runtime_input_state import world_pointer_state

    native = SimpleNamespace(_read_world_id=7, frame_index=101,
                             _read_owner=SimpleNamespace(retired=False))
    provider = SimpleNamespace(instance=lambda: native)
    monkeypatch.setattr(native_module, "InputManager", provider)
    monkeypatch.setattr(state_module, "InputManager", provider)
    root, canvas = make_canvas(scene)
    control = make_control(scene, root, TrackedButton)
    host_processor = UIEventProcessor()
    view = SimpleNamespace(_ui_event_processor=host_processor)
    host_callbacks, clicks = [], []
    runtime_event_queue.enqueue(lambda: host_callbacks.append("host"))

    def clicked():
        assert view._active_pointer_world == 7
        assert view._ui_event_processor is processor
        assert native.frame_index == 102
        clicks.append("click")
    control.on_click.add_listener(clicked)
    factories = {"_ui_event_processor": UIEventProcessor}
    with world_pointer_state(view, factories):
        processor = view._ui_event_processor
        processor.process_pointers([canvas], [pointer(1, down=True)], 0.016)
        assert control.events == [], "callback ran before pointer traversal returned"
    # This is the logical Fixed boundary: native scope is still F, view is Host.
    assert any(event[0] == "down" for event in control.events)
    assert view._ui_event_processor is host_processor
    assert host_callbacks == []
    native.frame_index = 102
    with world_pointer_state(view, factories):
        before = len(control.events)
        processor.process_pointers([canvas], [pointer(1, up=not cancel, canceled=cancel)], 0.016)
        assert len(control.events) == before
    assert ("up", 1, PointerButton.Left, cancel) in control.events
    assert clicks == ([] if cancel else ["click"])
    assert host_callbacks == []
    runtime_event_queue.drain()
    assert host_callbacks == ["host"]


def test_logical_ui_drag_and_terminal_cancel_apply_before_fixed(scene, monkeypatch):
    from types import SimpleNamespace
    import infernux.lib as native_module
    from infernux.ui.ui_event_system import logical_ui_callback_batch

    native = SimpleNamespace(_read_world_id=11, frame_index=201,
                             _read_owner=SimpleNamespace(retired=False))
    monkeypatch.setattr(native_module, "InputManager", SimpleNamespace(instance=lambda: native))
    root, canvas = make_canvas(scene)
    slider = make_control(scene, root, TrackedSlider)
    processor = UIEventProcessor()
    samples = []
    slider.on_value_changed.add_listener(lambda value: samples.append((native._read_world_id, native.frame_index, value)))
    for index, frame in enumerate((pointer(1, 20, down=True), pointer(1, 170, held=True),
                                   pointer(1, 170, canceled=True)), 201):
        native.frame_index = index
        before = len(samples)
        with logical_ui_callback_batch():
            processor.process_pointers([canvas], [frame], 0.016)
            assert len(samples) == before
        # Fixed reads the value written by the actual UI callback this F.
        if index == 202:
            assert slider.value == pytest.approx(0.85)
    assert samples and all(world == 11 for world, _frame, _value in samples)
    assert ("up", True) in slider.terminals
    assert ("end", True) in slider.terminals
    assert runtime_event_queue.drain() == 0
