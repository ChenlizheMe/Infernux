"""Live UI eligibility applies to captured and queued pointer transactions."""
import pytest

from infernux.engine import runtime_event_queue
from infernux.ui import UIButton, UICanvas, UIGroup, UISlider, UIPointerFrame, PointerType
from infernux.ui.ui_event_system import UIEventProcessor


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


MODES = ("enabled", "owner_active", "parent_active", "group", "interactable",
         "canvas_enabled", "canvas_removed", "reparent")


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


@pytest.mark.parametrize("mode", MODES)
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
    processor.process_pointers([canvas], [pointer(1, 40, down=True), pointer(2, 290, down=True)], .016)
    runtime_event_queue.drain()
    processor.process_pointers([canvas], [pointer(1, 40, up=True), pointer(2, 290, up=True)], .016)
    runtime_event_queue.drain()
    assert clicks == ["first"]
    assert not processor._pointers


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("timing", ["before_frame", "before_drain"])
@pytest.mark.parametrize("pointer_type", [PointerType.Mouse, PointerType.Touch])
def test_disabled_capture_cancels_once_without_late_value_changes(scene, mode, timing, pointer_type):
    root, canvas = make_canvas(scene)
    slider = make_control(scene, root, TrackedSlider)
    changes = []
    slider.on_value_changed.add_listener(changes.append)
    processor = UIEventProcessor()
    processor.process_pointers([canvas], [pointer(1, 20, pointer_type, down=True)], .016)
    runtime_event_queue.drain()
    processor.process_pointers([canvas], [pointer(1, 80, pointer_type, held=True)], .016)
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
    processor.process_pointers(surfaces, [sample], .016)
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
    processor.process_pointers([canvas], [pointer(1, down=True, up=True, canceled=mode == "canceled")], .016)
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
                        [canvas], [pointer(1, 40, **flags), pointer(2, 290, **flags)], .016,
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
