from types import SimpleNamespace

import pytest

from Infernux.ui import UICanvas, UIFillDirection, UIProgressBar, UISlider
from Infernux.ui.ui_render_dispatch import dispatch


def _pointer(x, y):
    return SimpleNamespace(
        position=(x, y),
        canvas=SimpleNamespace(),
        canvas_size=(1920.0, 1080.0),
    )


def test_progress_bar_clamps_value_and_normalizes_without_fallback_range():
    progress = UIProgressBar()
    progress.minimum = 10.0
    progress.maximum = 20.0

    assert progress.set_value(15.0) == 15.0
    assert progress.normalized_value == pytest.approx(0.5)
    assert progress.set_value(100.0) == 20.0

    progress.maximum = 5.0
    assert progress.normalized_value == 0.0


def test_progress_bar_value_event_and_silent_set_are_unified():
    progress = UIProgressBar()
    progress.minimum = 10.0
    progress.maximum = 20.0
    values = []
    progress.on_value_changed.add_listener(values.append)

    assert progress.set_value(15.0) == 15.0
    assert progress.set_value(100.0) == 20.0
    assert progress.set_value_without_notify(12.0) == 12.0
    assert progress.set_value_without_notify(12.0) == 12.0
    assert values == pytest.approx([15.0, 20.0])


def _screen_slider(scene):
    root = scene.create_game_object('Slider Canvas')
    root.add_py_component(UICanvas())
    child = scene.create_game_object('Slider')
    child.set_parent(root)
    slider = UISlider()
    child.add_py_component(slider)
    return slider


def test_slider_pointer_updates_value_and_emits_only_real_changes(scene):
    slider = _screen_slider(scene)
    slider.width = 100.0
    slider.height = 40.0
    slider.x = 10.0
    slider.y = 0.0
    slider.minimum = 0.0
    slider.maximum = 10.0
    values = []
    slider.on_value_changed.add_listener(values.append)

    slider.on_pointer_down(_pointer(60.0, 20.0))
    slider.on_drag(_pointer(110.0, 20.0))
    slider.on_drag(_pointer(110.0, 20.0))

    assert slider.value == pytest.approx(10.0)
    assert values == pytest.approx([5.0, 10.0])


def test_slider_silent_set_preserves_clamping_without_event():
    slider = UISlider()
    slider.minimum = -1.0
    slider.maximum = 1.0
    values = []
    slider.on_value_changed.add_listener(values.append)

    assert slider.set_value_without_notify(4.0) == 1.0
    assert slider.value == 1.0
    assert values == []


def test_legacy_ui_layout_fields_migrate_to_the_attached_transform(scene):
    root = scene.create_game_object("Legacy Canvas")
    root.add_py_component(UICanvas())
    obj = scene.create_game_object("Legacy Label")
    obj.set_parent(root)
    text = UISlider()
    obj.add_py_component(text)
    text._deserialize_fields_document({
        "x": 120.0,
        "y": 80.0,
        "rotation": 25.0,
        "width": 100.0,
        "height": 40.0,
    })

    position = obj.transform.local_position
    angles = obj.transform.local_euler_angles
    rect_x, rect_y, rect_w, rect_h = text.get_rect(1920.0, 1080.0)
    assert position.x == pytest.approx(rect_x + rect_w * 0.5 - 960.0)
    assert position.y == pytest.approx(-(rect_y + rect_h * 0.5 - 540.0))
    assert angles.z == pytest.approx(25.0)


def test_slider_pointer_mapping_respects_rotation_and_fill_direction(scene):
    slider = _screen_slider(scene)
    slider.width = 100.0
    slider.height = 40.0
    slider.x = 10.0
    slider.y = 0.0
    slider.rotation = 90.0
    slider.minimum = 0.0
    slider.maximum = 1.0

    slider.on_pointer_down(_pointer(60.0, 70.0))
    assert slider.value == pytest.approx(1.0)

    slider.fill_direction = UIFillDirection.RightToLeft
    slider.on_pointer_down(_pointer(60.0, 70.0))
    assert slider.value == pytest.approx(0.0)


class _Renderer:
    def __init__(self):
        self.rects = []

    def add_filled_rect(self, *args):
        self.rects.append(args)


def test_value_controls_submit_managed_runtime_geometry():
    renderer = _Renderer()
    progress = UIProgressBar()
    progress.value = 0.25
    assert dispatch(
        progress, "runtime", renderer=renderer, ui_list=1,
        sx=10.0, sy=20.0, sw=200.0, sh=20.0,
        scale_x=1.0, scale_y=1.0,
    )
    assert len(renderer.rects) == 2
    assert renderer.rects[1][3] == pytest.approx(60.0)

    slider = UISlider()
    slider.value = 0.5
    assert dispatch(
        slider, "runtime", renderer=renderer, ui_list=1,
        sx=10.0, sy=20.0, sw=200.0, sh=20.0,
        scale_x=1.0, scale_y=1.0,
    )
    assert len(renderer.rects) == 5
