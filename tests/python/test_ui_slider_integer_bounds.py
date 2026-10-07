"""A whole-number slider must publish only integers inside its authoring range."""
import math

import pytest

from infernux.ui import PointerEventData, UICanvas, UIFillDirection, UISlider


@pytest.mark.parametrize("minimum,maximum,requested,expected", [
    (0.25, 2.75, -10, 1),
    (0.25, 2.75, 10, 2),
    (0.25, 2.75, 1.4, 1),
    (0.25, 2.75, 1.5, 2),
    (-3.75, -1.25, -10, -3),
    (-3.75, -1.25, 10, -2),
    (-3.75, -1.25, -2.5, -2),
    (-0.2, 0.2, 10, 0),
    (2, 2, -10, 2),
    (0, 4, 2.5, 2),  # Retain the existing ties-to-even rounding rule.
])
@pytest.mark.parametrize("notify", [True, False])
def test_integer_slider_return_storage_and_notifications_share_valid_bounds(
    minimum, maximum, requested, expected, notify,
):
    slider = UISlider()
    slider.minimum, slider.maximum, slider.whole_numbers = minimum, maximum, True
    before = slider.value
    observed = []
    slider.on_value_changed.add_listener(observed.append)
    setter = slider.set_value if notify else slider.set_value_without_notify
    assert setter(requested) == expected
    assert slider.value == expected
    assert slider.value.is_integer() and minimum <= slider.value <= maximum
    assert setter(requested) == expected
    assert observed == ([expected] if notify and before != expected else [])


@pytest.mark.parametrize("minimum,maximum,requested", [
    (0.25, 0.75, 0.5),
    (-0.75, -0.25, -0.5),
    (2, 1, 1),
    (0.5, 0.5, 0.5),
    (math.nan, 2, 1),
    (0, math.nan, 1),
    (-math.inf, 2, 1),
    (0, math.inf, 1),
    (0, 2, math.nan),
    (0, 2, math.inf),
    (0, 2, -math.inf),
])
@pytest.mark.parametrize("notify", [True, False])
def test_invalid_integer_slider_update_does_not_mutate_or_publish(minimum, maximum, requested, notify):
    slider = UISlider()
    slider.value = 1.0
    slider.minimum, slider.maximum, slider.whole_numbers = minimum, maximum, True
    observed = []
    slider.on_value_changed.add_listener(observed.append)
    with pytest.raises(ValueError, match="integer|finite"):
        slider.set_value(requested, notify=notify)
    assert slider.value == 1.0
    assert observed == []


@pytest.mark.parametrize("direction", list(UIFillDirection))
def test_pointer_drag_clamps_to_available_integers_in_all_directions(scene, direction):
    root = scene.create_game_object("Integer canvas")
    canvas = UICanvas()
    root.add_py_component(canvas)
    owner = scene.create_game_object("Integer slider")
    owner.set_parent(root)
    slider = UISlider()
    owner.add_py_component(slider)
    slider.minimum, slider.maximum, slider.whole_numbers = 0.25, 2.75, True
    slider.fill_direction = direction
    slider.set_rect(10, 10, 100, 100, 1920, 1080)
    event = PointerEventData()
    event.canvas, event.canvas_size, event.target = canvas, (1920, 1080), slider
    observed = []
    slider.on_value_changed.add_listener(observed.append)
    for position in ((-100, -100), (300, 300)):
        event.position = position
        slider.on_drag(event)
        assert slider.value in (1.0, 2.0)
    assert set(observed) == {1.0, 2.0}
    restored = UISlider()
    restored._deserialize_fields_document(slider._serialize_fields_document())
    assert (restored.minimum, restored.maximum, restored.whole_numbers, restored.value) == (
        0.25, 2.75, True, slider.value,
    )
