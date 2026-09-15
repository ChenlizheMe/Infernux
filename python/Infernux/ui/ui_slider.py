"""Pointer-driven scalar slider built on the shared UI event transaction."""

from __future__ import annotations

import math

from Infernux.components import add_component_menu, serialized_field
from Infernux.components.fields import FieldType

from .enums import UIFillDirection
from .ui_event import UIEvent1
from .ui_selectable import UISelectable


@add_component_menu("UI/Slider")
class UISlider(UISelectable):
    minimum: float = serialized_field(default=0.0, group="Value")
    maximum: float = serialized_field(default=1.0, group="Value")
    value: float = serialized_field(default=0.0, group="Value")
    whole_numbers: bool = serialized_field(default=False, group="Value")
    fill_direction: UIFillDirection = serialized_field(
        default=UIFillDirection.LeftToRight, group="Value"
    )
    background_color: list = serialized_field(
        default=[0.12, 0.13, 0.16, 1.0], field_type=FieldType.COLOR,
        hdr=True, group="Appearance",
    )
    fill_color: list = serialized_field(
        default=[0.922, 0.341, 0.341, 1.0], field_type=FieldType.COLOR,
        hdr=True, group="Appearance",
    )
    handle_color: list = serialized_field(
        default=[0.96, 0.96, 0.97, 1.0], field_type=FieldType.COLOR,
        hdr=True, group="Appearance",
    )
    handle_size: float = serialized_field(
        default=16.0, range=(1.0, 256.0), group="Appearance"
    )

    def awake(self):
        super().awake()
        self._ensure_slider_state()

    def _ensure_slider_state(self):
        if not hasattr(self, "_on_value_changed"):
            self._on_value_changed = UIEvent1()

    @property
    def on_value_changed(self) -> UIEvent1:
        self._ensure_slider_state()
        return self._on_value_changed

    @property
    def normalized_value(self) -> float:
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if maximum <= minimum:
            return 0.0
        return max(0.0, min(1.0, (float(self.value) - minimum) / (maximum - minimum)))

    def set_value(self, value: float, notify: bool = True) -> float:
        minimum = float(self.minimum)
        maximum = max(minimum, float(self.maximum))
        value = max(minimum, min(maximum, float(value)))
        if self.whole_numbers:
            value = float(round(value))
        changed = value != float(self.value)
        self.value = value
        if changed and notify:
            self.on_value_changed.invoke(value)
        return value

    def set_value_without_notify(self, value: float) -> float:
        """Set the clamped value without invoking ``on_value_changed``."""
        return self.set_value(value, notify=False)

    def _set_from_pointer(self, event_data) -> None:
        if event_data.canvas is None:
            return
        width, height = event_data.canvas_size
        width = float(width)
        height = float(height)
        local_rect = getattr(event_data.canvas, "element_rect", None)
        if callable(local_rect):
            x, y, rect_width, rect_height = local_rect(self)
            rotation = 0.0
        else:
            x, y, rect_width, rect_height = self.get_rect(width, height)
            rotation = float(self.get_layout_rotation())
        if rect_width <= 0.0 or rect_height <= 0.0:
            return
        px, py = event_data.position
        center_x = x + rect_width * 0.5
        center_y = y + rect_height * 0.5
        radians = math.radians(-rotation)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        dx = float(px) - center_x
        dy = float(py) - center_y
        local_x = dx * cosine - dy * sine + rect_width * 0.5
        local_y = dx * sine + dy * cosine + rect_height * 0.5
        if self.fill_direction == UIFillDirection.LeftToRight:
            normalized = local_x / rect_width
        elif self.fill_direction == UIFillDirection.RightToLeft:
            normalized = 1.0 - local_x / rect_width
        elif self.fill_direction == UIFillDirection.BottomToTop:
            normalized = 1.0 - local_y / rect_height
        else:
            normalized = local_y / rect_height
        normalized = max(0.0, min(1.0, normalized))
        self.set_value(float(self.minimum) + normalized * (float(self.maximum) - float(self.minimum)))

    def on_pointer_down(self, event_data):
        super().on_pointer_down(event_data)
        if self.interactable:
            self._set_from_pointer(event_data)

    def on_begin_drag(self, event_data):
        if self.interactable:
            self._set_from_pointer(event_data)

    def on_drag(self, event_data):
        if self.interactable:
            self._set_from_pointer(event_data)
