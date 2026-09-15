"""Progress display using the shared UI rectangle and render path."""

from __future__ import annotations

from Infernux.components import add_component_menu, serialized_field
from Infernux.components.fields import FieldType

from .enums import UIFillDirection
from .inx_ui_screen_component import InxUIScreenComponent
from .ui_event import UIEvent1


@add_component_menu("UI/Progress Bar")
class UIProgressBar(InxUIScreenComponent):
    minimum: float = serialized_field(default=0.0, group="Value")
    maximum: float = serialized_field(default=1.0, group="Value")
    value: float = serialized_field(default=0.0, group="Value")
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
    raycast_target: bool = serialized_field(default=False, group="Interaction")

    def awake(self):
        super().awake()
        self._ensure_progress_state()

    def _ensure_progress_state(self):
        if not hasattr(self, "_on_value_changed"):
            self._on_value_changed = UIEvent1()

    @property
    def on_value_changed(self) -> UIEvent1:
        """Invoked with the clamped value after a real value change."""
        self._ensure_progress_state()
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
        next_value = max(minimum, min(maximum, float(value)))
        changed = next_value != float(self.value)
        self.value = next_value
        if changed and notify:
            self.on_value_changed.invoke(next_value)
        return next_value

    def set_value_without_notify(self, value: float) -> float:
        """Set the clamped value without invoking ``on_value_changed``."""
        return self.set_value(value, notify=False)
