from Infernux.ui.enums import UIFillDirection
from Infernux.ui.ui_event import UIEvent1
from Infernux.ui.ui_selectable import UISelectable

class UISlider(UISelectable):
    minimum: float
    maximum: float
    value: float
    whole_numbers: bool
    fill_direction: UIFillDirection
    background_color: list[float]
    fill_color: list[float]
    handle_color: list[float]
    handle_size: float
    @property
    def on_value_changed(self) -> UIEvent1: ...
    @property
    def normalized_value(self) -> float: ...
    def set_value(self, value: float, notify: bool = True) -> float: ...
    def set_value_without_notify(self, value: float) -> float: ...
