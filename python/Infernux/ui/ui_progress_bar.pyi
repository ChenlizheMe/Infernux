from Infernux.ui.enums import UIFillDirection
from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent
from Infernux.ui.ui_event import UIEvent1

class UIProgressBar(InxUIScreenComponent):
    minimum: float
    maximum: float
    value: float
    fill_direction: UIFillDirection
    background_color: list[float]
    fill_color: list[float]
    @property
    def on_value_changed(self) -> UIEvent1: ...
    @property
    def normalized_value(self) -> float: ...
    def set_value(self, value: float, notify: bool = True) -> float: ...
    def set_value_without_notify(self, value: float) -> float: ...
