from .inx_ui_component import InxUIComponent


class UILayoutElement(InxUIComponent):
    ignore_layout: bool
    min_width: float
    min_height: float
    preferred_width: float
    preferred_height: float
    flexible_width: float
    flexible_height: float
    layout_priority: int

    def calculate_layout_input_horizontal(self) -> None: ...
    def calculate_layout_input_vertical(self) -> None: ...
