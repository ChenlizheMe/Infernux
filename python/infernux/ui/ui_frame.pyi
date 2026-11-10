from .enums import UILayoutAlign, UILayoutDirection, UILayoutJustify
from .inx_ui_screen_component import InxUIScreenComponent

class UIFrame(InxUIScreenComponent):
    def rebuild_layout(self, canvas_width: float | None = None, canvas_height: float | None = None) -> bool: ...
    layout_direction: UILayoutDirection
    gap: float
    padding_left: float
    padding_right: float
    padding_top: float
    padding_bottom: float
    align_items: UILayoutAlign
    justify_content: UILayoutJustify
    clip_content: bool
