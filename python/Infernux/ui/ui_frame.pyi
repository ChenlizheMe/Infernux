from .enums import UILayoutAlign, UILayoutDirection, UILayoutJustify
from .inx_ui_screen_component import InxUIScreenComponent

class UIFrame(InxUIScreenComponent):
    layout_direction: UILayoutDirection
    gap: float
    padding_left: float
    padding_right: float
    padding_top: float
    padding_bottom: float
    align_items: UILayoutAlign
    justify_content: UILayoutJustify
    clip_content: bool
