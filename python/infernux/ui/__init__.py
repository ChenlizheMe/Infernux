"""Infernux UI module — screen-space UI components."""

from .enums import RenderMode, ScreenAlignH, ScreenAlignV, TextAlignH, TextAlignV, TextOverflow, TextResizeMode, UITransitionType, UIScaleMode, ScreenMatchMode, UILayoutDirection, UILayoutSizing, UILayoutPosition, UILayoutAlign, UILayoutJustify, UIFillDirection
from .inx_ui_component import InxUIComponent
from .inx_ui_screen_component import InxUIScreenComponent
from .ui_canvas import UICanvas
from .ui_frame import UIFrame
from .ui_group import UIGroup
from .ui_progress_bar import UIProgressBar
from .ui_slider import UISlider
from .ui_text import UIText
from .ui_image import UIImage
# Unity-compatible name: RawImage is the same managed texture/RenderTexture
# control in Infernux; keep one component and one renderer.
UIRawImage = UIImage
from .ui_selectable import UISelectable
from .ui_button import UIButton
from .ui_event_data import PointerEventData, PointerButton, PointerType
from .ui_event import UIEvent, UIEvent1
from .ui_event_system import UIEventProcessor, UIPointerFrame
from .ui_texture_cache import UITextureCache, get_shared_cache
from .ui_render_dispatch import register_ui_renderer, dispatch as ui_dispatch

__all__ = [
    "RenderMode",
    "ScreenAlignH",
    "ScreenAlignV",
    "TextAlignH",
    "TextAlignV",
    "TextOverflow",
    "TextResizeMode",
    "UITransitionType",
    "UIScaleMode",
    "ScreenMatchMode",
    "UILayoutDirection",
    "UILayoutSizing",
    "UILayoutPosition",
    "UILayoutAlign",
    "UILayoutJustify",
    "UIFillDirection",
    "InxUIComponent",
    "InxUIScreenComponent",
    "UICanvas",
    "UIFrame",
    "UIGroup",
    "UIProgressBar",
    "UISlider",
    "UIText",
    "UIImage",
    "UIRawImage",
    "UISelectable",
    "UIButton",
    "PointerEventData",
    "PointerButton",
    "PointerType",
    "UIEvent",
    "UIEvent1",
    "UIEventProcessor",
    "UIPointerFrame",
    "UITextureCache",
    "get_shared_cache",
    "register_ui_renderer",
    "ui_dispatch",
]
