"""
Inspector-only layout and color tokens for the editor.

``Theme`` subclasses :class:`InspectorThemeBase` so ``Theme.INSPECTOR_*`` and
related names keep working. Edit values here to tune the Inspector in isolation
from the rest of the editor chrome.
"""

from __future__ import annotations

from typing import Tuple

RGBA = Tuple[float, float, float, float]


class InspectorThemeBase:
    """Layout and colors for the Inspector, component headers, and list bodies."""

    # -- Layout sizes
    INSPECTOR_INIT_SIZE = (300, 500)
    INSPECTOR_MIN_PROPS_H = 100
    INSPECTOR_MIN_RAWDATA_H = 100
    INSPECTOR_SPLITTER_H = 8
    INSPECTOR_DEFAULT_RATIO = 0.4
    INSPECTOR_LABEL_PAD = 18.0
    INSPECTOR_MIN_LABEL_WIDTH = 156.0
    INSPECTOR_FRAME_PAD = (4.0, 2.0)
    OBJECT_FIELD_TEXT_INSET_X: float = 12.0
    # Whole object-reference row shifted right (ImGui "two cells" ~ 2× default spacing)
    OBJECT_FIELD_LEADING_INDENT: float = 16.0
    INSPECTOR_ITEM_SPC = (4.0, 2.0)
    INSPECTOR_SUBITEM_SPC = (4.0, 2.0)
    INSPECTOR_SECTION_GAP = 6.0
    INSPECTOR_TITLE_GAP = 10.0

    # -- Component header
    INSPECTOR_HEADER_PRIMARY_FRAME_PAD = (4.0, 2.0)
    INSPECTOR_HEADER_SECONDARY_FRAME_PAD = (4.0, 2.0)
    INSPECTOR_HEADER_LIST_FRAME_PAD = (4.0, 2.0)
    INSPECTOR_HEADER_PRIMARY_FONT_SCALE = 1.0
    INSPECTOR_HEADER_SECONDARY_FONT_SCALE = 1.0
    INSPECTOR_HEADER_LIST_FONT_SCALE = 1.0
    INSPECTOR_HEADER_ITEM_SPC = (4.0, 2.0)
    INSPECTOR_HEADER_BORDER_SIZE = 0.0
    INSPECTOR_ACTION_ALIGN_X = 0.0
    INSPECTOR_HEADER_CONTENT_INDENT = 28.0
    ADD_COMP_SEARCH_W = 240
    COMPONENT_ICON_SIZE = 16
    COMP_ENABLED_CB_OFFSET = 40

    # -- Checkbox
    INSPECTOR_CHECKBOX_FONT_SCALE = 1.0
    INSPECTOR_CHECKBOX_FRAME_PAD = (4.0, 2.0)
    INSPECTOR_CHECKBOX_SLOT_W = 22.0

    # -- Header colors
    INSPECTOR_HEADER_PRIMARY: RGBA = (0.235, 0.235, 0.235, 1.0)
    INSPECTOR_HEADER_PRIMARY_HOVERED: RGBA = (0.28, 0.24, 0.24, 1.0)
    INSPECTOR_HEADER_PRIMARY_ACTIVE: RGBA = (0.32, 0.25, 0.25, 1.0)
    INSPECTOR_HEADER_SECONDARY: RGBA = (0.18, 0.18, 0.18, 1.0)
    INSPECTOR_HEADER_SECONDARY_HOVERED: RGBA = (0.22, 0.20, 0.20, 1.0)
    INSPECTOR_HEADER_SECONDARY_ACTIVE: RGBA = (0.26, 0.22, 0.22, 1.0)
    INSPECTOR_HEADER_LIST: RGBA = (0.16, 0.16, 0.16, 1.0)
    INSPECTOR_HEADER_LIST_HOVERED: RGBA = (0.20, 0.18, 0.18, 1.0)
    INSPECTOR_HEADER_LIST_ACTIVE: RGBA = (0.24, 0.20, 0.20, 1.0)

    # -- Inline buttons
    INSPECTOR_INLINE_BTN_IDLE: RGBA = (0.20, 0.20, 0.20, 1.0)
    INSPECTOR_INLINE_BTN_HOVER: RGBA = (0.28, 0.24, 0.24, 1.0)
    INSPECTOR_INLINE_BTN_ACTIVE: RGBA = (0.922, 0.341, 0.341, 1.0)
    INSPECTOR_INLINE_BTN_ON: RGBA = (0.80, 0.30, 0.30, 1.0)
    INSPECTOR_INLINE_BTN_GAP: float = 4.0
    INSPECTOR_INLINE_BTN_H: float = 0.0

    # -- List body
    INSPECTOR_LIST_BODY_BG: RGBA = (0.10, 0.10, 0.10, 0.82)
    INSPECTOR_LIST_BODY_BG_HOVERED: RGBA = (0.20, 0.20, 0.22, 0.92)
    INSPECTOR_LIST_BODY_BORDER: RGBA = (0.22, 0.22, 0.22, 1.0)
    INSPECTOR_LIST_BODY_ROUNDING: float = 0.0
    INSPECTOR_LIST_BODY_PAD_X: float = 4.0
    INSPECTOR_LIST_BODY_PAD_Y: float = 2.0
    INSPECTOR_SMALL_ICON_BTN_FRAME_PAD: tuple = (4.0, 2.0)

    COLOR_SWATCH_BORDER: RGBA = (0.4, 0.4, 0.4, 1.0)
