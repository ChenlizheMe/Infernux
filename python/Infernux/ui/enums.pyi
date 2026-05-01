"""Type stubs for Infernux.ui.enums — UI system enumerations."""

from __future__ import annotations

from enum import IntEnum


class RenderMode(IntEnum):
    """How a UICanvas renders its content."""
    ScreenOverlay = 0
    """Rendered on top of everything (screen-space)."""
    CameraOverlay = 1
    """Rendered on top of a specific camera's output."""


class ScreenAlignH(IntEnum):
    """Horizontal anchor for screen-space UI layout."""
    Left = 0
    Center = 1
    Right = 2


class ScreenAlignV(IntEnum):
    """Vertical anchor for screen-space UI layout."""
    Top = 0
    Center = 1
    Bottom = 2


class TextAlignH(IntEnum):
    """Horizontal text alignment (Figma-style)."""
    Left = 0
    Center = 1
    Right = 2


class TextAlignV(IntEnum):
    """Vertical text alignment (Figma-style)."""
    Top = 0
    Center = 1
    Bottom = 2


class TextOverflow(IntEnum):
    """How text overflows its bounding box."""
    Visible = 0
    """Draw beyond box."""
    Clip = 1
    """Clip at box edge (visual only)."""
    Truncate = 2
    """Add ellipsis when text is too long."""


class TextResizeMode(IntEnum):
    """How a UIText determines its clipping box size."""
    AutoWidth = 0
    """Width adjusts to fit content; height fixed."""
    AutoHeight = 1
    """Height adjusts to fit content; width fixed."""
    FixedSize = 2
    """Both width and height are fixed."""


class UIScaleMode(IntEnum):
    """How a UICanvas scales UI elements, aligned with Unity CanvasScaler."""
    ConstantPixelSize = 0
    """UI elements keep their design pixel size."""
    ScaleWithScreenSize = 1
    """Scale elements from reference resolution to current screen size."""
    ConstantPhysicalSize = 2
    """Future DPI-aware physical sizing; currently behaves like constant pixels."""


class ScreenMatchMode(IntEnum):
    """How ScaleWithScreenSize chooses width/height scale."""
    MatchWidthOrHeight = 0
    """Blend between width and height using match_width_or_height."""
    Expand = 1
    """Scale so the canvas expands and nothing is cropped."""
    Shrink = 2
    """Scale so the canvas shrinks and everything fits."""


class UITransitionType(IntEnum):
    """Visual transition type for interactive elements (UISelectable)."""
    ColorTint = 0
    """Tint the target graphic's color."""
    SpriteSwap = 1
    """Swap the target image sprite (future)."""
    Animation = 2
    """Trigger an animator state (future)."""
    None_ = 3
    """No visual feedback."""
