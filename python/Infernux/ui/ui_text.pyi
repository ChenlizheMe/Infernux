"""Type stubs for Infernux.ui.ui_text — text label UI element."""

from __future__ import annotations

from typing import Callable, Tuple

from Infernux.core.asset_types import FontAssetInfo

from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent
from Infernux.ui.enums import TextAlignH, TextAlignV, TextOverflow, TextResizeMode


class UIText(InxUIScreenComponent):
    """Figma-style text label rendered with ImGui draw primitives.

    Inherits ``width`` and ``height`` from ``InxUIScreenComponent``; position
    and rotation come from the GameObject Transform.

    Attributes:
        text: Display string.
        font: Optional imported Font asset (``.ttf`` / ``.otf``), persisted by GUID.
        fallback_fonts: Ordered fallback Font assets, persisted by GUID.
        font_size: Font size in canvas pixels.
        line_height: Line height multiplier.
        letter_spacing: Extra letter spacing in pixels.
        text_align_h: Horizontal text alignment.
        text_align_v: Vertical text alignment.
        overflow: Text overflow mode.
        resize_mode: How the clipping box resizes with content.
        color: Text color as ``[R, G, B, A]`` (0–1 each).

    Example::

        text = game_object.add_component(UIText)
        text.text = "Hello World"
        text.font_size = 32.0
        text.color = [1.0, 0.8, 0.0, 1.0]
    """

    text: str
    font: FontAssetInfo | None
    fallback_fonts: list[FontAssetInfo | None]
    font_size: float
    line_height: float
    letter_spacing: float
    text_align_h: TextAlignH
    text_align_v: TextAlignV
    overflow: TextOverflow
    resize_mode: TextResizeMode
    color: list

    def is_auto_width(self) -> bool:
        """Return ``True`` if resize mode is ``AutoWidth``."""
        ...

    def is_auto_height(self) -> bool:
        """Return ``True`` if resize mode is ``AutoHeight``."""
        ...

    def is_fixed_size(self) -> bool:
        """Return ``True`` if resize mode is ``FixedSize``."""
        ...

    def get_wrap_width(self) -> float:
        """Return the wrap width for text layout (0 = no wrap)."""
        ...

    def resolve_text_layout(
        self,
        measure_text: Callable[[str, float, float, str, float, float], Tuple[float, float]],
        scale: float = ...,
    ) -> bool:
        """Resolve the transient intrinsic size in logical canvas pixels."""
        ...

    def get_resolved_size(self) -> Tuple[float, float]:
        """Return the effective box shared by layout, drawing, and input."""
        ...

    def is_width_editable(self) -> bool:
        """Return ``True`` if width can be manually edited (not AutoWidth)."""
        ...

    def is_height_editable(self) -> bool:
        """Return ``True`` if height can be manually edited (not AutoHeight)."""
        ...
