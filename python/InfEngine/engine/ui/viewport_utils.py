"""
viewport_utils — Shared helpers for viewport coordinate conversion.

Both SceneViewPanel and GameViewPanel render an off-screen texture into an
ImGui ``ctx.image()`` widget.  After the image is drawn, they need the same
information:

* The image's screen-space bounding rect (min/max)
* Whether the image is hovered
* A helper to convert absolute mouse position to viewport-local coordinates

This module provides ``ViewportInfo`` and ``capture_viewport_info()`` to
eliminate that duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from InfEngine.lib import InfGUIContext


@dataclass(slots=True)
class ViewportInfo:
    """Snapshot of a viewport's screen-space state after ``ctx.image()``."""

    # Screen-space bounding rect of the rendered image
    image_min_x: float = 0.0
    image_min_y: float = 0.0
    image_max_x: float = 0.0
    image_max_y: float = 0.0

    # Whether the image widget is hovered
    is_hovered: bool = False

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def width(self) -> float:
        return self.image_max_x - self.image_min_x

    @property
    def height(self) -> float:
        return self.image_max_y - self.image_min_y

    def mouse_local(self, ctx: InfGUIContext) -> Tuple[float, float]:
        """Return mouse position relative to the image top-left corner.

        Coordinates may be negative or exceed ``(width, height)`` when the
        cursor is outside the viewport.
        """
        return (
            ctx.get_mouse_pos_x() - self.image_min_x,
            ctx.get_mouse_pos_y() - self.image_min_y,
        )

    def is_mouse_inside(self, ctx: InfGUIContext) -> bool:
        """Return ``True`` if the mouse is within the image bounds."""
        lx, ly = self.mouse_local(ctx)
        return 0 <= lx <= self.width and 0 <= ly <= self.height


def capture_viewport_info(ctx: InfGUIContext) -> ViewportInfo:
    """Capture viewport metrics immediately after a ``ctx.image()`` call.

    Must be called right after ``ctx.image(...)`` so that
    ``get_item_rect_*`` and ``is_item_hovered`` refer to that image widget.

    Returns:
        A ``ViewportInfo`` populated with the image's screen rect and hover
        state.
    """
    return ViewportInfo(
        image_min_x=ctx.get_item_rect_min_x(),
        image_min_y=ctx.get_item_rect_min_y(),
        image_max_x=ctx.get_item_rect_max_x(),
        image_max_y=ctx.get_item_rect_max_y(),
        is_hovered=ctx.is_item_hovered(),
    )
