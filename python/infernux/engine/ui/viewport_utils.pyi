"""viewport_utils — viewport bounding-box helper for Scene / Game views."""

from __future__ import annotations

from typing import Tuple

from infernux.lib import InxGUIContext


class ViewportInfo:
    """Axis-aligned bounding box of the viewport image region.

    Usage::

        info = capture_viewport_info(ctx)
        if info.is_hovered:
            mx, my = info.mouse_local(ctx)
    """

    __slots__ = (
        "image_min_x",
        "image_min_y",
        "image_max_x",
        "image_max_y",
        "is_hovered",
    )

    image_min_x: float
    image_min_y: float
    image_max_x: float
    image_max_y: float
    is_hovered: bool

    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...

    def mouse_local(self, ctx: InxGUIContext) -> Tuple[float, float]:
        """Return ``(x, y)`` of the mouse in local viewport coordinates."""
        ...

    def is_mouse_inside(self, ctx: InxGUIContext) -> bool: ...


def capture_viewport_info(ctx: InxGUIContext) -> ViewportInfo:
    """Capture the current viewport bounds from ImGui draw state."""
    ...
