"""Type stubs for Infernux.ui.inx_ui_screen_component — 2D screen-space UI element base."""

from __future__ import annotations

from typing import List, Optional, Tuple

from Infernux.ui.inx_ui_component import InxUIComponent
from Infernux.ui.enums import ScreenAlignH, ScreenAlignV
from Infernux.ui.ui_event_data import PointerEventData


def clear_rect_cache(frame_id: int = ...) -> None:
    """Call once per frame before any ``get_rect()`` usage to clear per-frame cache."""
    ...


class InxUIScreenComponent(InxUIComponent):
    """Base class for 2D screen-space UI elements with a canvas-pixel rectangle.

    Provides anchor-aware position, size, and appearance data.  All concrete
    UI widgets (``UIText``, ``UIImage``, ``UIButton``) inherit from this.

    Attributes:
        align_h: Horizontal anchor within parent (Left / Center / Right).
        align_v: Vertical anchor within parent (Top / Center / Bottom).
        x: Horizontal offset from anchor in canvas pixels.
        y: Vertical offset from anchor in canvas pixels.
        width: Width in canvas pixels (unrotated content size).
        height: Height in canvas pixels (unrotated content size).
        rotation: Visual rotation in degrees (any angle).
        mirror_x: Mirror element horizontally.
        mirror_y: Mirror element vertically.
        lock_aspect_ratio: Preserve width/height ratio while resizing.
        opacity: Element opacity (0.0–1.0).
        corner_radius: Corner rounding in canvas pixels.
        raycast_target: Whether this element receives pointer events.
    """

    _hide_transform_: bool

    align_h: ScreenAlignH
    align_v: ScreenAlignV
    x: float
    y: float
    rotation: float
    mirror_x: bool
    mirror_y: bool
    width: float
    height: float
    lock_aspect_ratio: bool
    opacity: float
    corner_radius: float
    raycast_target: bool

    # ------------------------------------------------------------------
    # Rect computation
    # ------------------------------------------------------------------

    def get_rect(
        self, canvas_width: Optional[float] = ..., canvas_height: Optional[float] = ...,
    ) -> Tuple[float, float, float, float]:
        """Return ``(x, y, w, h)`` of the *unrotated* content rect in canvas-space.

        Position is parent-relative: ``x`` / ``y`` are offsets from the
        parent UI element's top-left (or from the canvas origin).

        Args:
            canvas_width: Reference canvas width in pixels.
            canvas_height: Reference canvas height in pixels.
        """
        ...

    def get_visual_rect(
        self, canvas_width: Optional[float] = ..., canvas_height: Optional[float] = ...,
    ) -> Tuple[float, float, float, float]:
        """Return the axis-aligned bounding box of the rotated content rect.

        Rotation is applied around the center of the unrotated rect.
        Returns ``(vx, vy, vw, vh)`` in canvas-space.
        """
        ...

    def calc_visual_size(self, width: float, height: float) -> Tuple[float, float]:
        """Return rotated AABB size for the given unrotated width/height."""
        ...

    def get_rotated_corners(
        self, canvas_width: Optional[float] = ..., canvas_height: Optional[float] = ...,
    ) -> List[Tuple[float, float]]:
        """Return rotated rect corners in TL, TR, BR, BL order."""
        ...

    def set_rect(
        self,
        rect_x: float, rect_y: float, rect_w: float, rect_h: float,
        canvas_width: float, canvas_height: float,
    ) -> None:
        """Store a canvas-space rect back into parent-relative serialized fields.

        Args:
            rect_x: Target X position in canvas-space.
            rect_y: Target Y position in canvas-space.
            rect_w: Target width.
            rect_h: Target height.
            canvas_width: Reference canvas width.
            canvas_height: Reference canvas height.
        """
        ...

    def set_visual_position(
        self,
        vis_x: float, vis_y: float,
        canvas_width: float, canvas_height: float,
    ) -> None:
        """Move the element so the visual AABB top-left is at ``(vis_x, vis_y)``.

        Keeps width/height/rotation unchanged; only adjusts x/y.
        """
        ...

    def set_size_preserve_visual_position(
        self,
        width: float, height: float,
        canvas_width: float, canvas_height: float,
    ) -> None:
        """Set width/height while keeping current visual AABB top-left unchanged."""
        ...

    def set_size_preserve_center(
        self,
        width: float, height: float,
        canvas_width: float, canvas_height: float,
    ) -> None:
        """Set width/height while keeping the element's visual center fixed."""
        ...

    def set_size_preserve_corner(
        self,
        width: float, height: float,
        canvas_width: float, canvas_height: float,
        corner: str = ...,
    ) -> None:
        """Set width/height while keeping a rotated corner fixed.

        Args:
            width: New width in canvas pixels.
            height: New height in canvas pixels.
            canvas_width: Reference canvas width.
            canvas_height: Reference canvas height.
            corner: Which corner to preserve — one of
                ``"top_left"``, ``"top_right"``, ``"bottom_right"``, ``"bottom_left"``.
        """
        ...

    # ------------------------------------------------------------------
    # Pointer event hooks — override in subclasses
    # ------------------------------------------------------------------

    def on_pointer_enter(self, event_data: PointerEventData) -> None:
        """Called when the pointer enters this element's rect."""
        ...

    def on_pointer_exit(self, event_data: PointerEventData) -> None:
        """Called when the pointer leaves this element's rect."""
        ...

    def on_pointer_down(self, event_data: PointerEventData) -> None:
        """Called when a mouse button is pressed over this element."""
        ...

    def on_pointer_up(self, event_data: PointerEventData) -> None:
        """Called when a mouse button is released over this element."""
        ...

    def on_pointer_click(self, event_data: PointerEventData) -> None:
        """Called on a complete click (down + up on the same element)."""
        ...

    def on_begin_drag(self, event_data: PointerEventData) -> None:
        """Called when a drag gesture starts on this element."""
        ...

    def on_drag(self, event_data: PointerEventData) -> None:
        """Called each frame during a drag gesture."""
        ...

    def on_end_drag(self, event_data: PointerEventData) -> None:
        """Called when a drag gesture ends."""
        ...

    def on_scroll(self, event_data: PointerEventData) -> None:
        """Called when the scroll wheel is used over this element."""
        ...

    # ------------------------------------------------------------------
    # Hit-testing
    # ------------------------------------------------------------------

    def contains_point(
        self,
        px: float, py: float,
        canvas_width: float, canvas_height: float,
        tolerance: float = ...,
    ) -> bool:
        """Test whether canvas-space point ``(px, py)`` lies inside this element.

        Uses the oriented (rotated) bounding box for accurate hit-testing.
        ``tolerance`` expands the hit area in canvas pixels.
        """
        ...
