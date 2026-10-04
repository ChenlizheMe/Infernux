"""Type stubs for infernux.ui.ui_event_system — per-frame pointer state machine."""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from infernux.ui.ui_canvas import UICanvas


class UIEventProcessor:
    """Per-frame pointer event dispatcher for screen-space UI.

    Converts raw mouse state into high-level pointer events (enter / exit /
    down / up / click / drag / scroll) dispatched to ``InxUIScreenComponent``
    handlers.  One processor is created per Game View.

    Example::

        processor = UIEventProcessor()
        # Each frame, after rendering screen UI:
        processor.process(canvases, canvas_positions, mouse_down, mouse_up,
                          mouse_held, scroll_delta, dt)
    """

    def __init__(self) -> None: ...

    def process(
        self,
        canvases: List[UICanvas],
        canvas_positions: List[Tuple[float, float]],
        mouse_down: bool,
        mouse_up: bool,
        mouse_held: bool,
        scroll_delta: Tuple[float, float],
        dt: float,
    ) -> None:
        """Run one frame of event processing.

        Args:
            canvases: Sorted list of active canvases (by sort_order).
            canvas_positions: Per-canvas pointer position in design pixels.
                Must be the same length as *canvases*.
            mouse_down: ``True`` during the frame left-button was pressed.
            mouse_up: ``True`` during the frame left-button was released.
            mouse_held: ``True`` while left-button is held.
            scroll_delta: ``(sx, sy)`` scroll delta this frame.
            dt: Delta time in seconds since last frame.
        """
        ...

    def reset(self) -> None:
        """Clear all transient state (e.g. when play mode stops)."""
        ...
