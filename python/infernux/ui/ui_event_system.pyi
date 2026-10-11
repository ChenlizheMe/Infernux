"""Type stubs for infernux.ui.ui_event_system — per-frame pointer state machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, TYPE_CHECKING

from .ui_event_data import PointerType

if TYPE_CHECKING:
    from infernux.ui.ui_canvas import UICanvas


@dataclass(frozen=True, slots=True)
class UIPointerFrame:
    """One physical pointer snapshot expressed in every canvas's coordinates."""

    pointer_id: int
    pointer_type: PointerType
    canvas_positions: Tuple[Tuple[float, float], ...]
    down: bool = False
    up: bool = False
    held: bool = False
    canceled: bool = False
    scroll_delta: Tuple[float, float] = (0.0, 0.0)
    press_canvas_positions: Tuple[Tuple[float, float], ...] = ()


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

    def process_pointers(
        self, canvases: Sequence[UICanvas], pointers: Sequence[UIPointerFrame], dt: float,
    ) -> None:
        """Dispatch one complete physical pointer snapshot."""
        ...

    def reset(self) -> None:
        """Cancel every active pointer transaction."""
        ...

    def discard(self) -> None:
        """Forget state after the owning scene has retired."""
        ...

    def debug_state(self) -> dict:
        """Return the latest transition without polling input."""
        ...
