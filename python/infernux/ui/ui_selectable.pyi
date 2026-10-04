"""Type stubs for infernux.ui.ui_selectable — interactive UI element base."""

from __future__ import annotations

from typing import List

from infernux.ui.inx_ui_screen_component import InxUIScreenComponent
from infernux.ui.enums import UITransitionType
from infernux.ui.ui_event_data import PointerEventData


class SelectionState:
    """Visual state indices — mirrors Unity's ``SelectionState``."""
    Normal: int
    Highlighted: int
    Pressed: int
    Disabled: int


class UISelectable(InxUIScreenComponent):
    """Base class for interactive UI elements with visual state feedback.

    Provides Normal / Highlighted / Pressed / Disabled visual states
    with ColorTint transitions.  ``UIButton`` inherits this and adds
    an ``on_click`` event.

    Attributes:
        interactable: Whether the user can interact with this element.
        transition: How visual states are displayed.
        normal_color: RGBA tint when idle.
        highlighted_color: RGBA tint when hovered.
        pressed_color: RGBA tint when pressed.
        disabled_color: RGBA tint when disabled.
    """

    interactable: bool
    transition: UITransitionType
    normal_color: list
    highlighted_color: list
    pressed_color: list
    disabled_color: list

    @property
    def current_selection_state(self) -> int:
        """The current visual state index (see ``SelectionState``)."""
        ...

    def get_current_tint(self) -> List[float]:
        """Return the ``[R, G, B, A]`` tint for the current visual state."""
        ...

    def awake(self) -> None: ...
    def on_pointer_enter(self, event_data: PointerEventData) -> None: ...
    def on_pointer_exit(self, event_data: PointerEventData) -> None: ...
    def on_pointer_down(self, event_data: PointerEventData) -> None: ...
    def on_pointer_up(self, event_data: PointerEventData) -> None: ...
