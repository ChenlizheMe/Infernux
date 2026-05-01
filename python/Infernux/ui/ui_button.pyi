"""Type stubs for Infernux.ui.ui_button — clickable button UI element."""

from __future__ import annotations

from typing import List

from Infernux.ui.enums import TextAlignH, TextAlignV
from Infernux.ui.ui_selectable import UISelectable
from Infernux.ui.ui_event import UIEvent
from Infernux.ui.ui_event_entry import UIEventEntry
from Infernux.ui.ui_event_data import PointerEventData


class UIButton(UISelectable):
    """A clickable button with visual state feedback.

    Combines **Image** (background) and **Text** (label) capabilities.
    Fires ``on_click`` when the user performs a full click (down + up).

    Attributes:
        label: Button label text.
        font_size: Label font size in canvas pixels.
        font_path: Optional font asset path.
        label_color: Label text colour as ``[R, G, B, A]``.
        text_align_h: Horizontal text alignment.
        text_align_v: Vertical text alignment.
        line_height: Line height multiplier.
        letter_spacing: Extra letter spacing in pixels.
        texture_path: Background image texture path.
        background_color: Background fill colour as ``[R, G, B, A]``.
        on_click_entries: Persistent click handlers (serialized).

    Example::

        class MyUI(InxComponent):
            def start(self):
                start_btn = GameObject.find("StartBtn")
                if start_btn is None:
                    return
                btn = start_btn.get_component(UIButton)
                btn.on_click.add_listener(self.on_start)

            def on_start(self):
                print("Start clicked!")
    """

    label: str
    font_size: float
    font_path: str
    label_color: list
    text_align_h: TextAlignH
    text_align_v: TextAlignV
    line_height: float
    letter_spacing: float
    texture_path: str
    background_color: list
    on_click_entries: List[UIEventEntry]

    def awake(self) -> None: ...

    @property
    def on_click(self) -> UIEvent:
        """The click event — call ``add_listener()`` to subscribe.

        Example::

            btn.on_click.add_listener(my_handler)
        """
        ...

    def on_pointer_click(self, event_data: PointerEventData) -> None:
        """Internal — fires ``on_click`` and persistent entries on click."""
        ...
