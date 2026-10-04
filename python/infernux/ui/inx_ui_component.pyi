"""Type stubs for infernux.ui.inx_ui_component — abstract base for all UI components."""

from __future__ import annotations

from infernux.components import InxComponent


class InxUIComponent(InxComponent):
    """Base class for every UI component in infernux.

    All UI-related components (screen-space, world-space, canvas, etc.)
    should inherit from this class instead of ``InxComponent`` directly.

    The ``_component_category_`` is set to ``"UI"`` so that all UI
    components are grouped together in the *Add Component* menu.
    """

    _component_category_: str
