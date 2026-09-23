"""InxUIComponent — abstract base for all UI components.

All UI-related components (screen-space, world-space, canvas, etc.) should
inherit from this class instead of InxComponent directly.

Hierarchy:
    InxComponent
        └─ InxUIComponent
             ├─ InxUIScreenComponent   (Transform-backed 2D rect: w, h)
             └─ InxUIWorldComponent    (3D world-space UI — future)
"""

from Infernux.components import InxComponent


class InxUIComponent(InxComponent):
    """Base class for every UI component in Infernux.

    Provides:
    - ``_component_category_ = "UI"`` so that all UI components are grouped
      together in the *Add Component* menu.
    """

    _component_category_ = "UI"
