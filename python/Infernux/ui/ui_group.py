"""Subtree-wide UI presentation and interaction policy."""

from __future__ import annotations

from Infernux.components import add_component_menu, serialized_field

from .inx_ui_component import InxUIComponent
from .ui_render_revision import is_unchanged_ui_scalar, mark_runtime_ui_dirty


@add_component_menu("UI/Group")
class UIGroup(InxUIComponent):
    """Multiply opacity and gate interaction for a GameObject subtree."""

    alpha: float = serialized_field(
        default=1.0, range=(0.0, 1.0), slider=True, group="Group"
    )
    interactable: bool = serialized_field(default=True, group="Group")
    blocks_raycast: bool = serialized_field(default=True, group="Group")

    def __setattr__(self, name, value):
        unchanged = is_unchanged_ui_scalar(self, name, value)
        super().__setattr__(name, value)
        if not name.startswith("_") and not unchanged:
            mark_runtime_ui_dirty()
            if name == "blocks_raycast":
                from .ui_render_revision import _mark_hit_policy_dirty

                _mark_hit_policy_dirty()
