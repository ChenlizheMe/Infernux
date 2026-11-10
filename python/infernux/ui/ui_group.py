"""Subtree-wide UI presentation and interaction policy."""

from __future__ import annotations

from infernux.components import add_component_menu, serialized_field

from .inx_ui_component import InxUIComponent
from .ui_render_revision import (
    _invalidate_ui_command_groups,
    _mark_hit_policy_dirty,
    is_unchanged_ui_value,
    mark_runtime_ui_dirty,
)


@add_component_menu("UI/Group")
class UIGroup(InxUIComponent):
    """Multiply opacity and gate interaction for a GameObject subtree."""

    alpha: float = serialized_field(
        default=1.0, range=(0.0, 1.0), slider=True, group="Group"
    )
    interactable: bool = serialized_field(default=True, group="Group")
    blocks_raycast: bool = serialized_field(default=True, group="Group")

    def __setattr__(self, name, value):
        group_field = name in ("alpha", "interactable", "blocks_raycast", "enabled")
        changed = group_field and not is_unchanged_ui_value(self, name, value)
        super().__setattr__(name, value)
        if changed:
            self._invalidate_group(name)

    def _invalidate_group(self, name):
        from .inx_ui_screen_component import is_ui_screen_component

        owner = self._try_get_game_object()
        if owner is None:
            return
        if name in ("blocks_raycast", "enabled"):
            _mark_hit_policy_dirty()
        mark_runtime_ui_dirty(self)
        # Group inheritance includes visuals on the owner and crosses nested
        # Canvases. Keep all unrelated command groups and hit geometry warm.
        pending = [owner]
        while pending:
            node = pending.pop()
            for element in node.get_py_components():
                if is_ui_screen_component(element):
                    element.__dict__.pop("_group_state_cache", None)
                    _invalidate_ui_command_groups(element, alpha_only=name == "alpha")
            pending.extend(node.get_children())
