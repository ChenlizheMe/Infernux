"""
Reusable ImGui style bundles for editor panels.

Centralizes common ``push_style_var`` / ``push_style_color`` combinations so
spacing stays consistent when ``Theme`` or inspector tokens change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .theme import Theme, ImGuiStyleVar

if TYPE_CHECKING:
    from Infernux.lib import InxGUIContext


def push_inspector_body_layout(ctx: "InxGUIContext") -> int:
    """Frame + item spacing for the main Inspector property list."""
    ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_FRAME_PAD)
    ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_ITEM_SPC)
    return 2


def push_inspector_subitem_layout(ctx: "InxGUIContext") -> int:
    """Spacing for nested rows under a component."""
    ctx.push_style_var_vec2(ImGuiStyleVar.FramePadding, *Theme.INSPECTOR_FRAME_PAD)
    ctx.push_style_var_vec2(ImGuiStyleVar.ItemSpacing, *Theme.INSPECTOR_SUBITEM_SPC)
    return 2


def push_inspector_material_block(ctx: "InxGUIContext") -> int:
    """Spacing used by material / render-stack inspector blocks."""
    return push_inspector_body_layout(ctx)
