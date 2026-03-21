"""Scene View shortcuts / input helpers.

Centralises keyboard constants and shortcut queries for the 3D Scene View
panel.  This keeps scene_view_panel.py focused on camera logic and rendering.

Usage (inside SceneViewPanel)::

    from .scene_shortcuts import SceneInput, TOOL_TRANSLATE, TOOL_ROTATE, TOOL_SCALE

    inp = SceneInput(ctx, viewport_hovered)
    if inp.tool_translate_pressed:
        self._set_tool(TOOL_TRANSLATE)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from InfEngine.lib import InfGUIContext

from .imgui_keys import (
    KEY_Q,
    KEY_W,
    KEY_E,
    KEY_R,
    KEY_A,
    KEY_S,
    KEY_D,
    KEY_F,
    KEY_LEFT_SHIFT,
    KEY_RIGHT_SHIFT,
)

# Tool-mode constants (must match C++ EditorTools::ToolMode)
TOOL_NONE      = 0
TOOL_TRANSLATE = 1
TOOL_ROTATE    = 2
TOOL_SCALE     = 3


class SceneInput:
    """Frame-snapshot of keyboard / mouse state for the Scene View.

    Create once per frame inside the Scene View's render loop.
    """

    __slots__ = (
        "ctx",
        "hovered",
        # movement (RMB held)
        "key_w",
        "key_s",
        "key_a",
        "key_d",
        "key_q",
        "key_e",
        "shift_down",
        # tool shortcuts (single press while viewport hovered, no RMB)
        "tool_none_pressed",
        "tool_translate_pressed",
        "tool_rotate_pressed",
        "tool_scale_pressed",
        # focus shortcut
        "focus_pressed",
    )

    def __init__(self, ctx: "InfGUIContext", hovered: bool, right_down: bool = False):
        self.ctx = ctx
        self.hovered = hovered

        # Fly-mode movement keys (only when RMB held)
        self.key_w = right_down and ctx.is_key_down(KEY_W)
        self.key_s = right_down and ctx.is_key_down(KEY_S)
        self.key_a = right_down and ctx.is_key_down(KEY_A)
        self.key_d = right_down and ctx.is_key_down(KEY_D)
        self.key_q = right_down and ctx.is_key_down(KEY_Q)
        self.key_e = right_down and ctx.is_key_down(KEY_E)
        self.shift_down = (
            ctx.is_key_down(KEY_LEFT_SHIFT) or ctx.is_key_down(KEY_RIGHT_SHIFT)
        )

        # Tool switching (only when viewport hovered, not during RMB)
        can_switch = hovered and not right_down
        self.tool_none_pressed = can_switch and ctx.is_key_pressed(KEY_Q)
        self.tool_translate_pressed = can_switch and ctx.is_key_pressed(KEY_W)
        self.tool_rotate_pressed = can_switch and ctx.is_key_pressed(KEY_E)
        self.tool_scale_pressed = can_switch and ctx.is_key_pressed(KEY_R)

        # F to focus on selection
        self.focus_pressed = can_switch and ctx.is_key_pressed(KEY_F)
