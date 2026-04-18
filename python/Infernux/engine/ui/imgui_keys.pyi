"""imgui_keys — ImGui key-code constants.

Usage::

    from Infernux.engine.ui.imgui_keys import KEY_ENTER, KEY_ESCAPE
"""

from __future__ import annotations

# Navigation
KEY_TAB: int
KEY_LEFT_ARROW: int
KEY_RIGHT_ARROW: int
KEY_UP_ARROW: int
KEY_DOWN_ARROW: int
KEY_PAGE_UP: int
KEY_PAGE_DOWN: int
KEY_HOME: int
KEY_END: int

# Editing
KEY_INSERT: int
KEY_DELETE: int
KEY_BACKSPACE: int
KEY_SPACE: int
KEY_ENTER: int
KEY_ESCAPE: int

# Modifier bit flags (use with ``ctx.is_key_down(MOD_CTRL)``)
MOD_CTRL: int

# Modifiers
KEY_LEFT_CTRL: int
KEY_LEFT_SHIFT: int
KEY_LEFT_ALT: int
KEY_LEFT_SUPER: int
KEY_RIGHT_CTRL: int
KEY_RIGHT_SHIFT: int
KEY_RIGHT_ALT: int
KEY_RIGHT_SUPER: int

# Digits  (KEY_0 = 535 … KEY_9 = 544)
KEY_0: int
KEY_1: int
KEY_2: int
KEY_3: int
KEY_4: int
KEY_5: int
KEY_6: int
KEY_7: int
KEY_8: int
KEY_9: int

# Letters  (KEY_A = 545 … KEY_Z = 570)
KEY_A: int
KEY_B: int
KEY_C: int
KEY_D: int
KEY_E: int
KEY_F: int
KEY_G: int
KEY_H: int
KEY_I: int
KEY_J: int
KEY_K: int
KEY_L: int
KEY_M: int
KEY_N: int
KEY_O: int
KEY_P: int
KEY_Q: int
KEY_R: int
KEY_S: int
KEY_T: int
KEY_U: int
KEY_V: int
KEY_W: int
KEY_X: int
KEY_Y: int
KEY_Z: int

# Function keys  (KEY_F1 = 572 … KEY_F12 = 583)
KEY_F1: int
KEY_F2: int
KEY_F3: int
KEY_F4: int
KEY_F5: int
KEY_F6: int
KEY_F7: int
KEY_F8: int
KEY_F9: int
KEY_F10: int
KEY_F11: int
KEY_F12: int
