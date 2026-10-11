from __future__ import annotations


def should_route_game_input(*, is_playing: bool, panel_focused: bool, cursor_locked: bool) -> bool:
    """Return whether runtime input should be visible to game scripts.

    In the editor, keyboard/gameplay input should flow while Play Mode is
    active and the Game panel owns focus. Cursor lock always keeps input
    active so the escape-to-unlock safety path still works.
    """
    if cursor_locked:
        return True
    if not is_playing:
        return False
    return panel_focused


def should_process_game_ui_events(*, is_playing: bool, panel_focused: bool, cursor_locked: bool) -> bool:
    """Return whether screen-space UI pointer events should be dispatched.

    The GPU-backed screen UI is not an ImGui item, so its raycast must not be
    gated by the current ImGui hover result. A Game View that owns focus must
    receive its pointer down/up frames even when an invisible viewport overlay
    is the active ImGui item.
    """
    return is_playing and panel_focused and not cursor_locked


__all__ = [
    "should_process_game_ui_events",
    "should_route_game_input",
]
