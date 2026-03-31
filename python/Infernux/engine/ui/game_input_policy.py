from __future__ import annotations


def should_route_game_input(*, is_playing: bool, panel_focused: bool, viewport_hovered: bool, cursor_locked: bool) -> bool:
    """Return whether runtime input should be visible to game scripts.

    In the editor, keyboard/gameplay input should flow while Play Mode is
    active and the Game panel owns focus. Hovering the rendered viewport is
    also sufficient, because clicking the image can happen before ImGui focus
    settles on the dock tab. Cursor lock always keeps input active so the
    escape-to-unlock safety path still works.
    """
    if cursor_locked:
        return True
    if not is_playing:
        return False
    return panel_focused or viewport_hovered


def should_process_game_ui_events(*, is_playing: bool, viewport_hovered: bool, cursor_locked: bool) -> bool:
    """Return whether screen-space UI pointer events should be dispatched."""
    return is_playing and viewport_hovered and not cursor_locked


__all__ = [
    "should_process_game_ui_events",
    "should_route_game_input",
]