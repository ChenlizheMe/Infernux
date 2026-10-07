from __future__ import annotations

from infernux.engine.ui.game_input_policy import should_process_game_ui_events, should_route_game_input


class TestShouldRouteGameInput:
    def test_routes_input_when_cursor_locked(self):
        assert should_route_game_input(
            is_playing=False,
            panel_focused=False,
            cursor_locked=True,
        ) is True

    def test_routes_input_when_playing_and_panel_focused(self):
        assert should_route_game_input(
            is_playing=True,
            panel_focused=True,
            cursor_locked=False,
        ) is True

    def test_blocks_input_when_only_viewport_is_hovered(self):
        assert should_route_game_input(
            is_playing=True,
            panel_focused=False,
            cursor_locked=False,
        ) is False

    def test_blocks_input_when_not_playing_and_not_locked(self):
        assert should_route_game_input(
            is_playing=False,
            panel_focused=True,
            cursor_locked=False,
        ) is False


class TestShouldProcessGameUiEvents:
    def test_processes_ui_while_playing_and_game_panel_focused(self):
        assert should_process_game_ui_events(
            is_playing=True,
            panel_focused=True,
            cursor_locked=False,
        ) is True

    def test_blocks_ui_when_cursor_locked(self):
        assert should_process_game_ui_events(
            is_playing=True,
            panel_focused=True,
            cursor_locked=True,
        ) is False

    def test_blocks_ui_when_game_panel_is_not_focused(self):
        assert should_process_game_ui_events(
            is_playing=True,
            panel_focused=False,
            cursor_locked=False,
        ) is False
