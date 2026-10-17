from __future__ import annotations

import pytest

from infernux.engine.ui.game_input_policy import (
    should_process_game_ui_events,
    should_route_game_input,
)


@pytest.mark.parametrize(
    "policy,playing,focused,locked,expected",
    [
        pytest.param(should_route_game_input, False, False, True, False, id="case_1"),
        pytest.param(should_route_game_input, True, True, False, True, id="case_2"),
        pytest.param(should_route_game_input, True, False, False, False, id="case_3"),
        pytest.param(should_route_game_input, False, True, False, False, id="case_4"),
        pytest.param(should_process_game_ui_events, True, True, False, True, id="case_5"),
        pytest.param(should_process_game_ui_events, True, True, True, False, id="case_6"),
        pytest.param(should_process_game_ui_events, True, False, False, False, id="case_7"),
    ],
)
def test_game_input_policy(policy, playing, focused, locked, expected):
    assert policy(
        is_playing=playing,
        panel_focused=focused,
        cursor_locked=locked,
    ) is expected
