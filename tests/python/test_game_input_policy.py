from __future__ import annotations

from types import SimpleNamespace

import pytest

import infernux.input as input_api
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


def test_hovered_world_receives_mouse_motion_without_keyboard_focus(monkeypatch):
    sample = SimpleNamespace(
        _read_world_id=2,
        _sampled_pointer_active=True,
        _sampled_game_focused=False,
        has_synthetic_gameplay_input=False,
        mouse_delta_x=3.0,
        mouse_delta_y=-2.0,
    )
    monkeypatch.setattr(input_api, "_NativeInputManager", SimpleNamespace(instance=lambda: sample))
    controls = input_api.Input
    assert controls.get_mouse_delta() == (3.0, -2.0)
    assert controls.get_axis_raw("Mouse Y") == -2.0 * controls.mouse_sensitivity
    assert not controls.get_key("d")
    sample._sampled_pointer_active = False
    assert controls.get_mouse_delta() == (0.0, 0.0)
