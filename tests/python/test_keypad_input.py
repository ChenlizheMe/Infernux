"""Keypad names match the real native enum and graphical input events."""
import pytest

from infernux.input import Input, KeyCode
from infernux.lib import InputManager
from infernux.renderstack import RenderStackPipeline


@pytest.mark.parametrize('digit', range(10))
@pytest.mark.parametrize('spelling', ['keypad_{}', 'KEYPAD_{}', 'Keypad {}'])
def test_keypad_name_matches_native_code(digit, spelling):
    name = spelling.format(digit)
    expected = getattr(KeyCode, f'KEYPAD{digit}')
    assert InputManager.name_to_scancode(name) == expected
    assert Input._resolve_key(name) == expected


def test_all_keypad_digits_reach_public_queries_through_graphical_events(engine):
    observed = []
    frame = 0
    focused = Input.is_game_focused()

    def update(_dt):
        nonlocal frame
        digit, released = divmod(frame, 2)
        name = f'keypad_{digit}'
        observed.append((Input.get_key(name), Input.get_key_down(name), Input.get_key_up(name)))
        frame += 1
        if frame == 20:
            engine.exit()
        elif not released:
            engine.queue_synthetic_key_input(getattr(KeyCode, f'KEYPAD{digit}'), False)
        else:
            engine.queue_synthetic_key_input(getattr(KeyCode, f'KEYPAD{digit + 1}'), True)

    try:
        Input.set_game_focused(True)
        engine.set_render_pipeline(RenderStackPipeline())
        engine.set_pre_scene_update_callback(update)
        engine.queue_synthetic_key_input(KeyCode.KEYPAD0, True)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_render_pipeline(None)
        Input.set_game_focused(focused)
    assert observed == [(True, True, False), (False, False, True)] * 10
