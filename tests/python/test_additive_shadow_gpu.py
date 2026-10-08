"""Active-scene changes must not change shadows in the resident render world."""
import json
import time

import pytest

from infernux.lib import ConsolePanel
from tests.gpu.additive_shadow_case import AdditiveShadowCase


@pytest.mark.parametrize('path', ('forward', 'forward_plus', 'deferred'))
def test_resident_light_shadow_pixels(engine, scene, tmp_path, path):
    case = AdditiveShadowCase(scene, path)
    engine.resize_game_render_target(240, 180)
    engine.set_render_pipeline(case.pipeline)
    engine.set_game_camera_enabled(True)
    phase, frames, ticket = 0, 0, None
    failures, observations = [], []
    deadline = time.monotonic() + 60
    try:
        def tick():
            nonlocal phase, frames, ticket
            try:
                engine.request_full_speed_frame()
                frames += 1
                assert time.monotonic() < deadline
                if ticket is None and frames >= 8:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    assert not ticket.error, ticket.error
                    observations.append(case.observe(ticket.result_numpy()))
                    phase += 1
                    if phase == len(case.phases):
                        engine.exit()
                    else:
                        case.change(phase)
                        ticket, frames = None, 0
            except BaseException as error:
                failures.append(error)
                engine.exit()
        engine.set_post_draw_callback(tick)
        engine.run()
        if failures:
            raise failures[0]
        assert len(observations) == len(case.phases)
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        case.close()
        (tmp_path / 'additive-shadows.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')
