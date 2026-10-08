import json
import time

from infernux.lib import ConsolePanel
from tests.gpu.line_bounds_case import LineBoundsCase


def test_visible_ribbon_is_not_culled_by_its_offscreen_centerline(engine, scene, tmp_path):
    case = LineBoundsCase(engine, scene)
    engine.resize_game_render_target(640, 480)
    engine.set_render_pipeline(case.pipeline)
    engine.set_game_camera_enabled(True)
    phase, frames, ticket = 0, 0, None
    failures, observations = [], []
    deadline = time.monotonic() + 30
    try:
        def tick():
            nonlocal phase, frames, ticket
            try:
                engine.request_full_speed_frame()
                frames += 1
                assert time.monotonic() < deadline
                if ticket is None and frames >= 3:
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
        (tmp_path/'line-bounds-pixels.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')
