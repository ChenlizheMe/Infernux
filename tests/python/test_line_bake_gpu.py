import json
import time

import pytest

from infernux.lib import ConsolePanel
from tests.gpu.line_bake_case import LineBakeCase


@pytest.mark.parametrize('scenario',LineBakeCase.scenarios)
def test_baked_line_matches_realtime_ribbon_pixels(engine,scene,tmp_path,scenario):
    case=LineBakeCase(engine,scene,scenario)
    engine.resize_game_render_target(480,360)
    engine.set_render_pipeline(case.pipeline)
    engine.set_game_camera_enabled(True)
    mesh_ready_execution, ticket = None, None
    failures,observations=[],[]
    deadline=time.monotonic()+25
    try:
        def tick():
            nonlocal mesh_ready_execution, ticket
            try:
                engine.request_full_speed_frame()
                assert time.monotonic()<deadline,observations
                if ticket is None:
                    execution = engine.renderer_frame_snapshot["game_render_graph_execution_count"]
                    if mesh_ready_execution is None:
                        ribbon = case.line_object if case.realtime is None else case.baked_object
                        try:
                            engine.get_object_mesh_index_gpu_info(ribbon.id)
                        except IndexError:
                            return
                        mesh_ready_execution = execution
                    elif engine.game_presentation_ready and execution > mesh_ready_execution:
                        ticket = engine.request_render_target_readback(True)
                elif ticket.done:
                    assert not ticket.error,ticket.error
                    result=case.observe(ticket.result_numpy())
                    observations.append(result)
                    if result['done']:
                        engine.exit()
                    else:
                        mesh_ready_execution, ticket = None, None
            except BaseException as error:
                failures.append(error)
                engine.exit()
        engine.set_post_draw_callback(tick)
        engine.run()
        if failures:
            raise failures[0]
        assert observations and observations[-1]["done"], observations
        console=ConsolePanel()
        assert console.get_error_count()==console.get_warning_count()==0,console._get_visible_log_snapshot(100)
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        case.close()
        (tmp_path/'line-bake-pixels.json').write_text(json.dumps(observations,indent=2),encoding='utf-8')
