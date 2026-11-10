import json
import time

from infernux.lib import ConsolePanel
from tests.gpu.line_bounds_case import LineBoundsCase


def test_visible_ribbon_is_not_culled_by_its_offscreen_centerline(engine, scene, tmp_path):
    case = LineBoundsCase(engine, scene)
    engine.resize_game_render_target(640, 480)
    engine.set_render_pipeline(case.pipeline)
    engine.set_game_camera_enabled(True)
    phase, phase_execution, ticket = 0, 0, None
    observations = []
    mesh_published = False
    deadline = time.monotonic() + 30
    try:
        # The shared engine may already have exited a previous test's main
        # loop. Drive rendering directly; culling does not depend on Run state.
        while phase < len(case.phases):
            engine.request_full_speed_frame()
            engine.tick(1 / 60)
            assert time.monotonic() < deadline
            if ticket is None:
                execution = engine.renderer_frame_snapshot['game_render_graph_execution_count']
                if not mesh_published:
                    try:
                        engine.get_object_mesh_index_gpu_info(case.line_object.id)
                    except IndexError:
                        continue
                    mesh_published = True
                    phase_execution = execution
                # A curve change can replace the ribbon buffers. The zero-width
                # phase can be culled and does not need a new allocation.
                if engine.pending_mesh_gpu_upload_count:
                    phase_execution = execution
                if engine.game_presentation_ready and execution > phase_execution:
                    ticket = engine.request_render_target_readback(True)
            elif ticket.done:
                assert not ticket.error, ticket.error
                observations.append(case.observe(ticket.result_numpy()))
                phase += 1
                if phase < len(case.phases):
                    case.change(phase)
                    phase_execution = engine.renderer_frame_snapshot['game_render_graph_execution_count']
                    ticket = None
        assert len(observations) == len(case.phases)
        console = ConsolePanel()
        assert console.get_error_count() == console.get_warning_count() == 0
    finally:
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        case.close()
        (tmp_path/'line-bounds-pixels.json').write_text(json.dumps(observations, indent=2), encoding='utf-8')
