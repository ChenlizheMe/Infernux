"""Material AlphaClip must not disable authored coverage in any linked pass."""
import json

import pytest

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager
from tests.gpu.surface_discard_case import STEPS, SurfaceDiscardCase


@pytest.mark.parametrize('kind', ['direct', 'helper', 'texture'])
@pytest.mark.parametrize('pipeline,samples', [('forward',1), ('forward',4), ('forward_plus',1),
                                            ('forward_plus',4), ('deferred',1), ('depth',1)])
def test_surface_discard_survives_material_alpha_clip_switch(engine, scene, tmp_path, monkeypatch, kind, pipeline, samples):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', engine.get_asset_database())
    case = None
    console = ConsolePanel()
    manager = SceneManager.instance()
    previous_rendering = engine.is_play_mode_rendering()
    try:
        case = SurfaceDiscardCase(engine, scene, tmp_path, kind, pipeline, samples)
        scene.main_camera = case.camera
        engine.resize_game_render_target(320, 240)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(case.pipeline)
        manager.play()
        manager.pause()
        engine.set_play_mode_rendering(True)
        phase = frame = changed = 0
        ticket = None
        results, failures = [], []

        def after_draw():
            nonlocal phase, frame, changed, ticket
            try:
                frame += 1
                assert frame < 180, results
                if ticket is None and frame >= changed + 8:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    assert manager.get_active_scene().world_id == scene.world_id
                    assert scene.effective_game_camera.component_id == case.camera.component_id
                    results.append(case.observe(ticket.result_numpy()))
                    (tmp_path / 'surface-discard.json').write_text(json.dumps(dict(
                        kind=kind, pipeline=pipeline, samples=samples, phases=results)), encoding='utf-8')
                    issues = [item for item in console._get_visible_log_snapshot(1000)
                              if item['level'] in ('ERROR','FATAL','WARN','WARNING')]
                    assert not issues, issues
                    phase += 1
                    if phase == len(STEPS):
                        engine.exit()
                    else:
                        case.change(STEPS[phase])
                        ticket, changed = None, frame
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert phase == len(STEPS)
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        manager.stop()
        engine.set_play_mode_rendering(previous_rendering)
        if case is not None:
            case.close()
