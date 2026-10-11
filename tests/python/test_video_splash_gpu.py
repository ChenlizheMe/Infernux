"""Native decode, upload, GUI draws and editor framebuffer capture for Splash."""
import json
import io

import pytest
from PIL import Image

from infernux.lib import ConsolePanel
from infernux.renderstack.render_stack_pipeline import RenderStackPipeline
from tests.gpu.video_splash_case import VideoSplashCase


def test_consecutive_video_frames_render_and_retire(engine,scene,tmp_path):
    pipeline = RenderStackPipeline()
    engine.set_render_pipeline(pipeline)
    case = VideoSplashCase(engine,tmp_path/'Videos')
    console = ConsolePanel()
    frame = phase = 0
    capture = None
    results,failures = [],[]
    try:
        def after_draw():
            nonlocal frame,phase,capture
            try:
                frame += 1
                assert frame < 500, (phase,case.poll())
                if capture is None:
                    ready = case.poll()
                    if ready['done']:
                        capture = engine.request_capture('editor',str(tmp_path/f'frame-{phase}.png'))
                        results.append(ready)
                else:
                    status = engine.query_capture(capture)
                    assert status['status'] not in ('failed','cancelled','source_expired'),status
                    if status['status'] == 'completed':
                        results[-1]['capture'] = case.observe_capture(tmp_path/f'frame-{phase}.png')
                        issues = [r for r in console._get_visible_log_snapshot(1000)
                                  if r['level'] in ('ERROR','FATAL','WARN','WARNING')]
                        assert not issues,issues
                        phase += 1
                        if phase == case.phases:
                            engine.exit()
                        else:
                            case.change(phase)
                            capture = None
            except BaseException as error:
                failures.append(error)
                engine.exit()
        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert len(results) == case.phases
    finally:
        engine.set_post_draw_callback(None)
        case.close()
        engine.set_render_pipeline(None)
        pipeline.dispose()
        (tmp_path/'splash-results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


@pytest.mark.parametrize('reuse_key',[False,True])
def test_retired_decode_cannot_republish_into_stream_slot(engine,scene,tmp_path,reuse_key):
    key = f'retired-splash-{tmp_path.name}'
    control_key = key if reuse_key else key+'-control'
    def jpeg(size,color):
        output = io.BytesIO()
        Image.new('RGB',size,color).save(output,format='JPEG')
        return output.getvalue()
    assert engine.schedule_texture_preview_from_memory(key,jpeg((16,8),'red'),1,False)
    # Retire before the first owner-thread pump; the worker may already have
    # decoded, but neither its result nor its upload may regain publication.
    engine.release_texture_preview_task(key)
    assert engine.get_texture_preview_texture_id(key) == 0
    assert tuple(engine.get_texture_preview_size(key)) == (0,0)
    assert engine.schedule_texture_preview_from_memory(control_key,jpeg((8,16),'blue'),1,False)
    pipeline = RenderStackPipeline()
    engine.set_render_pipeline(pipeline)
    frame,ready_frame = 0,None
    failures = []
    try:
        def after_draw():
            nonlocal frame,ready_frame
            try:
                frame += 1
                engine.request_full_speed_frame()
                engine.pump_preview_tasks()
                assert frame < 250
                if engine.get_texture_preview_texture_id(control_key):
                    assert tuple(engine.get_texture_preview_size(control_key)) == (8,16)
                    if ready_frame is None:
                        ready_frame = frame
                if ready_frame is not None and frame >= ready_frame + 8:
                    if not reuse_key:
                        assert engine.get_texture_preview_texture_id(key) == 0
                        assert tuple(engine.get_texture_preview_size(key)) == (0,0)
                    engine.exit()
            except BaseException as error:
                failures.append(error)
                engine.exit()
        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert ready_frame is not None
    finally:
        engine.set_post_draw_callback(None)
        engine.release_texture_preview_task(key)
        engine.release_texture_preview_task(control_key)
        engine.set_render_pipeline(None)
        pipeline.dispose()
