"""Exercise public preview lifetime boundaries with actual GPU presentation."""
import json
import time

from infernux.lib import ConsolePanel
from infernux.renderstack.render_stack_pipeline import RenderStackPipeline
from tests.gpu.preview_settings_case import PreviewSettingsCase


def test_image_preview_preserves_settings_across_load_boundaries(engine, scene, tmp_path):
    pipeline = RenderStackPipeline()
    engine.set_render_pipeline(pipeline)
    case = PreviewSettingsCase(engine, tmp_path/'Images')
    phase = 0
    capture = None
    deadline = time.monotonic() + 20
    results, failures = [], []
    try:
        def tick():
            nonlocal phase, capture
            try:
                assert time.monotonic() < deadline, case.poll()
                if capture is None:
                    ready = case.poll()
                    if ready['done']:
                        capture = engine.request_capture('editor', str(tmp_path/f'preview-{phase}.png'))
                        results.append(ready)
                else:
                    status = engine.query_capture(capture)
                    assert status['status'] not in ('failed', 'cancelled', 'source_expired'), status
                    if status['status'] == 'completed':
                        results[-1]['capture'] = case.observe_capture(tmp_path/f'preview-{phase}.png')
                        phase += 1
                        if phase == len(case.phases):
                            engine.exit()
                        else:
                            case.change(phase)
                            capture = None
            except BaseException as error:
                failures.append(error)
                engine.exit()
        engine.set_post_draw_callback(tick)
        engine.run()
        if failures:
            raise failures[0]
        assert len(results) == len(case.phases)
        console = ConsolePanel()
        assert console.get_error_count() == 0
        assert console.get_warning_count() == 0
    finally:
        engine.set_post_draw_callback(None)
        case.close()
        engine.set_render_pipeline(None)
        pipeline.dispose()
        (tmp_path/'preview-settings.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
