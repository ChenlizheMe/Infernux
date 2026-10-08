"""Button alignment positions actual GPU glyph coverage inside its box."""
import json

import pytest

from infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission
from infernux.engine.runtime_screen_ui_pipeline import RuntimeScreenUIRenderPipeline
from infernux.renderstack import RenderStackPipeline
from infernux.ui import UIButton, UICanvas, TextAlignV


@pytest.mark.parametrize('label', ('Play', '开始游戏'))
def test_button_top_center_bottom_align_visible_text(engine, scene, capfd, label):
    root = scene.create_game_object('Button alignment canvas')
    canvas = root.add_py_component(UICanvas())
    canvas.reference_width, canvas.reference_height = 256, 128
    owner = scene.create_game_object('Aligned button')
    owner.set_parent(root)
    button = owner.add_py_component(UIButton())
    button.label = label
    button.width, button.height = 160, 80
    button.font_size, button.line_height = 32.22, 1.2
    button.background_color = (.1, .2, .3, 1)
    button.label_color = (1, 1, 1, 1)
    alignments = (TextAlignV.Top, TextAlignV.Center, TextAlignV.Bottom)
    button.text_align_v = alignments[0]
    scene.create_game_object('Button alignment camera').add_component('Camera')
    engine.resize_game_render_target(256, 128)
    submission = RuntimeScreenUISubmission(engine)
    submission.set_target_size(256, 128)
    state = {'frames': 0, 'phase': 0, 'ticket': None}
    observations, errors = [], []

    def after_draw():
        try:
            state['frames'] += 1
            assert state['frames'] <= 60
            if state['ticket'] is None:
                # Owner maintenance callbacks also run during MSAA rebuilds,
                # before the new target has received its first rendered frame.
                if not engine.renderer_frame_snapshot['game_render_graph_current_executed']:
                    return
                state['ticket'] = engine.request_render_target_readback(True)
                return
            ticket = state['ticket']
            if not ticket.done:
                return
            assert not ticket.error, ticket.error
            pixels = ticket.result_numpy()
            assert pixels.shape == (128, 256, 4) and str(pixels.dtype) == 'float16'
            rows = (pixels[:, :, :3] > .85).all(axis=2).nonzero()[0]
            assert rows.size, 'Button label produced no white glyph coverage'
            x, y, width, height = button.get_rect(256, 128)
            min_y, max_y = int(rows.min()), int(rows.max()) + 1
            phase = state['phase']
            actual = (min_y, (min_y + max_y) / 2, max_y)[phase]
            expected = y + height * (0, .5, 1)[phase]
            observation = dict(label=label, alignment=str(alignments[phase]),
                               button_rect=(x, y, width, height), glyph_rows=(min_y, max_y),
                               actual_anchor=actual, expected_anchor=expected)
            observations.append(observation)
            assert abs(actual - expected) <= 2.0, observation
            state['ticket'] = None
            state['phase'] += 1
            if state['phase'] == len(alignments):
                engine.exit()
            else:
                button.text_align_v = alignments[state['phase']]
        except BaseException as error:
            errors.append(error)
            engine.exit()

    try:
        # Exercise the attachment rebuild independently of previous GPU tests.
        engine.set_msaa_samples(1)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(RuntimeScreenUIRenderPipeline(submission, RenderStackPipeline()))
        engine.set_post_draw_callback(after_draw)
        engine.run()
        assert not errors, repr(errors)
        assert state['phase'] == 3
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        engine.get_screen_ui_renderer().begin_frame(256, 128)
    captured = capfd.readouterr()
    assert not any(token in captured.out + captured.err for token in (
        'VUID-', 'SYNC-HAZARD', 'Validation Error', 'Screen UI packet outlived',
        'Validation layers requested but not available',
    )), captured.out + captured.err
    print(json.dumps(observations, ensure_ascii=False))
