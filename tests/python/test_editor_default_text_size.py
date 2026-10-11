"""Custom panel text inherits the same font size as native editor labels."""
import pytest

from infernux.lib import InxGUIRenderable
from infernux.renderstack import RenderStackPipeline


def test_custom_panel_default_text_matches_native_labels(engine, capfd):
    scales = (1.0, 1.25, 1.75)
    labels = ("0.40", "时间轴", "变换", "预览")
    observations, errors = [], []
    frames = 0

    class Panel(InxGUIRenderable):
        def on_render(self, ctx):
            ctx.set_next_window_pos(0, 0, 1, 0, 0)
            ctx.set_next_window_size(64, 64, 1)
            ctx.begin_window("Default text size regression", True, 0)
            try:
                for scale in scales:
                    ctx.set_window_font_scale(scale)
                    for label in labels:
                        expected_width = ctx.calc_text_width(label)
                        width, height = ctx.calc_text_size(label)
                        # Native ImGui labels round their measured width up.
                        assert width == pytest.approx(expected_width, abs=1.0), (scale, label, width, expected_width)
                        assert ctx.calc_text_size_wrapped(label, wrap_width=1000) == pytest.approx((width, height))
                        ctx.draw_text(8, 30, label, 1, 1, 1, 1)
                        ctx.draw_text_aligned(8, 30, 60, 60, label, 1, 1, 1, 1)
                        ctx.draw_text_rotated_90_aligned(8, 30, 60, 60, label, 1, 1, 1, 1)
                        ctx.draw_text_ex_aligned(8, 30, 60, 60, label, 1, 1, 1, 1)
                        observations.append((scale, label, width, height))
            except BaseException as error:
                errors.append(error)
            finally:
                ctx.set_window_font_scale(1.0)
                ctx.end_window()

    def after_draw():
        nonlocal frames
        frames += 1
        if frames >= 3 or errors:
            engine.exit()

    panel = Panel()
    try:
        engine.set_render_pipeline(RenderStackPipeline())
        engine.register_gui_renderable("default-text-size-test", panel, 0)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        assert not errors, repr(errors)
        assert frames == 3
        assert {(scale, label) for scale, label, *_ in observations} == {
            (scale, label) for scale in scales for label in labels
        }
    finally:
        engine.set_post_draw_callback(None)
        engine.unregister_gui_renderable("default-text-size-test")
        engine.set_render_pipeline(None)
    captured = capfd.readouterr()
    assert not any(token in captured.out + captured.err for token in (
        "VUID-", "SYNC-HAZARD", "Validation Error",
    )), captured.out + captured.err
