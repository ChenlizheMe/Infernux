"""Authored custom UI draws publish new intrinsic sizes to native GUI frames."""
import pytest


def test_custom_text_edit_matches_native_font_draw_and_hitbox(engine, scene, monkeypatch, capfd):
    from infernux.engine import runtime_screen_ui as runtime
    from infernux.engine.runtime_screen_ui_pipeline import RuntimeScreenUIRenderPipeline
    from infernux.renderstack import RenderStackPipeline
    from infernux.ui import UICanvas, UIButton, UIText, UIFrame, UILayoutDirection, TextResizeMode
    from infernux.ui import ui_render_dispatch as dispatch
    from infernux.ui.ui_font_asset import ui_font_paths

    def add(cls, parent):
        owner = scene.create_game_object(cls.__name__)
        owner.set_parent(parent)
        return owner.add_py_component(cls())

    root = scene.create_game_object("Measured UI")
    canvas = root.add_py_component(UICanvas())
    canvas.reference_width, canvas.reference_height = 256, 128
    row = add(UIFrame, root)
    row.width, row.height = 240, 80
    row.layout_direction, row.clip_content = UILayoutDirection.Horizontal, True
    custom = add(UIButton, row.game_object)
    custom.width = 24
    target = add(UIText, row.game_object)
    target.text, target.font_size = "A", 18
    target.resize_mode, target.raycast_target = TextResizeMode.AutoWidth, True
    tail = add(UIText, row.game_object)
    tail.text, tail.width = "end", 30
    scene.create_game_object("UI camera").add_component("Camera")

    original_custom = dispatch.get_ui_renderer("UIButton", "runtime")
    observed, errors = [], []
    original_dispatch = runtime._ui_dispatch

    def inspect(element, *args, **kwargs):
        if element is target:
            observed.append((target.text, kwargs["sw"]))
        return original_dispatch(element, *args, **kwargs)

    monkeypatch.setattr(runtime, "_ui_dispatch", inspect)
    dispatch.register_ui_renderer("UIButton", "runtime", lambda *args, **kwargs: setattr(target, "text", "MMMMMM"))
    submission = runtime.RuntimeScreenUISubmission(engine)
    submission.set_target_size(256, 128)
    renderer = engine.get_screen_ui_renderer()
    frames = 0

    def after_draw():
        nonlocal frames
        try:
            frames += 1
            assert frames <= 4
            font, fallbacks = ui_font_paths(target)
            expected = renderer.measure_text("MMMMMM", 18, 0, font, 1.2, 0, fallbacks)[0]
            original_width = renderer.measure_text("A", 18, 0, font, 1.2, 0, fallbacks)[0]
            assert expected > 3 * original_width > 0
            assert observed and observed[-1][0] == "MMMMMM"
            assert observed[-1][1] == pytest.approx(expected, abs=.01)
            assert target.get_resolved_size()[0] == pytest.approx(expected, abs=.01)
            x, y, width, height = target.get_rect(256, 128)
            assert canvas.raycast(x + width - 2, y + height / 2) is target
            if frames == 3:
                engine.exit()
        except BaseException as error:
            errors.append(error)
            engine.exit()

    try:
        engine.resize_game_render_target(256, 128)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(RuntimeScreenUIRenderPipeline(submission, RenderStackPipeline()))
        engine.set_post_draw_callback(after_draw)
        engine.run()
        assert not errors, repr(errors)
        assert frames == 3
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        # The session renderer outlives this disposable Scene. Retire its
        # retained UI snapshot before another test installs a bare pipeline.
        renderer.begin_frame(256, 128)
        dispatch.register_ui_renderer("UIButton", "runtime", original_custom)
    captured = capfd.readouterr()
    assert not any(token in captured.out + captured.err for token in (
        "VUID-", "SYNC-HAZARD", "Validation Error", "Screen UI packet outlived",
    ))
