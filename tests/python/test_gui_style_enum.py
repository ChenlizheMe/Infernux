"""Editor style indices must follow the ImGui compiled into the native module."""
import re
from pathlib import Path

from infernux.engine.ui.theme import ImGuiStyleVar


def test_every_style_variable_matches_vendored_imgui():
    header = Path(__file__).resolve().parents[2] / "external/imgui_for_infernux/imgui.h"
    source = header.read_text(encoding="utf-8")
    body = source.split("enum ImGuiStyleVar_\n{", 1)[1].split("ImGuiStyleVar_COUNT", 1)[0]
    names = re.findall(r"^\s+ImGuiStyleVar_(\w+),", body, re.MULTILINE)
    assert len(names) >= 41
    mismatches = [(name, index, getattr(ImGuiStyleVar, name, None))
                  for index, name in enumerate(names)
                  if getattr(ImGuiStyleVar, name, None) != index]
    assert not mismatches


def test_style_indices_cross_the_actual_python_gui_binding(engine):
    from infernux import lib
    from infernux.renderstack import RenderStackPipeline

    # Check this before rendering: passing a scalar index to the vec2 ImGui API
    # would assert in a debug build, not merely fail a Python assertion.
    assert ImGuiStyleVar.ButtonTextAlign == 35
    assert ImGuiStyleVar.ImageBorderSize == 24
    observed = {"renders": 0, "frames": 0, "error": None}

    class Probe(lib.InxGUIRenderable):
        def on_render(self, ctx):
            visible = ctx.begin_window("Style indices###style_contract", True, 0)
            try:
                if visible:
                    ctx.push_style_var_float(ImGuiStyleVar.ImageRounding, 3.)
                    ctx.push_style_var_float(ImGuiStyleVar.ImageBorderSize, 2.)
                    ctx.push_style_var_vec2(ImGuiStyleVar.ButtonTextAlign, .25, .75)
                    ctx.push_style_var_vec2(ImGuiStyleVar.SeparatorTextPadding, 3., 5.)
                    try:
                        ctx.button("Styled button", lambda: None)
                    finally:
                        ctx.pop_style_var(4)
                    observed["renders"] += 1
            except Exception as error:
                observed["error"] = error
            finally:
                ctx.end_window()

    def update(_delta):
        observed["frames"] += 1
        if observed["frames"] >= 5:
            engine.exit()

    probe = Probe()
    engine.register_gui_renderable("test.style_enum", probe)
    try:
        # The graphical fixture also renders its Scene view. Give it a valid
        # pipeline while checking GUI styles, rather than emitting unrelated
        # missing-pipeline errors on every frame.
        engine.set_render_pipeline(RenderStackPipeline())
        engine.set_pre_scene_update_callback(update)
        engine.run()
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.unregister_gui_renderable("test.style_enum")
        engine.set_render_pipeline(None)
    assert observed["error"] is None, repr(observed["error"])
    assert observed["renders"] > 0
