"""Public fullscreen state must survive projection without implicit attachment loads."""
import pytest
import infernux as inx

rg = inx.rendergraph


def make_graph(*, initialized=True):
    graph = rg.RenderGraph("Fullscreen raster state")
    graph.create_texture("color", format=rg.Format.RGBA16_SFLOAT)
    graph.create_texture("depth", format=rg.Format.D32_SFLOAT)
    if initialized:
        graph.add_pass("Initialize").write_color("color").write_depth("depth").set_clear(
            color=(0.2, 0.3, 0.4, 1.), depth=1.)
    graph.set_output("color")
    return graph


def test_replacement_defaults_are_unchanged():
    graph = make_graph(initialized=False)
    graph.add_pass("Replacement").write_color("color").fullscreen_quad("Effect")
    command = graph.build().passes[-1].commands[0]
    assert not command.depth_test and not command.depth_write and not command.alpha_blend
    assert command.depth_compare == rg.DepthCompare.ALWAYS


@pytest.mark.parametrize("comparison", [rg.DepthCompare.LESS, rg.DepthCompare.LESS_EQUAL,
                                       rg.DepthCompare.GREATER, rg.DepthCompare.ALWAYS])
@pytest.mark.parametrize("write", [False, True])
def test_depth_blend_state_is_projected_to_native(comparison, write):
    graph = make_graph()
    graph.add_pass("Overlay").write_color("color").write_depth("depth").fullscreen_quad(
        "Effect", depth_test=comparison, depth_write=write, alpha_blend=True)
    native_pass = graph.build().passes[-1]
    assert native_pass.write_depth == "depth"
    assert not native_pass.clear_color and not native_pass.clear_depth
    command = native_pass.commands[0]
    assert command.depth_test and command.depth_compare == comparison
    assert command.depth_write == write and command.alpha_blend


@pytest.mark.parametrize("kwargs", [{"alpha_blend": True}, {"depth_test": rg.DepthCompare.LESS}])
def test_new_attachment_cannot_be_loaded(kwargs):
    graph = make_graph(initialized=False)
    graph.add_pass("Overlay").write_color("color").write_depth("depth").fullscreen_quad("Effect", **kwargs)
    with pytest.raises(ValueError, match="clear or load an earlier output"):
        graph.build()


def test_explicit_clear_initializes_blended_depth_output():
    graph = make_graph(initialized=False)
    graph.add_pass("Overlay").write_color("color").write_depth("depth").set_clear(
        color=(0., 0., 0., 1.), depth=1.).fullscreen_quad(
        "Effect", depth_test=rg.DepthCompare.ALWAYS, depth_write=True, alpha_blend=True)
    native = graph.build().passes[0]
    assert native.clear_color and native.clear_depth


def test_depth_clear_is_independent_of_color_clear():
    graph = make_graph(initialized=False)
    graph.add_pass("Overlay").write_color("color").write_depth("depth").set_clear(
        color=(0., 0., 0., 1.)).fullscreen_quad("Effect", depth_test=rg.DepthCompare.LESS)
    with pytest.raises(ValueError, match="earlier output: depth"):
        graph.build()


def test_depth_write_requires_explicit_comparison():
    graph = make_graph()
    with pytest.raises(ValueError, match="depth_write requires depth_test"):
        graph.add_pass("Overlay").fullscreen_quad("Effect", depth_write=True)


@pytest.mark.parametrize("value", [True, "less", 3])
def test_depth_test_is_not_a_bool_string_or_raw_integer(value):
    with pytest.raises(TypeError, match="DepthCompare"):
        make_graph().add_pass("Overlay").fullscreen_quad("Effect", depth_test=value)


def test_depth_test_requires_attachment():
    graph = make_graph()
    graph.add_pass("Overlay").write_color("color").fullscreen_quad("Effect", depth_test=rg.DepthCompare.LESS)
    with pytest.raises(ValueError, match="requires a depth output"):
        graph.build()


@pytest.mark.parametrize("bind", [False, True])
def test_attached_depth_cannot_also_be_sampled(bind):
    graph = make_graph()
    overlay = graph.add_pass("Overlay").write_color("color").write_depth("depth")
    if bind:
        overlay.set_texture("_Depth", "depth")
    else:
        overlay.read("depth")
    overlay.fullscreen_quad("Effect", depth_test=rg.DepthCompare.ALWAYS, depth_write=True)
    with pytest.raises(ValueError, match="cannot sample its attached depth"):
        graph.build()


def test_copy_preserves_sampleable_old_depth_while_new_depth_is_written():
    graph = make_graph()
    graph.create_texture("previous_depth", format=rg.Format.D32_SFLOAT)
    graph.add_copy_pass("Copy depth").copy_texture("depth", "previous_depth")
    graph.add_pass("Overlay").write_color("color").write_depth("depth").set_texture(
        "_Depth", "previous_depth").fullscreen_quad(
        "Effect", depth_test=rg.DepthCompare.ALWAYS, depth_write=True, alpha_blend=True)
    assert graph.build().passes[-1].commands[0].input_bindings == [("_Depth", "previous_depth")]


def test_fullscreen_state_cannot_leak_into_another_draw_action():
    graph = make_graph()
    render_pass = graph.add_pass("Overlay").write_color("color").write_depth("depth")
    render_pass.fullscreen_quad("Effect", alpha_blend=True)
    render_pass.draw_renderers()
    with pytest.raises(ValueError, match="fullscreen state with another draw action"):
        graph.build()


def test_documented_fullscreen_overlay_uses_public_api():
    from pathlib import Path
    text = (Path(__file__).parents[2] / "docs/learn/rendergraph-advanced.md").read_text(encoding="utf-8")
    section = text.split("{#fullscreen-raster-state}", 1)[1]
    source = section.split("```python\n", 1)[1].split("```", 1)[0]
    graph = make_graph()
    previous = graph.create_texture("previous_depth", format=rg.Format.D32_SFLOAT)
    namespace = dict(graph=graph, scene_color="color", scene_depth="depth", previous_depth=previous)
    exec(compile(source, "rendergraph-advanced.md#fullscreen-raster-state", "exec"), namespace)
    assert graph.build().passes[-1].commands[0].depth_write
