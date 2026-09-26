"""World UI stage filtering keeps the native Camera/depth contract intact."""
import numpy as np
import pytest
import infernux as inx

rg = inx.rendergraph
ALL = 0xffffffff
LABELS = 1 << 30


def make_graph():
    graph = rg.RenderGraph("World UI stages")
    graph.create_texture("color", format=rg.Format.RGBA16_SFLOAT)
    graph.create_texture("depth", format=rg.Format.D32_SFLOAT)
    graph.add_pass("Opaque").write_color("color").write_depth("depth").set_clear(
        color=(0., 0., 0., 1.), depth=1.).draw_renderers()
    graph.set_output("color")
    return graph


@pytest.mark.parametrize("mask", [0, 1, LABELS, 1 << 31, ALL, np.uint32(LABELS)])
def test_mask_projects_to_native_without_signed_truncation(mask):
    graph = make_graph()
    graph.add_pass("Labels").write_color("color").write_depth("depth").draw_world_ui(layer_mask=mask)
    desc = graph.build().passes[-1]
    assert desc.commands[0].world_ui_layer_mask == int(mask)
    assert desc.write_depth == "depth"
    assert not desc.clear_color and not desc.clear_depth


@pytest.mark.parametrize("mask", [True, False, 1.5, "1", None, [], {}])
def test_mask_rejects_non_integer_input_without_changing_action(mask):
    graph = make_graph()
    draw = graph.add_pass("Labels").write_color("color").write_depth("depth")
    draw.draw_world_ui(layer_mask=LABELS)
    with pytest.raises(TypeError):
        draw.draw_world_ui(layer_mask=mask)
    assert graph.build().passes[-1].commands[0].world_ui_layer_mask == LABELS


@pytest.mark.parametrize("mask", [-1, 1 << 32])
def test_mask_rejects_out_of_range_input(mask):
    with pytest.raises(ValueError, match="unsigned 32-bit"):
        make_graph().add_pass("Labels").draw_world_ui(layer_mask=mask)


@pytest.mark.parametrize("helper", ["camera_ui_section", "screen_ui_section"])
def test_section_filters_only_world_ui(helper):
    graph = make_graph()
    getattr(graph, helper)(world_ui_layer_mask=ALL ^ LABELS)
    passes = {p.name: p for p in graph.build().passes}
    assert passes["_WorldUI"].commands[0].world_ui_layer_mask == ALL ^ LABELS
    assert passes["_ScreenUI_Camera"].commands[0].world_ui_layer_mask == ALL
    assert passes["Opaque"].commands[0].world_ui_layer_mask == ALL


def test_default_still_draws_all_camera_visible_layers():
    graph = make_graph()
    graph.screen_ui_section()
    world = next(p for p in graph.build().passes if p.name == "_WorldUI")
    assert world.commands[0].world_ui_layer_mask == ALL


def test_predeclared_world_pass_keeps_its_authored_filter():
    graph = make_graph()
    graph.add_pass("_WorldUI").write_color("color").write_depth("depth").draw_world_ui(layer_mask=1)
    graph.screen_ui_section(world_ui_layer_mask=LABELS)
    world = [p for p in graph.build().passes if p.name == "_WorldUI"]
    assert len(world) == 1 and world[0].commands[0].world_ui_layer_mask == 1


def test_replacing_action_does_not_leak_world_filter():
    graph = make_graph()
    draw = graph.add_pass("Changed").write_color("color").write_depth("depth")
    draw.draw_world_ui(layer_mask=LABELS).draw_renderers()
    assert graph.build().passes[-1].commands[0].world_ui_layer_mask == ALL
    draw.draw_world_ui()
    assert graph.build().passes[-1].commands[0].world_ui_layer_mask == ALL


def test_late_ui_uses_authored_depth_after_effects_and_before_display_encoding():
    graph = make_graph()
    snapshot = graph.create_texture("label_depth", format=rg.Format.D32_SFLOAT)
    graph.add_copy_pass("DepthSnapshot").copy_texture("depth", snapshot)
    graph.camera_ui_section(world_ui_layer_mask=ALL ^ LABELS)
    graph.add_pass("ProjectGrade").write_color("color").fullscreen_quad("Grade")
    graph.add_pass("LateLabels").write_color("color").write_depth(snapshot).draw_world_ui(layer_mask=LABELS)
    graph.screen_ui_overlay_section()
    desc = graph.build()
    names = [p.name for p in desc.passes]
    assert names.index("_WorldUI") < names.index("ProjectGrade") < names.index("LateLabels")
    assert names.index("LateLabels") < names.index("_DisplayEncode") < names.index("_ScreenUI_Overlay")
    early = desc.passes[names.index("_WorldUI")]
    late = desc.passes[names.index("LateLabels")]
    assert early.commands[0].world_ui_layer_mask & late.commands[0].world_ui_layer_mask == 0
    assert early.write_depth == "depth" and late.write_depth == "label_depth"
    assert not late.clear_depth and not late.clear_color


def test_missing_depth_is_not_implicitly_repaired_by_projection():
    graph = make_graph()
    graph.add_pass("Labels").write_color("color").draw_world_ui(layer_mask=LABELS)
    # Attachment validity is enforced by the native graph compiler.
    assert graph.build().passes[-1].write_depth == ""
