"""Layout contracts exercised with the real ImGui font atlas and renderer."""

from __future__ import annotations

import copy

import pytest

from infernux import lib
from infernux.core.node_graph import (
    NodeGraph, NodeInlineFieldDef, NodeTypeDef, PinDef, PinKind,
)
from infernux.engine.ui.node_graph_view import NodeGraphView


@pytest.mark.parametrize("zoom", [0.3, 0.5, 0.75, 1.0, 1.5, 2.5])
@pytest.mark.parametrize("style", ["graph", "context"])
def test_subtitle_ports_and_body_share_the_rendered_geometry(engine, zoom, style):
    graph = NodeGraph()
    definition = NodeTypeDef(
        "layout", "Long title / 长标题", visual_style=style,
        pins=[
            PinDef("in", "输入 / Input", PinKind.INPUT),
            PinDef("out", "Output", PinKind.OUTPUT),
        ],
        inline_fields=[
            NodeInlineFieldDef("in", "Input", "f32", 1.0),
            NodeInlineFieldDef("mode", "Vector", "vec3", [1.0, 2.0, 3.0]),
            NodeInlineFieldDef("hidden", "Hidden", "f32", 3.0,
                               visible_when_field="mode", visible_when_value=99),
        ],
    )
    graph.register_type(definition)
    plain = graph.add_node("layout", -20.0, -40.0)
    captioned = graph.add_node("layout", 800.0, -40.0, subtitle="Assets/动画/Idle.anim")
    before = copy.deepcopy([node.to_dict() for node in graph.nodes])
    view = NodeGraphView()
    view.graph = graph
    view.zoom = zoom
    samples = []
    errors = []

    class GuiOnlyPipeline(lib.RenderPipelineCallback):
        def render(self, context, camera):
            pass

    class Probe(lib.InxGUIRenderable):
        def on_render(self, ctx):
            ctx.set_next_window_pos(0.0, 0.0, 0, 0.0, 0.0)
            ctx.set_next_window_size(64.0, 64.0, 0)
            visible = ctx.begin_window("Graph layout###graph_layout", True, 0)
            try:
                if visible:
                    view._layout_measure_context = ctx
                    view._compute_layouts()
                    left = view._layouts[plain.uid]
                    right = view._layouts[captioned.uid]
                    # Both draw paths run against the real native draw list.
                    view._draw_one_node(ctx, left)
                    view._draw_one_node(ctx, right)
                    view._draw_minimap(ctx, 0.0, 0.0, 64.0, 64.0)
                    title_width = ctx.calc_text_size(definition.label, view._zoom_font(18.0))[0]
                    pin_width = ctx.calc_text_size(definition.pins[0].label, view._zoom_font(15.0))[0]
                    samples.append((left, right, title_width, pin_width))
            except Exception as error:
                errors.append(error)
            finally:
                ctx.end_window()

    probe = Probe()
    engine.set_render_pipeline(GuiOnlyPipeline())
    engine.register_gui_renderable("test.graph_layout", probe)
    try:
        for _ in range(3):
            engine.tick(1.0 / 60.0)
    finally:
        engine.unregister_gui_renderable("test.graph_layout")
        engine.set_render_pipeline(None)

    assert not errors, errors
    assert samples
    for left, right, title_width, pin_width in samples:
        assert right.h - left.h == pytest.approx(22.0 * zoom if style == "graph" else 0.0)
        assert right.input_pins[0].cy - left.input_pins[0].cy == pytest.approx(right.h - left.h)
        assert right.input_pins[0].cy == pytest.approx(right.output_pins[0].cy)
        assert left.w - 42.0 * zoom >= title_width - 0.01
        assert left.w * 0.45 - 9.0 * zoom >= pin_width - 0.01
        assert left.w * 0.52 - 12.0 * zoom >= 150.0 * zoom - 0.01
        # Hidden fields consume no row, and detached controls sit below ports.
        body_top = view._pin_area_bottom(captioned, definition) * zoom + right.sy
        assert right.input_pins[0].cy < body_top < right.sy + right.h
        assert view._hit_test_pin(right.input_pins[0].cx, right.input_pins[0].cy)[:2] == (captioned.uid, "in")
    assert [node.to_dict() for node in graph.nodes] == before


def test_center_view_uses_visible_fields_and_hanging_control_padding():
    graph = NodeGraph()
    graph.register_type(NodeTypeDef(
        "center", "Center",
        pins=[PinDef("in", "In", PinKind.INPUT)],
        inline_fields=[NodeInlineFieldDef("in", "In", "f32", 1.0)],
    ))
    graph.add_node("center", -300.0, -200.0, subtitle="Metadata")
    view = NodeGraphView()
    view.graph = graph
    view._canvas_w, view._canvas_h = 1000.0, 700.0
    view.center_on_nodes()
    view._compute_layouts()
    layout = next(iter(view._layouts.values()))
    assert layout.sx + (layout.w - layout.left_reserve) * 0.5 == pytest.approx(500.0)
    assert layout.sy + layout.h * 0.5 == pytest.approx(350.0)
