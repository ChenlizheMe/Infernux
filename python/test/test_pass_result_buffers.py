from __future__ import annotations

import pytest

from Infernux.rendergraph.graph import Format, RenderGraph
from Infernux.renderstack.geometry_buffers import (
    GeometryBufferTopologyError,
    geometry_buffer,
)
from Infernux.renderstack.render_pipeline import RenderPipeline
from Infernux.renderstack.pipeline_dsl import PipelineBuilder


def test_pass_result_write_preserves_parent_revision():
    graph = RenderGraph("Pass Result Revisions")
    color_before = graph.create_texture("color_before", format=Format.RGBA16_SFLOAT)
    color_after = graph.create_texture("color_after", format=Format.RGBA16_SFLOAT)
    opaque = graph.publish_pass_result("opaque", {"color": color_before})
    opacity = graph.write_buffer("opacity", opaque, "color", color_after)

    assert opaque.sample("color") is color_before
    assert opacity.sample("color") is color_after
    assert opacity.revision > opaque.revision


def test_pass_resources_reject_foreign_graph_handles_even_with_matching_names():
    view = RenderGraph("Scene View")
    other_view = RenderGraph("Game View")
    color = view.create_texture("color", camera_target=True)
    depth = view.create_texture("depth", format=Format.D32_SFLOAT)
    foreign_color = other_view.create_texture("color", camera_target=True)
    foreign_depth = other_view.create_texture("depth", format=Format.D32_SFLOAT)
    assert foreign_color == color  # Public name equality cannot prove ownership.

    with pytest.raises(ValueError, match="does not belong"):
        view.publish_pass_result("foreign", {"color": foreign_color})
    opaque = view.publish_pass_result("opaque", {"color": color, "depth": depth})
    with pytest.raises(ValueError, match="does not belong"):
        view.derive_pass_result("after", opaque, {"depth": foreign_depth})
    with pytest.raises(ValueError, match="does not belong"):
        view.derive_pass_result("after", other_view.publish_pass_result("opaque", {"color": foreign_color}), {})
    with pytest.raises(ValueError, match="does not belong"):
        opaque._publish_materialized("normal", foreign_color)
    with pytest.raises(ValueError, match="does not belong"):
        view.set_output(foreign_color)
    with pytest.raises(ValueError, match="does not belong"):
        with view.add_pass("Read Opaque") as render_pass:
            render_pass.set_texture("_SceneDepth", foreign_depth)

    with view.add_pass("Read Local Opaque") as render_pass:
        render_pass.set_texture("_SceneDepth", opaque.sample("depth"))
        render_pass.write_color(color)
    assert render_pass._input_bindings == {"_SceneDepth": "depth"}

    local_buffer = view.create_buffer("lighting", 64)
    foreign_buffer = other_view.create_buffer("lighting", 64)
    with pytest.raises(ValueError, match="does not belong"):
        view.add_pass("Read Foreign Lighting").read_buffer(foreign_buffer)
    view.add_pass("Read Local Lighting").read_buffer(local_buffer)
    with pytest.raises(ValueError, match="does not belong"):
        view.append_pass(other_view.add_pass("Foreign Pass"))


def test_custom_geometry_buffer_dependencies_are_topologically_sorted():
    order = []

    class IndexPipeline(RenderPipeline):
        @geometry_buffer("classification", dependencies={"index"})
        def classification(self, context):
            order.append("classification")
            return context.graph.create_texture(
                "classification", format=Format.RGBA8_UNORM
            )

        @geometry_buffer("index", dependencies={"depth"})
        def index(self, context):
            order.append("index")
            return context.graph.create_texture("index", format=Format.RG32_UINT)

    graph = RenderGraph("Custom Geometry Buffer")
    pipeline = IndexPipeline()
    pipeline._defining_graph = graph
    pipeline.require_buffer("classification")
    result = pipeline.geometry_stage(
        graph,
        "opaque",
        buffers={
            "depth": graph.create_texture("depth", format=Format.D32_SFLOAT)
        },
        queue_range=(0, 2500),
    )

    assert result.has("index")
    assert result.has("classification")
    assert order == ["index", "classification"]


def test_geometry_buffer_cycle_reports_source_and_dependency_chain():
    class CyclePipeline(RenderPipeline):
        @geometry_buffer("a", dependencies={"b"})
        def a(self, context):
            raise AssertionError("cycle must fail before provider execution")

        @geometry_buffer("b", dependencies={"a"})
        def b(self, context):
            raise AssertionError("cycle must fail before provider execution")

    graph = RenderGraph("Cycle")
    pipeline = CyclePipeline()
    pipeline._defining_graph = graph
    pipeline.require_buffer("a")
    with pytest.raises(
        GeometryBufferTopologyError,
        match=r"source 'opaque'.*a -> b -> a",
    ):
        pipeline.geometry_stage(
            graph,
            "opaque",
            buffers={},
            queue_range=(0, 2500),
        )


def test_builtin_geometry_stage_only_materializes_requested_buffers():
    graph = RenderGraph("Demand Driven Geometry")
    pipeline = RenderPipeline.__new__(RenderPipeline)
    RenderPipeline.__init__(pipeline)
    pipeline._defining_graph = graph
    graph.set_geometry_buffer_requirements({"normal"})
    result = pipeline.geometry_stage(
        graph,
        "opaque",
        buffers={
            "color": graph.create_texture("color", format=Format.RGBA16_SFLOAT),
            "depth": graph.create_texture("depth", format=Format.D32_SFLOAT),
        },
        queue_range=(0, 2500),
    )

    assert result.has("normal")
    assert not result.has("motion")
    assert not result.has("base_color")
    assert [p._material_pass for p in graph._passes] == ["normal"]


def test_declarative_pipeline_propagates_effect_result_revision():
    builder = PipelineBuilder()
    builder.opaque().forward()
    builder.effects("stylized")

    class DeclarativePipeline(RenderPipeline):
        def define(self, pipeline):
            raise AssertionError("definition is supplied directly by this test")

    pipeline = DeclarativePipeline()
    graph = RenderGraph("Effect Revision")
    pipeline._defining_graph = graph

    def replace_color(stage):
        current = graph.current_pass_result
        assert current is not None
        changed = graph.create_texture(
            f"{stage.stable_id}_color", format=Format.RGBA16_SFLOAT
        )
        graph.replace_current_pass_result(
            graph.write_buffer(f"effect:{stage.stable_id}", current, "color", changed)
        )

    graph._effect_stage_callback = replace_color
    from Infernux.renderstack.pipeline_compiler import compile_pipeline_definition

    compile_pipeline_definition(builder.build(), graph, pipeline=pipeline)
    effect_result = graph.get_pass_result("effect:stylized")
    assert effect_result is not None
    assert effect_result.sample("color").name == "stylized_color"
