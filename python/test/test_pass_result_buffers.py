from __future__ import annotations

import pytest

from infernux.rendergraph.graph import Format, RenderGraph
from infernux.renderstack.geometry_buffers import (
    GeometryBufferTopologyError,
    geometry_buffer,
)
from infernux.renderstack.render_pipeline import RenderPipeline
from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline
from infernux.renderstack.default_forward_plus_pipeline import DefaultForwardPlusPipeline
from infernux.renderstack.default_deferred_pipeline import DefaultDeferredPipeline
from infernux.renderstack.fullscreen_effect import FullScreenEffect
from infernux.renderstack.pipeline_dsl import PipelineBuilder
from infernux.renderstack.resource_bus import ResourceBus


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


@pytest.mark.parametrize(
    ("pipeline_type", "producer"),
    [(DefaultForwardPipeline, "opaque"), (DefaultDeferredPipeline, "gbuffer")],
)
def test_builtin_pipeline_publishes_view_shadow_map_to_effects(pipeline_type, producer):
    graph = RenderGraph("Shadow Consumer")
    observed = {}

    def capture_stage(stage):
        if stage.stable_id == "after_opaque":
            observed["result"] = graph.current_pass_result
            observed["inputs"] = stage.contract.inputs

    graph._effect_stage_callback = capture_stage
    pipeline_type().define_topology(graph)
    result = graph.get_pass_result(producer)
    assert result.sample("shadow_map") is graph.get_texture("shadow_map")
    assert "shadow_map" in observed["inputs"]
    assert observed["result"].sample("shadow_map") is result.sample("shadow_map")
    with graph.add_pass("Sample Shadow") as render_pass:
        render_pass.set_texture("shadowMap", result.sample("shadow_map"))
        render_pass.write_color(graph.create_texture("shadow_preview"))
        render_pass.fullscreen_quad("Shadow Preview")
    assert render_pass._input_bindings == {"shadowMap": "shadow_map"}


def test_deferred_public_normal_is_not_drawn_without_a_consumer():
    graph = RenderGraph("Deferred without normal consumer")
    DefaultDeferredPipeline().define_topology(graph)
    graph.build()

    gbuffer = graph.get_pass_result("gbuffer")
    opaque = graph.get_pass_result("opaque_lighting")
    assert graph.get_texture("gbuffer_normal").format == Format.RGBA16_SFLOAT
    assert not gbuffer.has("normal")
    assert not opaque.has("normal")
    assert not any(p._material_pass == "normal" for p in graph._passes)
    assert graph.get_texture("_result/gbuffer/normal") is None
    assert graph.get_texture("_result/opaque_lighting/normal") is None


def test_deferred_public_normal_preserves_coverage_and_effect_revision():
    graph = RenderGraph("Deferred normal coverage")
    graph.set_geometry_buffer_requirements({"normal"})
    authored_normal = None

    def replace_gbuffer_normal(stage):
        nonlocal authored_normal
        if stage.stable_id != "after_gbuffer":
            return
        current = graph.current_pass_result
        authored_normal = graph.create_texture(
            "authored_normal", format=Format.RGBA16_SFLOAT
        )
        with graph.add_pass("AuthorNormal") as render_pass:
            render_pass.set_texture("_SourceTex", current.sample("normal"))
            render_pass.write_color(authored_normal)
            render_pass.fullscreen_quad("Fullscreen Blit")
        graph.replace_current_pass_result(
            graph.write_buffer("effect:normal", current, "normal", authored_normal)
        )

    graph._effect_stage_callback = replace_gbuffer_normal
    DefaultDeferredPipeline().define_topology(graph)
    graph.build()

    gbuffer = graph.get_pass_result("gbuffer")
    opaque = graph.get_pass_result("opaque_lighting")
    packed = graph.get_texture("gbuffer_normal")
    public_gbuffer = gbuffer.sample("normal")
    public_opaque = opaque.sample("normal")
    assert packed is not public_gbuffer
    assert public_gbuffer.name == "_result/gbuffer/normal"
    assert public_opaque.name == "_result/opaque_lighting/normal"
    assert public_opaque is not public_gbuffer
    assert public_opaque is not authored_normal
    assert all(texture.format == Format.RGBA16_SFLOAT for texture in (
        packed, public_gbuffer, public_opaque
    ))

    passes = {render_pass.name: render_pass for render_pass in graph._passes}
    names = list(passes)
    assert names.index("GBufferPass") < names.index("gbuffer/Normal")
    assert names.index("gbuffer/Normal") < names.index("DeferredLightingPass")
    assert names.index("DeferredForwardFallbackPass") < names.index(
        "opaque_lighting/NormalBase"
    ) < names.index("opaque_lighting/Normal")
    assert names.index("opaque_lighting/Normal") < names.index("SkyboxPass")

    first = passes["gbuffer/Normal"]
    assert first._material_filter == "deferred_compatible"
    assert first._reads == ["depth"]
    assert first._write_colors == {0: public_gbuffer.name}
    assert first._write_depth is None
    assert first._clear_color == (0.5, 0.5, 1.0, 0.0)

    copy = passes["opaque_lighting/NormalBase"]
    assert copy._source_resource == authored_normal.name
    assert copy._destination_resource == public_opaque.name
    second = passes["opaque_lighting/Normal"]
    assert second._material_filter == "deferred_unsupported"
    assert second._reads == ["depth"]
    assert second._write_colors == {0: public_opaque.name}
    assert second._write_depth is None
    assert second._clear_color is None

    from pathlib import Path

    template = Path(
        "python/infernux/resources/shaders/_templates/surface_main_normal.glsl"
    ).read_text(encoding="utf-8")
    packed_template = Path(
        "python/infernux/resources/shaders/_templates/default_gbuffer_evaluate.glsl"
    ).read_text(encoding="utf-8")
    deferred_lighting = Path(
        "python/infernux/resources/shaders/deferred_lighting.frag"
    ).read_text(encoding="utf-8")
    assert "s.alpha < material._AlphaClipThreshold) discard;" in template
    assert "outNormal = vec4(normalWS * 0.5 + 0.5, 1.0);" in template
    assert "gbuf1 = vec4(normalize(s.normalWS) * 0.5 + 0.5, s.smoothness);" in packed_template
    assert "surfaceData.smoothness = normalData.a;" in deferred_lighting


@pytest.mark.parametrize("pipeline_type", [DefaultForwardPipeline, DefaultForwardPlusPipeline])
@pytest.mark.parametrize("samples", [1, 4])
def test_forward_pass_buffers_have_current_view_producers_and_resolved_inputs(pipeline_type, samples):
    graph = RenderGraph("View Pass Buffers")
    graph.set_geometry_buffer_requirements({"normal", "motion"})
    observed = {}

    def capture_stage(stage):
        if stage.stable_id == "after_opaque":
            observed["result"] = graph.current_pass_result
            observed["inputs"] = stage.contract.inputs

    graph._effect_stage_callback = capture_stage
    pipeline = pipeline_type()
    pipeline.msaa_samples = samples
    pipeline.define_topology(graph)
    graph.build()

    result = observed["result"]
    assert {"color", "depth", "normal", "motion"} <= set(observed["inputs"])
    assert result.sample("depth") is graph.get_texture("depth")
    assert result.sample("normal").samples == 1
    assert result.sample("motion").samples == 1
    assert result.sample("normal") is not result.sample("motion")

    opaque_index = next(i for i, p in enumerate(graph._passes) if p._name == "OpaquePass")
    for semantic, material_pass in (("normal", "normal"), ("motion", "motion")):
        producer_index, producer = next(
            (i, p) for i, p in enumerate(graph._passes) if p._material_pass == material_pass
        )
        assert producer_index > opaque_index
        assert producer._reads == ["depth"]
        assert producer._write_depth is None
        if samples == 4:
            assert producer._resolve_color == result.sample(semantic).name
            assert graph.get_texture(producer._write_colors[0]).samples == 4
        else:
            assert producer._resolve_color is None
            assert producer._write_colors[0] == result.sample(semantic).name

    class RecordingPass:
        def __init__(self):
            self.bindings = {}

        def set_texture(self, name, handle):
            self.bindings[name] = handle

    class PassBufferEffect(FullScreenEffect):
        name = "pass_buffer_effect"
        injection_point = "after_opaque"
        requires = {"color", "depth", "normal", "motion"}
        modifies = set()

    render_pass = RecordingPass()
    PassBufferEffect().bind_buffers(render_pass, ResourceBus(result.snapshot))
    assert render_pass.bindings == {
        "_InxPassColor": result.sample("color"),
        "_InxPassDepth": result.sample("depth"),
        "_InxPassNormal": result.sample("normal"),
        "_InxPassMotion": result.sample("motion"),
    }


@pytest.mark.parametrize("pipeline_type", [DefaultForwardPipeline, DefaultForwardPlusPipeline, DefaultDeferredPipeline])
def test_light_list_is_a_view_owned_read_only_buffer(pipeline_type):
    graph = RenderGraph(f"{pipeline_type.__name__} light list")
    graph.set_geometry_buffer_requirements({"light_list"})
    pipeline_type().define_topology(graph)

    source = "gbuffer" if pipeline_type is DefaultDeferredPipeline else "opaque"
    result = graph.get_pass_result(source)
    assert result is not None and result.has("light_list")
    light_list = result.sample("light_list")
    assert light_list is graph.get_buffer("light_list")
    assert light_list.view_light_list is True
    assert light_list.compute_buffer is None

    class LightEffect(FullScreenEffect):
        name = "light_list_effect"
        injection_point = "after_opaque"
        requires = {"light_list"}
        modifies = set()

    render_pass = graph.add_pass("Read Light List")
    LightEffect().bind_buffers(render_pass, ResourceBus(result.snapshot, graph=graph))
    assert render_pass._input_bindings == {"lightList": light_list.name}
    assert render_pass._buffer_accesses == [(light_list.name, "storage_read")]


def test_fullscreen_effect_binds_textures_and_extra_buffers_with_typed_apis():
    graph = RenderGraph("Mixed Effect Inputs")
    color = graph.create_texture("color")
    depth = graph.create_texture("depth", format=Format.D32_SFLOAT)
    lights = graph.create_view_light_list()
    extra = graph.create_buffer("extra_values", 16)
    bus = ResourceBus({"color": color, "depth": depth, "light_list": lights}, graph=graph)

    class MixedEffect(FullScreenEffect):
        name = "Mixed Effect"
        injection_point = "before_post_process"
        requires = {"depth", "light_list"}

    render_pass = graph.add_pass("Read Mixed Inputs")
    MixedEffect().bind_buffers(render_pass, bus, extra_bindings={"values": extra})

    assert render_pass._input_bindings == {
        "_InxPassColor": color.name,
        "_InxPassDepth": depth.name,
        "lightList": lights.name,
        "values": extra.name,
    }
    assert render_pass._buffer_accesses == [(lights.name, "storage_read"), (extra.name, "storage_read")]
    assert set(render_pass._reads) == {color.name, depth.name}


@pytest.mark.parametrize(
    ("pipeline_type", "terminal_source", "samples"),
    [
        (DefaultForwardPipeline, "opaque", 1),
        (DefaultForwardPipeline, "opaque", 4),
        (DefaultForwardPlusPipeline, "opaque", 1),
        (DefaultForwardPlusPipeline, "opaque", 4),
        (DefaultDeferredPipeline, "gbuffer", 1),
    ],
)
def test_all_builtin_pipelines_publish_same_view_normal_motion_depth(
    pipeline_type, terminal_source, samples
):
    """Normal/motion/depth must be one source-scoped View contract.

    Forward, Forward+ and Deferred use different geometry routes, but a
    consumer at the opaque stage must receive buffers produced for that same
    graph/View.  In particular, Deferred's private GBuffer normal is not
    exposed as the public normal resource and both derived buffers read the
    graph's depth attachment.
    """
    graph = RenderGraph(f"{pipeline_type.__name__} View contract", output_samples=samples)
    graph.set_geometry_buffer_requirements({"normal", "motion"})
    pipeline_type().define_topology(graph)

    result = graph.get_pass_result(terminal_source)
    assert result is not None
    assert {"color", "depth", "normal", "motion"} <= set(result.snapshot)
    assert result.sample("depth") is graph.get_texture("depth")
    assert result.sample("normal") is not result.sample("motion")
    assert result.sample("normal").samples == 1
    assert result.sample("motion").samples == 1
    assert result.sample("normal").name.startswith(f"_result/{terminal_source}/normal")
    assert result.sample("motion").name.startswith(f"_result/{terminal_source}/motion")

    passes = graph._passes
    for semantic, material_pass in (("normal", "normal"), ("motion", "motion")):
        producer = next(item for item in passes if item._material_pass == material_pass)
        assert producer._reads == ["depth"]
        assert producer._write_depth is None
        if samples > 1:
            assert producer._resolve_color == result.sample(semantic).name
        else:
            assert producer._write_colors[0] == result.sample(semantic).name


def test_view_geometry_results_do_not_cross_graph_boundaries():
    """Two camera/View graph builds keep equal aliases but distinct resources."""
    graphs = []
    for label in ("Scene View", "Game View"):
        graph = RenderGraph(label)
        graph.set_geometry_buffer_requirements({"normal", "motion"})
        DefaultDeferredPipeline().define_topology(graph)
        graphs.append(graph)

    scene_result = graphs[0].get_pass_result("gbuffer")
    game_result = graphs[1].get_pass_result("gbuffer")
    assert scene_result is not None and game_result is not None
    for semantic in ("depth", "normal", "motion"):
        assert scene_result.sample(semantic) is not game_result.sample(semantic)
        assert scene_result.sample(semantic).name == game_result.sample(semantic).name


def test_fullscreen_effect_shadow_binding_reports_missing_source():
    class ShadowEffect(FullScreenEffect):
        name = "shadow_sample"
        injection_point = "after_opaque"
        requires = {"shadow_map"}
        modifies = set()

    class RecordingPass:
        def __init__(self):
            self.bindings = {}

        def set_texture(self, name, handle):
            self.bindings[name] = handle

    shadow = object()
    render_pass = RecordingPass()
    ShadowEffect().bind_buffers(render_pass, ResourceBus({"shadow_map": shadow}))
    assert render_pass.bindings == {"shadowMap": shadow}
    with pytest.raises(RuntimeError, match="unavailable buffers.*shadow_map"):
        ShadowEffect().bind_buffers(render_pass, ResourceBus())


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


@pytest.mark.parametrize("declaration", ["graph", "pipeline"])
@pytest.mark.parametrize("phases", [("opaque", "transparent"), ("transparent", "opaque")])
def test_geometry_provider_dependencies_stay_in_their_phase(declaration, phases):
    calls = []

    class PhasePipeline(RenderPipeline):
        @geometry_buffer("phase_color", phase="opaque", dependencies={"opaque_input"})
        def opaque_provider(self, context):
            raise AssertionError("the derived opaque provider must replace this one")

        @geometry_buffer("phase_color", phase="transparent", dependencies={"transparent_input"})
        def transparent_provider(self, context):
            calls.append(("transparent", context.phase.value))
            return context.sample("transparent_input")

        @geometry_buffer("unused")
        def unused_provider(self, context):
            raise AssertionError("unrequested providers must remain lazy")

    class DerivedPipeline(PhasePipeline):
        @geometry_buffer("phase_color", phase="opaque", dependencies={"derived_input"})
        def derived_provider(self, context):
            calls.append(("derived", context.phase.value))
            return context.sample("derived_input")

    pipeline = DerivedPipeline()
    for build in range(2):
        graph = RenderGraph(f"Phase Local Dependencies {build}")
        pipeline._defining_graph = graph
        if declaration == "pipeline":
            requested = pipeline.require_buffer("phase_color")
        else:
            graph.require_geometry_buffers({"phase_color"})
            requested = "phase_color"
        for phase in phases:
            dependency = "derived_input" if phase == "opaque" else "transparent_input"
            handle = graph.create_texture(dependency, format=Format.RGBA16_SFLOAT)
            result = pipeline.geometry_stage(
                graph, phase, phase=phase, buffers={dependency: handle}, queue_range=(0, 2500),
            )
            assert result.sample(requested) is handle
            assert set(result.snapshot) == {dependency, "phase_color"}
        assert graph.geometry_buffer_requirements == frozenset({"phase_color"})
    assert calls == [("derived" if phase == "opaque" else "transparent", phase) for phase in phases] * 2


def test_available_geometry_semantic_does_not_demand_its_provider_dependencies():
    class SeededPipeline(RenderPipeline):
        @geometry_buffer("provided", dependencies={"not_needed"})
        def provider(self, context):
            raise AssertionError("an already provided semantic must not run its provider")

    graph = RenderGraph("Supplied Geometry Semantic")
    graph.require_geometry_buffers({"provided"})
    texture = graph.create_texture("supplied", format=Format.RGBA16_SFLOAT)
    result = SeededPipeline().geometry_stage(
        graph, "opaque", buffers={"provided": texture}, queue_range=(0, 2500),
    )
    assert result.sample("provided") is texture
    assert not result.has("not_needed")


@pytest.mark.parametrize("pipeline_type", [DefaultForwardPipeline, DefaultForwardPlusPipeline, DefaultDeferredPipeline, RenderPipeline])
def test_nested_geometry_requirement_supplies_view_light_list(pipeline_type):
    calls = []

    class DependentPipeline(pipeline_type):
        def define(self, pipeline):
            pipeline.opaque().forward()

        @geometry_buffer("dependent_probe", dependencies={"light_list"})
        def probe(self, context):
            light_list = context.sample("light_list")
            calls.append(light_list)
            target = context.graph.create_texture("dependent_probe", format=Format.RGBA16_SFLOAT)
            with context.graph.add_pass("DependentProbe") as render_pass:
                render_pass.read_buffer(light_list)
                render_pass.write_color(target)
                render_pass.set_clear(color=(0, 1, 0, 1))
            return target

    graph = RenderGraph("Nested View Light List")
    graph.require_geometry_buffers({"dependent_probe"})
    pipeline = DependentPipeline()
    pipeline.define_topology(graph)
    assert len(calls) == 1
    source = (
        "geometry" if pipeline_type is RenderPipeline
        else "gbuffer" if pipeline_type is DefaultDeferredPipeline else "opaque"
    )
    assert graph.get_pass_result(source).sample("light_list") is calls[0]
    assert graph.geometry_buffer_requirements == frozenset({"dependent_probe"})


def test_opaque_geometry_planning_does_not_create_transparent_view_dependencies():
    class PhasePipeline(DefaultForwardPipeline):
        @geometry_buffer("phase_color", phase="opaque")
        def opaque_color(self, context):
            return context.sample("color")

        @geometry_buffer("phase_color", phase="transparent", dependencies={"light_list"})
        def transparent_color(self, context):
            raise AssertionError("an opaque stage must not execute transparent providers")

    graph = RenderGraph("Opaque Without Transparent Dependencies")
    graph.require_geometry_buffers({"phase_color"})
    PhasePipeline().define_topology(graph)
    result = graph.get_pass_result("opaque")
    assert result.sample("phase_color") is result.sample("color")
    assert not result.has("light_list")
    assert graph.geometry_buffer_requirements == frozenset({"phase_color"})


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
    from infernux.renderstack.pipeline_compiler import compile_pipeline_definition

    compile_pipeline_definition(builder.build(), graph, pipeline=pipeline)
    effect_result = graph.get_pass_result("effect:stylized")
    assert effect_result is not None
    assert effect_result.sample("color").name == "stylized_color"
