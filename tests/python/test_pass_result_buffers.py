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


def test_deferred_public_normal_preserves_coverage_and_effect_revision():
    graph = RenderGraph("Deferred normal coverage")
    graph.set_geometry_buffer_requirements({"normal"})
    authored_normal = None
    authored_base_color = None

    def replace_gbuffer_normal(stage):
        nonlocal authored_normal, authored_base_color
        if stage.stable_id != "after_gbuffer":
            return
        current = graph.current_pass_result
        authored_normal = graph.create_texture("authored_normal", format=Format.RGBA16_SFLOAT)
        with graph.add_pass("AuthorNormal") as render_pass:
            render_pass.set_texture("source_tex", current.sample("normal"))
            render_pass.write_color(authored_normal)
            render_pass.fullscreen_quad("infernux/internal/fullscreen_blit")
        authored_base_color = graph.create_texture("authored_base_color", format=Format.RGBA16_SFLOAT)
        with graph.add_copy_pass("AuthorBaseColor") as copy_pass:
            copy_pass.copy_texture(current.sample("base_color"), authored_base_color)
        graph.replace_current_pass_result(graph.derive_pass_result(
            "effect:surface", current, {"normal": authored_normal, "base_color": authored_base_color},
        ))

    graph._effect_stage_callback = replace_gbuffer_normal
    DefaultDeferredPipeline().define_topology(graph)
    graph.build()

    gbuffer = graph.get_pass_result("gbuffer")
    opaque = graph.get_pass_result("opaque_lighting")
    packed = gbuffer.sample("surface_frame")
    assert packed is graph.get_texture("gbuffer_normal")
    assert gbuffer.sample("base_color") is graph.get_texture("gbuffer_albedo")
    assert opaque.sample("base_color") is authored_base_color
    public_gbuffer = gbuffer.sample("normal")
    public_opaque = opaque.sample("normal")
    assert packed is not public_gbuffer
    assert public_gbuffer.name == "_result/gbuffer/normal"
    assert public_opaque.name == "_result/opaque_lighting/normal"
    assert public_opaque is not public_gbuffer
    assert public_opaque is not authored_normal
    assert all(texture.format == Format.RGBA16_SFLOAT for texture in (packed, public_gbuffer, public_opaque))

    passes = {render_pass.name: render_pass for render_pass in graph._passes}
    names = list(passes)
    assert names.index("GBufferPass") < names.index("gbuffer/Normal")
    assert names.index("gbuffer/Normal") < names.index("DeferredLightingPass")
    assert (
        names.index("DeferredForwardFallbackPass")
        < names.index("opaque_lighting/NormalBase")
        < names.index("opaque_lighting/Normal")
    )
    assert names.index("opaque_lighting/Normal") < names.index("SkyboxPass")
    lighting = passes["DeferredLightingPass"]
    assert lighting._input_bindings["normal_tex"] == packed.name
    assert lighting._input_bindings["albedo_tex"] == authored_base_color.name
    formats = tuple(graph.get_texture(name).format for name in passes["GBufferPass"]._write_colors.values())
    assert formats == (Format.RGBA16_SFLOAT,) * 3 + (Format.RG32_UINT,)
    surface_records = gbuffer.sample("surface_records")
    assert lighting._input_bindings["surface_records"] == surface_records.name
    assert (surface_records.name, "storage_read") in lighting._buffer_accesses

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


def test_deferred_ray_surface_replays_raw_gbuffer_depth_and_surface_id_on_demand():
    graph = RenderGraph("Deferred ray surface")
    pipeline = DefaultDeferredPipeline()
    pipeline.define_topology(graph)
    assert not graph.get_pass_result("gbuffer").has("ray_primary_surface")

    graph = RenderGraph("Deferred with ray surface")
    graph.require_geometry_buffers({"ray_primary_surface"})
    pipeline.define_topology(graph)
    graph.build()

    source = graph.get_pass_result("gbuffer")
    primary = source.sample("ray_primary_surface")
    passes = {render_pass.name: render_pass for render_pass in graph._passes}
    gbuffer = passes["GBufferPass"]
    replay = passes["gbuffer/Ray Surface Replay"]
    witness = passes["gbuffer/Ray Surface Witness"]
    order = list(passes)
    assert order.index(gbuffer.name) < order.index(replay.name) < order.index(witness.name)
    assert replay._material_filter == gbuffer._material_filter == "deferred_compatible"
    assert replay._sort_mode == gbuffer._sort_mode
    assert replay._queue_min == gbuffer._queue_min
    assert replay._queue_max == gbuffer._queue_max
    assert replay._clear_depth == gbuffer._clear_depth == 0.0
    assert witness._input_bindings == {
        "gbuffer_surface": source.sample("surface").name,
        "gbuffer_depth": source.sample("depth").name,
        "gbuffer_normal": source.sample("surface_frame").name,
        "replay_surface": replay._write_colors[0],
        "replay_depth": replay._write_depth,
    }
    assert witness._write_colors[0] == primary.name
    assert graph.get_texture(replay._write_colors[0]).format == Format.RGBA32_UINT


@pytest.mark.parametrize("diffuse,specular", [(False, False), (True, False), (False, True), (True, True)])
def test_deferred_ray_indirect_builds_only_requested_lobes(diffuse, specular):
    graph = RenderGraph("Deferred ray lighting")
    pipeline = DefaultDeferredPipeline()
    pipeline.ray_diffuse = diffuse
    pipeline.ray_specular = specular
    pipeline.define_topology(graph)
    graph.build()

    passes = {render_pass.name: render_pass for render_pass in graph._passes}
    result = graph.get_pass_result("ray_indirect")
    reconstructed = graph.get_pass_result("ray_reconstructed")
    lighting = passes["DeferredLightingPass"]
    if not (diffuse or specular):
        assert result is None
        assert reconstructed is None
        assert lighting._shader_name == "infernux/internal/deferred_lighting"
        assert graph.get_buffer("world_ray_geometry") is None
        return

    assert lighting._shader_name == "infernux/internal/deferred_ray_lighting"
    assert lighting._parameters == {"diffuse_enabled": diffuse, "specular_enabled": specular}
    for enabled, name, label, resolve in (
        (diffuse, "ray_diffuse", "Ray Diffuse Irradiance", "Ray Diffuse Reconstruction"),
        (specular, "ray_specular", "Ray Specular Radiance", "Ray Specular Reconstruction"),
    ):
        assert result.has(name) == (label in passes) == enabled
        if enabled:
            trace = passes[label]
            assert trace._input_bindings["primary"] == result.sample("ray_primary_surface").name
            assert trace._input_bindings["world_ray_meshes"] == graph.get_buffer("world_ray_meshes").name
            assert graph.get_texture(trace._write_colors[0]).format == Format.RGBA32_UINT
            assert passes[resolve]._input_bindings["signal"] == trace._write_colors[0]
            assert passes[f"Decode {name} diagnostic"]._write_colors == {0: result.sample(name).name}
            assert lighting._input_bindings[name] == reconstructed.sample(name).name


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
        changed = graph.create_texture(f"{stage.stable_id}_color", format=Format.RGBA16_SFLOAT)
        graph.replace_current_pass_result(graph.write_buffer(f"effect:{stage.stable_id}", current, "color", changed))

    graph._effect_stage_callback = replace_color
    from infernux.renderstack.pipeline_compiler import compile_pipeline_definition

    compile_pipeline_definition(builder.build(), graph, pipeline=pipeline)
    effect_result = graph.get_pass_result("effect:stylized")
    assert effect_result is not None
    assert effect_result.sample("color").name == "stylized_color"
