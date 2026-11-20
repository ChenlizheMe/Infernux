from __future__ import annotations

import pytest
from infernux.components.fields import get_serialized_fields

from infernux.rendergraph.graph import Format, RenderGraph
from infernux.renderstack.geometry_buffers import (
    GeometryBufferTopologyError,
    geometry_buffer,
)
from infernux.renderstack.render_pipeline import RenderPipeline
from infernux.renderstack import DefaultRayTracer, RayTracer, RayTracingContext
from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline
from infernux.renderstack.default_forward_plus_pipeline import DefaultForwardPlusPipeline
from infernux.renderstack.default_deferred_pipeline import DefaultDeferredPipeline, RayTracingQuality
from infernux.renderstack.fullscreen_effect import FullScreenEffect
from infernux.renderstack.pipeline_dsl import PipelineBuilder
from infernux.renderstack.resource_bus import ResourceBus


def test_lazy_result_orders_nested_producers_before_the_pass_that_first_samples_them():
    graph = RenderGraph("Lazy lighting")
    target = graph.create_texture("color", camera_target=True)
    requested = []

    def produce_raw():
        requested.append("raw")
        image = graph.create_texture("raw")
        graph.add_pass("Trace").write_color(image)
        return image

    with graph.name_scope("lighting"):
        raw = graph.publish_pass_result("sampled", {}, providers={"light": produce_raw})

        def reconstruct():
            requested.append("denoised")
            output = graph.create_texture("denoised")
            with graph.add_pass("Denoise") as p:
                p.set_texture("incoming", raw.sample("light"))
                p.write_color(output)
                p.fullscreen_quad("test/denoise")
            return output

        result = graph.publish_pass_result("filtered", {}, providers={"light": reconstruct})

    assert requested == []
    with graph.name_scope("project"):
        view = graph.add_pass("View")
        view.set_texture("light", result.sample("light"))
        view.write_color(target)
        view.fullscreen_quad("test/display")
    assert result.sample("light") is result.sample("light")
    assert requested == ["denoised", "raw"]
    expected = ["lighting/Trace", "lighting/Denoise", "project/View"]
    assert [name for kind, name in graph.topology_sequence if kind == "pass"] == expected
    assert [p.name for p in graph.build().passes] == expected


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


@pytest.mark.parametrize("diffuse,specular,quality,trace_divisor,filter_divisor", [
    (False, False, RayTracingQuality.STANDARD, 4, 4),
    (True, False, RayTracingQuality.STANDARD, 4, 4),
    (False, True, RayTracingQuality.LIGHTWEIGHT, 8, 4),
    (True, True, RayTracingQuality.HIGH, 4, 2),
    (True, True, RayTracingQuality.LIGHTWEIGHT, 8, 4),
])
def test_deferred_gi_controls_reflection_passes(diffuse, specular, quality, trace_divisor, filter_divisor):
    graph = RenderGraph("Deferred ray lighting")
    pipeline = DefaultDeferredPipeline()
    pipeline.global_illumination = diffuse
    pipeline.ray_specular = specular
    pipeline.ray_transparency = not diffuse
    pipeline.ray_quality = quality
    fields = get_serialized_fields(type(pipeline))
    for name in ("ray_specular", "ray_transparency", "ray_quality", "gi_intensity", "gi_saturation",
                 "ssao_strength", "ssao_contrast"):
        assert fields[name].visible_when(pipeline) == diffuse
    pipeline.define_topology(graph)
    graph.build()

    passes = {render_pass.name: render_pass for render_pass in graph._passes}
    result = graph.get_pass_result("ray/raw")
    reconstructed = graph.get_pass_result("ray/denoised")
    lighting = passes["DeferredLightingPass"]
    if not diffuse:
        assert result is None
        assert reconstructed is None
        assert lighting._shader_name == "infernux/internal/deferred_lighting"
        assert graph.get_buffer("world_ray_geometry") is None
        assert "Thin Surface Capture" not in passes
        return

    assert lighting._shader_name == "infernux/internal/deferred_linear_lighting"
    assert lighting._parameters == {"diffuse_mode": 1, "specular_mode": 2 if specular else 0,
                                    "diffuse_intensity": 1.0, "diffuse_saturation": 1.0,
                                    "ssao_strength": 0.5, "ssao_contrast": 1.0}
    visibility = graph.get_pass_result("gbuffer").sample("ambient_visibility")
    primary = graph.get_pass_result("gbuffer").sample("ray_primary_surface")
    assert visibility.size_divisor == primary.size_divisor == 0
    assert visibility.format == Format.R8_UNORM
    assert lighting._input_bindings["ambient_visibility"] == visibility.name
    assert lighting._input_bindings["primary_tex"] == primary.name

    observation = passes["Ray Diffuse Irradiance"]
    assert observation._parameters == {"divisor": trace_divisor, "temporal": True}
    assert observation._input_bindings["primary"] == primary.name
    assert observation._input_bindings["world_ray_meshes"] == graph.get_buffer("world_ray_meshes").name
    surface_reprojection = passes["Ray Diffuse Surface Reprojection"]
    assert surface_reprojection._input_bindings["primary"] == primary.name
    assert observation._input_bindings["reprojection"] == surface_reprojection._write_colors[0]
    assert graph.get_texture(surface_reprojection._write_colors[0]).format == Format.RG32_UINT
    screen = passes["Ray Diffuse Half Geometry"]
    assert screen._write_colors == {0: observation._input_bindings["screen_depth"]}
    assert graph.get_texture(screen._write_colors[0]).size_divisor == 2
    ao = passes["gbuffer/GTAO"]
    upsample = passes["gbuffer/GTAO Upsample"]
    temporal = passes["gbuffer/GTAO Temporal"]
    assert screen._order < surface_reprojection._order < ao._order < upsample._order < temporal._order < observation._order
    assert ao._input_bindings["guide"] == upsample._input_bindings["guide"] == screen._write_colors[0]
    assert ao._write_colors[0] == upsample._input_bindings["visibility"]
    assert upsample._write_colors[0] == temporal._input_bindings["visibility"]
    assert temporal._input_bindings["reprojection"] == surface_reprojection._write_colors[0]
    assert temporal._write_colors[0] == visibility.name
    assert graph.get_texture(temporal._input_bindings["history"]).temporal_key == visibility.temporal_key
    previous_color = graph.get_texture(observation._input_bindings["previous_radiance"])
    capture = passes["Record Scene Radiance"]
    assert graph.get_texture(capture._write_colors[0]).temporal_key == previous_color.temporal_key
    assert capture._order < passes["_WorldUI"]._order
    assert len(observation._write_colors) == 1
    packet = graph.get_texture(observation._write_colors[0])
    assert packet.size_divisor == trace_divisor and packet.format == Format.RGBA32_UINT
    gather = passes["Ray Diffuse SH Gather"]
    assert gather._input_bindings["observations"] == packet.name
    assert gather._input_bindings["ambient_visibility"] == visibility.name
    assert all(graph.get_texture(name).size_divisor == filter_divisor for name in gather._write_colors.values())
    history = passes["Ray Diffuse Reconstruction"]
    fill = passes["Ray Diffuse Fill Disocclusion"]
    blur = passes["Ray Diffuse Recurrent Blur"]
    resolve = passes["Ray Diffuse Resolved Irradiance"]
    assert blur._input_bindings["visibility"] == visibility.name
    assert gather._input_bindings["guide"] == blur._input_bindings["guide"] == screen._write_colors[0]
    assert history._input_bindings["reprojection"] == surface_reprojection._write_colors[0]
    for slot, channel in enumerate(("red", "green", "blue")):
        assert history._input_bindings[f"fresh_{channel}"] == gather._write_colors[slot]
        assert fill._input_bindings[f"{channel}_sh"] == history._write_colors[slot]
        assert blur._input_bindings[f"{channel}_sh"] == fill._write_colors[slot]
        assert resolve._input_bindings[f"{channel}_sh"] == blur._write_colors[slot]
        previous = graph.get_texture(history._input_bindings[f"old_{channel}"])
        assert previous.size_divisor == filter_divisor and previous._responds_to_ray_lighting
        current = graph.get_texture(blur._write_colors[slot])
        assert current.temporal_key == previous.temporal_key and current.size_divisor == filter_divisor
        assert current.format == previous.format == Format.RGBA16_SFLOAT
    assert fill._input_bindings["state"] == history._write_colors[3]
    assert fill._input_bindings["old_state"] == history._input_bindings["old_state"]
    assert blur._input_bindings["state"] == resolve._input_bindings["state"] == fill._write_colors[3]
    old_state = graph.get_texture(history._input_bindings["old_state"])
    state = graph.get_texture(fill._write_colors[3])
    assert state.temporal_key == old_state.temporal_key and state._responds_to_ray_lighting
    assert state.format == old_state.format == Format.RG16_SFLOAT
    assert "world_ray_meshes" not in resolve._input_bindings
    assert lighting._input_bindings["diffuse_irradiance"] == reconstructed.sample("diffuse_irradiance").name
    assert reconstructed.sample("diffuse_irradiance").size_divisor == 0
    assert "Ray Diffuse Raw Irradiance" not in passes
    assert result.sample("diffuse_irradiance").size_divisor == trace_divisor
    distance = result.sample("diffuse_hit_distance")
    assert distance is result.sample("diffuse_hit_distance") and distance.size_divisor == trace_divisor
    assert distance.format == Format.R32_SFLOAT

    assert result.has("specular_hit_distance") == reconstructed.has("specular_radiance") == specular
    if specular:
        fresh = passes["Ray Specular Radiance Samples"]
        reuse = passes["Ray Specular Radiance Path Candidates"]
        trace = passes["Ray Specular Radiance"]
        assert fresh._write_colors[0] == reuse._input_bindings["fresh"] == trace._write_colors[0]
        assert trace._input_bindings["paths"] == reuse._write_colors[1]
        assert trace._input_bindings["estimates"] == reuse._write_colors[0]
        specular_history = passes["Ray Specular Reconstruction"]
        assert specular_history._input_bindings["signal"] == trace._write_colors[0]
        assert lighting._input_bindings["ray_specular"] == specular_history._write_colors[0]
        assert result.sample("specular_radiance").format == Format.RGBA32_SFLOAT
        assert result.sample("specular_hit_distance").size_divisor == 0
    graph.build()


def test_deferred_diffuse_uses_the_taa_history_without_recopying_scene_color():
    from infernux.core.asset_ref import RenderEffectRef
    from infernux.renderstack.render_effect import RenderEffect
    from infernux.renderstack.render_effect_asset import RenderEffectAsset
    from infernux.renderstack.render_stack import RenderStack

    stack = RenderStack()
    pipeline = DefaultDeferredPipeline()
    pipeline.global_illumination = True
    stack._pipeline = pipeline
    pipeline._render_stack = stack
    taa = RenderEffect(RenderEffectAsset(feature_type="infernux.post.temporal_aa"))
    stack.add_effect_slot("final", RenderEffectRef(effect=taa))
    description = stack.build_graph()

    passes = {p.name: p for p in description.passes}
    textures = {t.name: t for t in description.textures}
    render_pass, = (p for p in passes.values() if p.name.endswith("/TAA_Resolve"))
    trace = passes["Ray Diffuse Irradiance"]
    previous, = (t for t in trace.read_textures if t.endswith("_scene_radiance/read"))
    assert previous in render_pass.read_textures
    assert textures[dict(render_pass.write_colors)[0]].temporal_key == textures[previous].temporal_key
    assert "Record Scene Radiance" not in passes


def test_screen_ui_records_scene_radiance_from_a_custom_pipeline_color_target():
    graph = RenderGraph("custom scene color")
    color = graph.create_texture("color", format=Format.RGBA16_SFLOAT, camera_target=True)
    graph.create_texture("depth", format=Format.D32_SFLOAT, camera_target=True)
    graph.add_pass("Custom Lighting").write_color(color)
    _, target = graph._scene_radiance_history()
    graph.screen_ui_section()

    passes = list(graph.build().passes)
    names = [p.name for p in passes]
    capture = passes[names.index("Record Scene Radiance")]
    assert names.index("Custom Lighting") < names.index(capture.name) < names.index("_WorldUI")
    assert color.name in capture.read_textures
    assert dict(capture.write_colors)[0] == target.name


def test_zero_contact_strength_preserves_visibility_guided_diffuse():
    pipeline = DefaultDeferredPipeline()
    pipeline.global_illumination = True
    pipeline.gi_intensity = 1.75
    pipeline.gi_saturation = 0.25
    pipeline.ssao_strength = 0.0
    graph = RenderGraph("contact disabled")
    pipeline.define_topology(graph)
    graph.build()
    passes = {p.name: p for p in graph._passes}
    assert graph.get_pass_result("gbuffer").has("ambient_visibility")
    assert "gbuffer/GTAO Temporal" in passes
    assert passes["DeferredLightingPass"]._parameters["diffuse_intensity"] == 1.75
    assert passes["DeferredLightingPass"]._parameters["diffuse_saturation"] == 0.25
    assert passes["DeferredLightingPass"]._parameters["ssao_strength"] == 0.0
    assert passes["Ray Diffuse Recurrent Blur"]._parameters["use_visibility"]


def test_tracer_replacement_controls_the_entire_graph_and_one_lobe_independently():
    class CustomMirror(DefaultRayTracer):
        def reconstruct_specular(self, ctx, raw, guides):
            # Replace just the specular denoiser with the raw outgoing radiance.
            return raw.radiance

    pipeline = DefaultDeferredPipeline(CustomMirror())
    pipeline.global_illumination = pipeline.ray_specular = True
    pipeline.ssao_contrast = 2.0
    graph = RenderGraph("custom mirror")
    pipeline.define_topology(graph)
    graph.build()
    passes = {p.name: p for p in graph._passes}
    assert "Ray Diffuse Reconstruction" in passes and "Ray Specular Reconstruction" not in passes
    assert passes["DeferredLightingPass"]._shader_name == "infernux/internal/deferred_linear_lighting"
    assert passes["DeferredLightingPass"]._parameters == {"diffuse_mode": 1, "specular_mode": 1,
                                                          "diffuse_intensity": 1.0, "diffuse_saturation": 1.0,
                                                          "ssao_strength": 0.5, "ssao_contrast": 2.0}
    assert "Resolve diffuse_irradiance" not in passes
    assert passes["DeferredLightingPass"]._input_bindings["specular_radiance"] == graph.get_pass_result("ray/raw").sample("specular_radiance").name

    def scene_image(self, ctx):
        hdr = ctx.graph.create_texture("project/hdr", format=Format.RGBA16_SFLOAT)
        p = ctx.graph.add_pass("project renderer")
        p.write_color(hdr)
        p.set_clear(color=(0.25, 0.5, 0.75, 1))
        return ctx.publish(scene_color=hdr)

    for base in (RayTracer, DefaultRayTracer):
        pipeline.ray_tracer = type("ProjectRenderer", (base,), {"build": scene_image})()
        pipeline.ray_transparency = True
        graph = RenderGraph("whole scene")
        pipeline.define_topology(graph)
        graph.build()
        assert graph.get_buffer("world_ray_geometry") is None
        assert not any(t.temporal_key for t in graph._textures)
        names = {p.name for p in graph._passes}
        assert {"project renderer", "Ray Scene Color", "TransparentPass"} <= names
        assert not {"DeferredLightingPass", "SkyboxPass", "Thin Surface Capture", "Ray Tail Estimate"} & names


def test_custom_specular_trace_reads_world_cache_without_requesting_an_update():
    class Mirror(DefaultRayTracer):
        def build(self, ctx):
            guides = self.prepare_guides(ctx)
            mirror = self.trace(ctx, guides, specular=True, divisor=0)
            return ctx.publish(scene_color=mirror.radiance)

    graph = RenderGraph("read-only mirror")
    pipeline = DefaultDeferredPipeline(Mirror())
    pipeline.global_illumination = True
    pipeline.ray_specular = True
    pipeline.define_topology(graph)
    graph.build()
    assert {b.name for b in graph._buffers if "ray_tail_" in b.name} == {"ray_tail_read"}
    assert all(not p.name.startswith("Ray Tail") for p in graph._passes)


@pytest.mark.parametrize("own_guides", [False, True])
def test_custom_geometry_reuses_registered_surfaces_for_both_lobes(own_guides):
    graph = RenderGraph("Studio")
    ctx = RayTracingContext(graph, name="studio")
    hdr = graph.create_texture("project/camera", camera_target=True)
    layout = {"base_color": Format.RGBA16_SFLOAT, "surface": Format.RG32_UINT,
              "emission": Format.RGBA16_SFLOAT, "depth": Format.D32_SFLOAT}
    inputs = {key: graph.create_texture(f"project/{key}", format=fmt) for key, fmt in layout.items()}
    inputs["surface_records"] = ctx.world.surface_records
    normal = graph.create_texture("project/normals", format=Format.RGBA16_SFLOAT)
    if own_guides:
        previous, current = ctx.view.history("primary", format=Format.RGBA32_UINT)
        assert previous._depends_on_ray_world and current._depends_on_ray_world
        inputs.update(ray_previous_primary_surface=previous, ray_primary_surface=current,
                      ray_surface_motion=graph.create_texture("project/motion", format=Format.RGBA32_UINT))

    class StudioPipeline(RenderPipeline):
        @geometry_buffer("surface_frame")
        def normals(self, provider):
            return normal

    surface = StudioPipeline().geometry_stage(graph, "studio_meshes", buffers=inputs, queue_range=(0, 2500))
    ctx.geometry = ctx.surface = surface
    frame = DefaultRayTracer().build(ctx)
    ctx.shade_deferred(frame, hdr)
    passes = {p.name: p for p in graph._passes}
    primary = surface.sample("ray_primary_surface")
    for name in ("Ray Diffuse Irradiance", "Ray Specular Radiance"):
        assert passes[name]._input_bindings["primary"] == primary.name
    assert passes["Ray Specular Radiance"]._input_bindings["normal_tex"] == normal.name
    assert passes["Ray Diffuse Resolved Irradiance"]._input_bindings["frame"] == normal.name
    assert passes["DeferredLightingPass"]._input_bindings["surface_tex"] == inputs["surface"].name
    assert len([name for name in passes if "Ray Surface Replay" in name]) == int(not own_guides)
    assert len([name for name in passes if "Ray Surface Witness" in name]) == int(not own_guides)
    assert not any("Record Ray Primary" in name or name.startswith("Resolve ") for name in passes)
    if own_guides:
        assert primary is current
        assert passes["Ray Specular Reconstruction"]._input_bindings["previous_primary"] == previous.name
    else:
        assert surface.sample("ray_surface_depth").name == passes["studio_meshes/Ray Surface Replay"]._write_depth
        assert primary.temporal_key == surface.sample("ray_previous_primary_surface").temporal_key
    assert sum(p.name == "studio_meshes/Ray Surface Replay" for p in graph._passes) == int(not own_guides)


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


def test_deferred_thin_sheet_preserves_background_and_partitions_transparency():
    graph = RenderGraph("Thin sheet composition")
    pipeline = DefaultDeferredPipeline()
    pipeline.global_illumination = True
    pipeline.ray_transparency = True
    pipeline.define_topology(graph)
    passes = {render_pass._name: render_pass for render_pass in graph._passes}
    layer = graph.get_texture("_ray_thin_layer")
    optics = graph.get_texture("_ray_thin_optics")
    assert layer.format == optics.format == Format.RGBA32_UINT
    capture = passes["Thin Surface Capture"]
    assert capture._material_pass == "thin_surface"
    assert capture._material_filter == "thin_sheet"
    assert capture._write_colors == {0: layer.name, 1: optics.name}
    assert capture._clear_depth is None
    assert capture._write_depth == "_ray_thin_depth"
    for phase, label in (
        ("transparent_behind", "Transparent Behind Thin Surface"),
        ("transparent_front", "Transparent In Front Of Thin Surface"),
    ):
        render_pass = passes[label]
        assert render_pass._material_pass == phase
        assert render_pass._material_filter == "ordinary_transparent"
        assert render_pass._input_bindings["transparent_layer"] == layer.name
        assert render_pass._write_depth is None
    background = passes["Thin Surface Background"]
    assert background._input_bindings["source_tex"] == "color"
    assert background._write_colors == {0: "_ray_thin_background"}
    assert passes["Thin Surface Depth"]._destination_resource == capture._write_depth
    compose = passes["Thin Surface Composite"]
    assert compose._input_bindings["background"] == background._write_colors[0] != compose._write_colors[0]
    names = list(passes)
    assert names.index("Thin Surface Capture") < names.index("Transparent Behind Thin Surface")
    assert names.index("Transparent Behind Thin Surface") < names.index("Thin Surface Background")
    assert names.index("Thin Surface Background") < names.index("Thin Surface Composite")
    assert names.index("Thin Surface Composite") < names.index("Transparent In Front Of Thin Surface")
    assert "TransparentPass" not in passes


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
