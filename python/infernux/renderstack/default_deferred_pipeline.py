"""
DefaultDeferredPipeline — Standard deferred rendering pipeline.

Implements a classic deferred shading pipeline with a GBuffer pass
that writes multiple render targets (albedo, normals, material props),
followed by a fullscreen lighting pass, then transparent forward pass.

Private GBuffer layout (MRT)::

    Slot 0 — Base color and metallic (RGBA16_SFLOAT)
    Slot 1 — Octahedral normal, roughness and AO (RGBA16_SFLOAT)
    Slot 2 — Emission and specular strength (RGBA16_SFLOAT)
    Slot 3 — Surface ID and two packed model scalars (RG32_UINT)
    Depth  — Scene depth        (D32_SFLOAT)

The public ``normal`` PassResult is generated only when requested. Its RGB is
the current View's encoded world normal and its alpha is opaque geometry
coverage (1 where visible, 0 in the cleared background). Deferred-compatible
and Deferred-unsupported geometry are published in separate source revisions.

Topology::

    ShadowCasterPass → GBufferPass → after_gbuffer
    → DeferredLightingPass → DeferredForwardFallbackPass → after_opaque
    → SkyboxPass → after_sky
    → TransparentPass (Forward+) → after_transparent
    → [post-process injection points]

Usage::

    stack = game_object.add_component(RenderStack)
    stack.set_pipeline("Default Deferred")

.. note::
    This pipeline requires the engine to support:
    - MRT (multiple render targets) — already supported
    - Depth buffer as shader input — now supported
    - Deferred lighting program (``infernux/internal/deferred_lighting``)

    Deferred lighting uses the same camera-local light, shadow, layer-mask,
    and Forward+ tile data as the built-in Forward+ path.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from infernux.lib import _Infernux
from infernux.renderstack.render_pipeline import RenderPipeline
from infernux.renderstack.geometry_buffers import geometry_buffer
from infernux.rendergraph.graph import Format
from infernux.renderstack._platform_quality import effective_shadow_resolution
from infernux.components.fields import serialized_field
from infernux.renderstack._pipeline_common import (
    COLOR_TEXTURE,
    LIGHT_LIST_BUFFER,
    DEPTH_TEXTURE,
    DEFERRED_GBUFFER_CLEAR_COLOR,
    DEFERRED_LIGHTING_CLEAR_COLOR,
    NORMAL_TEXTURE,
    SHADOW_MAP_TEXTURE,
    SURFACE_RECORDS_BUFFER,
    add_normal_buffer_pass,
    add_shadow_caster_pass,
    add_skybox_pass,
    add_standard_post_process_section,
    add_transparent_pass,
    transparent_queue_range,
    bind_deferred_surface,
    create_deferred_gbuffer,
    create_main_scene_targets,
    draw_deferred_lighting,
    opaque_queue_range,
    result_resources,
    write_deferred_gbuffer,
)

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph


_RAY_PRIMARY_SURFACE = "ray_primary_surface"
_RAY_PRIMARY_HISTORY = "_ray_primary"
_RAY_DIFFUSE = "ray_diffuse"
_RAY_SPECULAR = "ray_specular"


class RayTracingQuality(IntEnum):
    LIGHTWEIGHT = 0
    STANDARD = 1
    HIGH = 2


_RAY_DIFFUSE_SIZE_DIVISORS = {
    RayTracingQuality.LIGHTWEIGHT: 4,
    RayTracingQuality.STANDARD: 2,
    RayTracingQuality.HIGH: 0,
}


def _draw_deferred_geometry(render_pass, depth, queue_range, sort_mode, material_pass):
    render_pass.write_depth(depth)
    render_pass.set_clear(color=DEFERRED_GBUFFER_CLEAR_COLOR, depth=0.0)
    render_pass.draw_renderers(
        queue_range=queue_range,
        sort_mode=sort_mode,
        material_pass=material_pass,
        material_filter="deferred_compatible",
    )


def _ray_replay_name(source):
    return f"_result/{source}/ray_surface_witness"


def _trace_ray_indirect(graph, gbuffer, lobes, temporal):
    world = graph._create_world_ray_lit()
    tail = graph._create_world_ray_tail()
    packets = {}
    for name, specular, divisor in lobes:
        output = graph.create_texture(f"_result/gbuffer/{name}_packet", format=Format.RGBA32_UINT,
                                      size_divisor=divisor, samples=1)
        label = "Ray Specular Radiance" if specular else "Ray Diffuse Irradiance"
        with graph.add_pass(label) as render_pass:
            render_pass.write_color(output)
            bind_deferred_surface(render_pass, gbuffer.snapshot)
            render_pass.set_texture("primary", gbuffer.sample(_RAY_PRIMARY_SURFACE))
            render_pass.set_buffer("tail_cache", tail["ray_tail_read"])
            for key in ("world_ray_geometry", "world_ray_meshes", "ray_lit_materials", "ray_texture_descriptors"):
                render_pass.set_buffer(key, world[key])
            for key in ("ray_color_pages", "ray_data_pages"):
                render_pass.set_texture(key, world[key])
            render_pass.set_param("specular", specular)
            render_pass.set_param("temporal", temporal)
            render_pass.set_param("resolution_divisor", divisor or 1)
            render_pass.fullscreen_quad("infernux/internal/ray_indirect")
        packets[name] = output
    return packets


def _decode_raw_signals(graph, packets):
    raw = {}
    for name, packet in packets.items():
        output = graph.create_texture(f"_result/gbuffer/{name}", format=Format.RGBA16_SFLOAT,
                                      size_divisor=packet.size_divisor, samples=1)
        with graph.add_pass(f"Decode {name} diagnostic") as render_pass:
            render_pass.set_texture("signal", packet)
            render_pass.write_color(output)
            render_pass.fullscreen_quad("infernux/internal/ray_raw_signal")
        raw[name] = output
    return raw


def _capture_thin_transparency(graph):
    layer = graph.create_texture("_ray_thin_layer", format=Format.RGBA32_UINT, samples=1)
    optics = graph.create_texture("_ray_thin_optics", format=Format.RGBA32_UINT, samples=1)
    depth = graph.create_texture("_ray_thin_depth", format=Format.D32_SFLOAT, samples=1)
    with graph.add_copy_pass("Thin Surface Depth") as render_pass:
        render_pass.copy_texture(DEPTH_TEXTURE, depth)
    with graph.add_pass("Thin Surface Capture") as render_pass:
        render_pass.write_color(layer, slot=0)
        render_pass.write_color(optics, slot=1)
        render_pass.write_depth(depth)
        render_pass.set_clear(color=(0.0, 0.0, 0.0, 0.0))
        render_pass.draw_renderers(
            queue_range=transparent_queue_range(), sort_mode="front_to_back",
            material_pass="thin_surface", material_filter="thin_sheet",
        )
    return layer, optics


def _update_ray_tail(graph, gbuffer, *, opaque, thin):
    world = graph._create_world_ray_lit()
    tail = graph._create_world_ray_tail()
    count = _Infernux._ray_tail_batch_size()
    requests = tuple(graph.create_texture(f"_ray_tail_request_{index}", format=Format.RGBA32_UINT,
                                          size=(count, 2), samples=1) for index in range(2))
    samples = graph.create_texture("_ray_tail_samples", format=Format.RGBA32_SFLOAT,
                                    size=(count, 1), samples=1)
    primary = gbuffer.sample(_RAY_PRIMARY_SURFACE) if opaque else thin[0]
    with graph.add_pass("Ray Tail Requests") as render_pass:
        render_pass._ray_tail_batch = 2
        bind_deferred_surface(render_pass, gbuffer.snapshot)
        render_pass.set_texture("primary", primary)
        render_pass.set_texture("thin_layer", thin[0] if thin else primary)
        render_pass.set_buffer("requests", tail["ray_tail_requests"])
        for key in ("world_ray_geometry", "world_ray_meshes", "ray_lit_materials", "ray_texture_descriptors"):
            render_pass.set_buffer(key, world[key])
        for key in ("ray_color_pages", "ray_data_pages"):
            render_pass.set_texture(key, world[key])
        for index, target in enumerate(requests):
            render_pass.write_color(target, slot=index)
        render_pass.set_param("opaque_enabled", opaque)
        render_pass.set_param("thin_enabled", bool(thin))
        render_pass.fullscreen_quad("infernux/internal/ray_tail_request")
    with graph.add_compute_pass("Ray Tail Prepare") as render_pass:
        render_pass._action = "ray_tail_prepare"
        render_pass._ray_tail_batch = 1
        render_pass._side_effect = True
        render_pass.set_textures({f"request_{index}": target for index, target in enumerate(requests)})
        render_pass.read_buffer(tail["ray_tail_read"])
        render_pass.read_buffer(tail["ray_tail_control"])
        render_pass.write_buffer(tail["ray_tail_write"])
        render_pass.write_buffer(tail["ray_tail_requests"])
    with graph.add_pass("Ray Tail Estimate") as render_pass:
        render_pass._ray_tail_batch = 2
        render_pass.set_buffer("requests", tail["ray_tail_requests"])
        render_pass.set_buffer("cache", tail["ray_tail_read"])
        render_pass.set_buffer("surface_records", gbuffer.snapshot[SURFACE_RECORDS_BUFFER])
        for key in ("world_ray_geometry", "world_ray_meshes", "ray_lit_materials", "ray_texture_descriptors"):
            render_pass.set_buffer(key, world[key])
        for key in ("ray_color_pages", "ray_data_pages"):
            render_pass.set_texture(key, world[key])
        render_pass.write_color(samples)
        render_pass.fullscreen_quad("infernux/internal/ray_tail_estimate")
    with graph.add_compute_pass("Ray Tail Merge") as render_pass:
        render_pass._action = "ray_tail_merge"
        render_pass._ray_tail_batch = 1
        render_pass._side_effect = True
        render_pass.set_texture("tail_samples", samples)
        render_pass.read_buffer(tail["ray_tail_requests"])
        render_pass.read_buffer(tail["ray_tail_control"])
        render_pass.write_buffer(tail["ray_tail_write"])


def _draw_thin_transparency(graph, surface_records, capture):
    layer, optics = capture
    reflection = graph.create_texture("_ray_thin_reflection", format=Format.RGBA16_SFLOAT, samples=1)
    background = graph.create_texture("_ray_thin_background", format=Format.RGBA16_SFLOAT, samples=1)
    world = graph._create_world_ray_lit()
    with graph.add_pass("Thin Surface Reflection") as render_pass:
        render_pass.set_textures({"layer": layer, "optics": optics})
        render_pass.set_buffer("surface_records", surface_records)
        render_pass.set_buffer("tail_cache", graph._create_world_ray_tail()["ray_tail_read"])
        for key in ("world_ray_geometry", "world_ray_meshes", "ray_lit_materials", "ray_texture_descriptors"):
            render_pass.set_buffer(key, world[key])
        for key in ("ray_color_pages", "ray_data_pages"):
            render_pass.set_texture(key, world[key])
        render_pass.write_color(reflection)
        render_pass.fullscreen_quad("infernux/internal/ray_thin_reflection")
    with graph.add_pass("Transparent Behind Thin Surface") as render_pass:
        render_pass.read(DEPTH_TEXTURE)
        render_pass.write_color(COLOR_TEXTURE)
        render_pass.set_textures({"transparent_layer": layer, "shadowMap": graph.get_texture(SHADOW_MAP_TEXTURE)})
        render_pass.draw_renderers(
            queue_range=transparent_queue_range(), sort_mode="back_to_front",
            material_pass="transparent_behind", material_filter="ordinary_transparent",
        )
    with graph.add_pass("Thin Surface Background") as render_pass:
        render_pass.set_texture("source_tex", graph.get_texture(COLOR_TEXTURE))
        render_pass.write_color(background)
        render_pass.fullscreen_quad("infernux/internal/fullscreen_copy")
    with graph.add_pass("Thin Surface Composite") as render_pass:
        render_pass.set_textures({"background": background, "layer": layer, "optics": optics, "reflection": reflection})
        render_pass.write_color(COLOR_TEXTURE)
        render_pass.fullscreen_quad("infernux/internal/thin_composite")
    with graph.add_pass("Transparent In Front Of Thin Surface") as render_pass:
        render_pass.read(DEPTH_TEXTURE)
        render_pass.write_color(COLOR_TEXTURE)
        render_pass.set_textures({"transparent_layer": layer, "shadowMap": graph.get_texture(SHADOW_MAP_TEXTURE)})
        render_pass.draw_renderers(
            queue_range=transparent_queue_range(), sort_mode="back_to_front",
            material_pass="transparent_front", material_filter="ordinary_transparent",
        )


def _reconstruct_ray_indirect(graph, source, lobes, packets, primary_history, surface_frame):
    previous_primary, primary = primary_history
    motion = graph.get_texture(_ray_replay_name(source))
    geometry = graph._create_world_ray_geometry()
    signals = {}
    for name, specular, divisor in lobes:
        previous, output = graph.create_temporal_history(f"_{name}", format=Format.RGBA32_UINT,
                                                         size_divisor=divisor)
        label = "Ray Specular Reconstruction" if specular else "Ray Diffuse Reconstruction"
        with graph.add_pass(label) as render_pass:
            render_pass.set_textures({
                "signal": packets[name], "primary": primary, "motion": motion,
                "previous_signal": previous, "previous_primary": previous_primary, "frame": surface_frame,
            })
            render_pass.set_buffer("geometry", geometry)
            render_pass.write_color(output)
            render_pass.set_param("specular", specular)
            render_pass.fullscreen_quad("infernux/internal/ray_reconstruct")
        if not specular:
            # Filter at the trace resolution. Only the temporal estimate feeds
            # the next frame, so spatially filtered neighbours never feed back.
            for stride in ((1,) if divisor == 4 else (1, 2)):
                filtered = graph.create_texture(f"_{name}/spatial_{stride}", format=Format.RGBA32_UINT,
                                                size_divisor=divisor, samples=1)
                with graph.add_pass(f"Ray Diffuse Spatial {stride}") as render_pass:
                    render_pass.set_textures({"signal": output, "primary": primary, "frame": surface_frame})
                    render_pass.write_color(filtered)
                    render_pass.set_param("stride", stride)
                    render_pass.fullscreen_quad("infernux/internal/ray_diffuse_filter")
                output = filtered
        signals[name] = output
    return signals


def _draw_ray_lighting(render_pass, gbuffer, signals):
    bind_deferred_surface(render_pass, gbuffer.snapshot)
    render_pass.set_texture("primary_tex", gbuffer.sample(_RAY_PRIMARY_SURFACE))
    # One enabled signal also supplies the unused descriptor when a lobe is off.
    existing = next(iter(signals.values()))
    render_pass.set_textures({name: signals.get(name, existing) for name in (_RAY_DIFFUSE, _RAY_SPECULAR)})
    render_pass.set_param("diffuse_enabled", _RAY_DIFFUSE in signals)
    render_pass.set_param("specular_enabled", _RAY_SPECULAR in signals)
    render_pass.fullscreen_quad("infernux/internal/deferred_ray_lighting")


class DeferredMSAA(IntEnum):
    """Deferred pipeline only supports OFF or resolve-after-lighting."""
    OFF = 1


class DefaultDeferredPipeline(RenderPipeline):
    """Standard deferred rendering pipeline.

    Uses GBuffer multi-target rendering to separate geometry and lighting,
    suitable for scenes with many light sources.

    Injection points:

    =============================  ==================================
    Injection Point                 Timing
    =============================  ==================================
    ``after_gbuffer``              After GBuffer, before lighting
    ``after_opaque``               After deferred lighting, before skybox
    ``after_sky``                  After skybox, before transparent
    ``after_transparent``          After transparent objects
    =============================  ==================================

    ``before_post_process`` / ``after_post_process`` are automatically
    inserted by ``graph.screen_ui_section()``.

    ``ray_diffuse`` and ``ray_specular`` independently replace the raster
    environment terms. ``ray_quality`` sets the diffuse tracing rate. Light
    transport samples are kept on their native grid through reconstruction.
    Lighting resolves them against the full-resolution surface; specular
    paths use the full-resolution grid at every quality.
    """

    name: str = "Default Deferred"

    # ------------------------------------------------------------------
    # Exposed parameters
    # ------------------------------------------------------------------
    shadow_resolution: int = serialized_field(
        default=4096,
        range=(256, 8192),
        slider=False,
        tooltip="Shadow map resolution (width & height)",
        header="Shadows",
    )
    ray_diffuse: bool = serialized_field(default=False, header="Ray Tracing")
    ray_specular: bool = serialized_field(default=False)
    ray_transparency: bool = serialized_field(default=False)
    ray_quality: RayTracingQuality = serialized_field(default=RayTracingQuality.STANDARD)
    _reconstruct_rays = True

    @geometry_buffer(NORMAL_TEXTURE, dependencies={DEPTH_TEXTURE})
    def _provide_deferred_normal(self, context):
        # The material frame stores octahedral normals with roughness and AO.
        # This provider publishes the display-oriented normal with coverage.
        return add_normal_buffer_pass(
            context.graph,
            source=context.source,
            depth=context.sample(DEPTH_TEXTURE),
            queue_range=context.queue_range,
            material_filter="deferred_compatible",
        )

    @geometry_buffer(_RAY_PRIMARY_SURFACE, dependencies={DEPTH_TEXTURE, "surface"})
    def _provide_ray_primary_surface(self, context):
        graph = context.graph
        stem = f"_result/{context.source}/ray_surface"
        replay = graph.create_texture(_ray_replay_name(context.source), format=Format.RGBA32_UINT, samples=1)
        depth = graph.create_texture(f"{stem}_depth", format=Format.D32_SFLOAT, samples=1)
        result = graph.get_texture(f"{_RAY_PRIMARY_HISTORY}/write")
        if result is None:
            result = graph.create_texture(f"{stem}_primary_witness", format=Format.RGBA32_UINT, samples=1)
        with graph.add_pass(f"{context.source}/Ray Surface Replay") as render_pass:
            render_pass.write_color(replay)
            _draw_deferred_geometry(render_pass, depth, context.queue_range, context.sort_mode, "ray_surface")
        with graph.add_pass(f"{context.source}/Ray Surface Witness") as render_pass:
            render_pass.write_color(result)
            render_pass.set_textures({
                "gbuffer_surface": context.sample("surface"),
                "gbuffer_depth": context.sample(DEPTH_TEXTURE),
                "gbuffer_normal": context.sample("surface_frame"),
                "replay_surface": replay,
                "replay_depth": depth,
            })
            render_pass.fullscreen_quad("infernux/internal/ray_primary_surface")
        return result

    # ------------------------------------------------------------------
    # RenderPipeline interface
    # ------------------------------------------------------------------

    def define_topology(self, graph: "RenderGraph") -> None:
        """Define the built-in deferred rendering topology.

        Topology::

            ShadowCaster → GBuffer (MRT) → after_gbuffer
            → DeferredLighting → Deferred Forward+ Fallback → after_opaque
            → Skybox → after_sky
            → Transparent (forward) → after_transparent
        """
        # Deferred pipeline does not support MSAA on GBuffer
        if graph.set_msaa_samples(1) != 1:
            raise ValueError("Default Deferred requires a single-sample Camera target")

        diffuse_divisor = _RAY_DIFFUSE_SIZE_DIVISORS[self.ray_quality]
        lobes = tuple((name, specular, divisor) for name, specular, divisor, enabled in (
            (_RAY_DIFFUSE, False, diffuse_divisor, self.ray_diffuse), (_RAY_SPECULAR, True, 0, self.ray_specular),
        ) if enabled)
        temporal = bool(lobes) and self._reconstruct_rays
        primary_history = None
        if temporal:
            primary_history = graph.create_temporal_history(_RAY_PRIMARY_HISTORY, format=Format.RGBA32_UINT)
        if lobes:
            graph.require_geometry_buffers({_RAY_PRIMARY_SURFACE})
        shadow_res = effective_shadow_resolution(self.shadow_resolution)

        # ---- GBuffer textures (MRT) ----
        create_main_scene_targets(
            graph,
            shadow_resolution=shadow_res,
            msaa_samples=1,
        )
        surface = create_deferred_gbuffer(graph)
        geometry_queues = opaque_queue_range()
        geometry_sort = "front_to_back"

        # ---- Pass 0: Shadow casters ----
        add_shadow_caster_pass(graph)

        # ---- Pass 1: GBuffer (opaque geometry → MRT) ----
        with graph.add_pass("GBufferPass") as p:
            write_deferred_gbuffer(p, surface)
            _draw_deferred_geometry(p, DEPTH_TEXTURE, geometry_queues, geometry_sort, "gbuffer")

        geometry_buffers = {
            **surface,
            "color": graph.get_texture(COLOR_TEXTURE),
            "depth": graph.get_texture(DEPTH_TEXTURE),
            "shadow_map": graph.get_texture(SHADOW_MAP_TEXTURE),
        }
        if LIGHT_LIST_BUFFER in self._required_geometry_buffers(graph):
            geometry_buffers[LIGHT_LIST_BUFFER] = graph.create_view_light_list()
        current = self.geometry_stage(
            graph,
            "gbuffer",
            buffers=geometry_buffers,
            queue_range=geometry_queues,
            sort_mode=geometry_sort,
            clear=True,
        )
        raw_gbuffer = current

        with graph.pass_result(current):
            resources = result_resources(current)
            graph.injection_point("after_gbuffer", resources=resources)
            graph.effects(
                "after_gbuffer", scope="stage", display_name="After GBuffer",
                inputs=resources, outputs=resources - {SURFACE_RECORDS_BUFFER, LIGHT_LIST_BUFFER, _RAY_PRIMARY_SURFACE},
                capabilities={"fullscreen", "multiple_render_targets"},
            )
            current = graph.current_pass_result

        ray_source = current
        signals = {}
        if lobes:
            if any(current.sample(name) is not raw_gbuffer.sample(name) for name in (DEPTH_TEXTURE, "surface")):
                raise ValueError("Ray indirect lighting requires after_gbuffer to retain depth and surface identity")
            packets = _trace_ray_indirect(graph, current, lobes, temporal)
            current = graph.derive_pass_result("ray_indirect", current, _decode_raw_signals(graph, packets))
            signals = packets
            if temporal:
                signals = _reconstruct_ray_indirect(graph, raw_gbuffer.source, lobes, packets, primary_history,
                                                    ray_source.sample("surface_frame"))
                current = graph.derive_pass_result("ray_reconstructed", current, signals)

        # ---- Pass 2: Deferred lighting (fullscreen) ----
        with graph.add_pass("DeferredLightingPass") as p:
            p.write_color(COLOR_TEXTURE)
            p.set_clear(color=DEFERRED_LIGHTING_CLEAR_COLOR)
            if signals:
                _draw_ray_lighting(p, current, signals)
            else:
                draw_deferred_lighting(p, current.snapshot)

        # Opaque shading models are Deferred-compatible by default. Models
        # that explicitly declare Unsupported [Deferred] skip the GBuffer and
        # render here with the camera's Forward+ light list while sharing the
        # same scene color/depth. This is a real topology fallback, not an
        # error-material substitution inside GBufferPass.
        with graph.add_pass("DeferredForwardFallbackPass") as p:
            p.write_color(COLOR_TEXTURE)
            p.write_depth(DEPTH_TEXTURE)
            p.draw_renderers(
                queue_range=opaque_queue_range(),
                sort_mode="front_to_back",
                material_pass="forward_plus",
                material_filter="deferred_unsupported",
            )

        opaque_buffers = {"color": graph.get_texture(COLOR_TEXTURE)}
        if current.has(NORMAL_TEXTURE):
            # Preserve the after_gbuffer result (including user normal writes),
            # then replace only visible DeferredUnsupported pixels against the
            # final opaque depth. Untouched pixels retain their coverage alpha.
            opaque_buffers[NORMAL_TEXTURE] = add_normal_buffer_pass(
                graph,
                source="opaque_lighting",
                depth=graph.get_texture(DEPTH_TEXTURE),
                queue_range=opaque_queue_range(),
                material_filter="deferred_unsupported",
                initial_normal=current.sample(NORMAL_TEXTURE),
            )
        current = graph.derive_pass_result("opaque_lighting", current, opaque_buffers)
        with graph.pass_result(current):
            resources = result_resources(current)
            graph.injection_point("after_opaque", resources=resources)
            graph.effects(
                "after_opaque", scope="stage",
                display_name="After Opaque Lighting", inputs=resources,
                outputs={"color"}, capabilities={"fullscreen"},
            )
            current = graph.current_pass_result

        # ---- Pass 3: Skybox ----
        add_skybox_pass(graph)
        current = graph.derive_pass_result("sky", current, {"color": graph.get_texture(COLOR_TEXTURE)})
        with graph.pass_result(current):
            resources = result_resources(current)
            graph.injection_point("after_sky", resources=resources)
            graph.effects(
                "after_sky", scope="composite", display_name="After Sky",
                inputs=resources, outputs={"color"}, capabilities={"fullscreen"},
            )
            current = graph.current_pass_result

        capture = _capture_thin_transparency(graph) if self.ray_transparency else None
        if lobes or capture:
            _update_ray_tail(graph, ray_source, opaque=bool(lobes), thin=capture)

        # ---- Pass 4: Transparent objects (Forward+ rendering) ----
        if self.ray_transparency:
            _draw_thin_transparency(graph, current.snapshot[SURFACE_RECORDS_BUFFER], capture)
        else:
            add_transparent_pass(graph, material_pass="forward_plus")
        current = graph.derive_pass_result(
            "transparent", current, {"color": graph.get_texture(COLOR_TEXTURE)}
        )
        with graph.pass_result(current):
            resources = result_resources(current)
            graph.injection_point("after_transparent", resources=resources)
            graph.effects(
                "after_transparent", scope="composite",
                display_name="After Transparent", inputs=resources,
                outputs={"color"}, capabilities={"fullscreen"},
            )
            current = graph.current_pass_result

        # ---- Camera UI + post-process + Screen UI tail ----
        with graph.pass_result(current):
            add_standard_post_process_section(graph, current)

        graph.set_output(COLOR_TEXTURE)
