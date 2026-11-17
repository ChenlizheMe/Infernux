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
    packets = {}
    for name, specular in lobes:
        output = graph.create_texture(f"_result/gbuffer/{name}_packet", format=Format.RGBA32_UINT, samples=1)
        label = "Ray Specular Radiance" if specular else "Ray Diffuse Irradiance"
        with graph.add_pass(label) as render_pass:
            render_pass.write_color(output)
            bind_deferred_surface(render_pass, gbuffer.snapshot)
            render_pass.set_texture("primary", gbuffer.sample(_RAY_PRIMARY_SURFACE))
            for key in ("world_ray_geometry", "world_ray_meshes", "ray_lit_materials", "ray_texture_descriptors"):
                render_pass.set_buffer(key, world[key])
            for key in ("ray_color_pages", "ray_data_pages"):
                render_pass.set_texture(key, world[key])
            render_pass.set_param("specular", specular)
            render_pass.set_param("temporal", temporal)
            render_pass.fullscreen_quad("infernux/internal/ray_indirect")
        packets[name] = output
    return packets


def _decode_raw_signals(graph, packets):
    raw = {}
    for name, packet in packets.items():
        output = graph.create_texture(f"_result/gbuffer/{name}", format=Format.RGBA16_SFLOAT, samples=1)
        with graph.add_pass(f"Decode {name} diagnostic") as render_pass:
            render_pass.set_texture("signal", packet)
            render_pass.write_color(output)
            render_pass.fullscreen_quad("infernux/internal/ray_raw_signal")
        raw[name] = output
    return raw


def _reconstruct_ray_indirect(graph, source, lobes, packets, primary_history):
    previous_primary, primary = primary_history
    motion = graph.get_texture(_ray_replay_name(source))
    signals = {}
    for name, specular in lobes:
        previous, output = graph.create_temporal_history(f"_{name}", format=Format.RGBA32_UINT)
        label = "Ray Specular Reconstruction" if specular else "Ray Diffuse Reconstruction"
        with graph.add_pass(label) as render_pass:
            render_pass.set_textures({
                "signal": packets[name], "primary": primary, "motion": motion,
                "previous_signal": previous, "previous_primary": previous_primary,
            })
            render_pass.write_color(output)
            render_pass.set_param("specular", specular)
            render_pass.fullscreen_quad("infernux/internal/ray_reconstruct")
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
    environment terms. Each enabled lobe accumulates one indirect sample per
    rendered view using per-view motion, surface and secondary-hit history.
    Lighting resolves a bounded neighborhood from that per-lobe estimate.
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

        lobes = tuple((name, specular) for name, specular, enabled in (
            (_RAY_DIFFUSE, False, self.ray_diffuse), (_RAY_SPECULAR, True, self.ray_specular),
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

        signals = {}
        if lobes:
            if any(current.sample(name) is not raw_gbuffer.sample(name) for name in (DEPTH_TEXTURE, "surface")):
                raise ValueError("Ray indirect lighting requires after_gbuffer to retain depth and surface identity")
            packets = _trace_ray_indirect(graph, current, lobes, temporal)
            current = graph.derive_pass_result("ray_indirect", current, _decode_raw_signals(graph, packets))
            signals = packets
            if temporal:
                signals = _reconstruct_ray_indirect(graph, raw_gbuffer.source, lobes, packets, primary_history)
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

        # ---- Pass 4: Transparent objects (Forward+ rendering) ----
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
