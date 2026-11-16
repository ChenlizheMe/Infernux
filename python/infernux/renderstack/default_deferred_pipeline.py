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
    create_deferred_gbuffer,
    create_main_scene_targets,
    draw_deferred_lighting,
    opaque_queue_range,
    result_resources,
    write_deferred_gbuffer,
)

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph


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

        shadow_res = effective_shadow_resolution(self.shadow_resolution)

        # ---- GBuffer textures (MRT) ----
        create_main_scene_targets(
            graph,
            shadow_resolution=shadow_res,
            msaa_samples=1,
        )
        surface = create_deferred_gbuffer(graph)

        # ---- Pass 0: Shadow casters ----
        add_shadow_caster_pass(graph)

        # ---- Pass 1: GBuffer (opaque geometry → MRT) ----
        with graph.add_pass("GBufferPass") as p:
            write_deferred_gbuffer(p, surface)
            p.write_depth(DEPTH_TEXTURE)
            p.set_clear(
                color=DEFERRED_GBUFFER_CLEAR_COLOR,
                depth=0.0,
            )
            p.draw_renderers(
                queue_range=opaque_queue_range(),
                sort_mode="front_to_back",
                material_pass="gbuffer",
                material_filter="deferred_compatible",
            )

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
            queue_range=opaque_queue_range(),
            clear=True,
        )

        with graph.pass_result(current):
            resources = result_resources(current)
            graph.injection_point("after_gbuffer", resources=resources)
            graph.effects(
                "after_gbuffer", scope="stage", display_name="After GBuffer",
                inputs=resources, outputs=resources - {SURFACE_RECORDS_BUFFER, LIGHT_LIST_BUFFER},
                capabilities={"fullscreen", "multiple_render_targets"},
            )
            current = graph.current_pass_result

        # ---- Pass 2: Deferred lighting (fullscreen) ----
        with graph.add_pass("DeferredLightingPass") as p:
            p.write_color(COLOR_TEXTURE)
            p.set_clear(color=DEFERRED_LIGHTING_CLEAR_COLOR)
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
