"""
DefaultDeferredPipeline — Standard deferred rendering pipeline.

Implements a classic deferred shading pipeline with a GBuffer pass
that writes multiple render targets (albedo, normals, material props),
followed by a fullscreen lighting pass, then transparent forward pass.

GBuffer layout (MRT)::

    Slot 0 — Lit Scene Color    (RGBA8_UNORM)
    Slot 1 — World Normals      (RGBA16_SFLOAT)
    Slot 2 — Material Params    (RGBA8_UNORM)
    Slot 3 — Emission           (RGBA16_SFLOAT)
    Depth  — Scene depth        (D32_SFLOAT)

Topology::

    ShadowCasterPass → GBufferPass → after_gbuffer
    → DeferredLightingPass → after_opaque
    → SkyboxPass → after_sky
    → TransparentPass (forward) → after_transparent
    → [post-process injection points]

Usage::

    stack = game_object.add_component(RenderStack)
    stack.set_pipeline("Default Deferred")

.. note::
    This pipeline requires the engine to support:
    - MRT (multiple render targets) — already supported
    - Depth buffer as shader input — now supported
    - Deferred lighting shader (``deferred_lighting.frag``)

    The deferred lighting shader is NOT yet shipped with the engine.
    Users must provide their own ``deferred_lighting`` shader, or this
    pipeline will fall back to a placeholder that outputs albedo only.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

from InfEngine.renderstack.render_pipeline import RenderPipeline
from InfEngine.components.serialized_field import serialized_field

if TYPE_CHECKING:
    from InfEngine.rendergraph.graph import RenderGraph


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

    enable_screen_ui: bool = serialized_field(
        default=True,
        tooltip="Enable screen-space UI rendering",
        header="Screen UI",
    )

    # ------------------------------------------------------------------
    # RenderPipeline interface
    # ------------------------------------------------------------------

    def define_topology(self, graph: "RenderGraph") -> None:
        """Define deferred rendering topology skeleton.

        Topology::

            ShadowCaster → GBuffer (MRT) → after_gbuffer
            → DeferredLighting → after_opaque → Skybox → after_sky
            → Transparent (forward) → after_transparent
        """
        from InfEngine.rendergraph.graph import Format

        # Deferred pipeline does not support MSAA on GBuffer
        graph.set_msaa_samples(1)

        shadow_res = self.shadow_resolution

        # ---- GBuffer textures (MRT) ----
        graph.create_texture("color", camera_target=True)
        graph.create_texture("depth", format=Format.D32_SFLOAT)
        graph.create_texture(
            "shadow_map",
            format=Format.D32_SFLOAT,
            size=(shadow_res, shadow_res),
        )
        graph.create_texture("gbuffer_albedo", format=Format.RGBA8_UNORM)
        graph.create_texture("gbuffer_normal", format=Format.RGBA16_SFLOAT)
        graph.create_texture("gbuffer_material", format=Format.RGBA8_UNORM)
        graph.create_texture("gbuffer_emission", format=Format.RGBA16_SFLOAT)

        # ---- Pass 0: Shadow casters ----
        with graph.add_pass("ShadowCasterPass") as p:
            p.write_depth("shadow_map")
            p.set_clear(depth=1.0)
            p.draw_shadow_casters(
                queue_range=(0, 2999),
                light_index=0,
                shadow_type="hard",
            )

        # ---- Pass 1: GBuffer (opaque geometry → MRT) ----
        with graph.add_pass("GBufferPass") as p:
            p.write_color("gbuffer_albedo", slot=0)
            p.write_color("gbuffer_normal", slot=1)
            p.write_color("gbuffer_material", slot=2)
            p.write_color("gbuffer_emission", slot=3)
            p.write_depth("depth")
            p.set_clear(
                color=(0.0, 0.0, 0.0, 0.0),
                depth=1.0,
            )
            p.draw_renderers(queue_range=(0, 2500), sort_mode="front_to_back")

        graph.injection_point(
            "after_gbuffer",
            resources={"gbuffer_albedo", "gbuffer_normal", "gbuffer_material", "gbuffer_emission", "depth"},
        )

        # ---- Pass 2: Deferred lighting (fullscreen) ----
        with graph.add_pass("DeferredLightingPass") as p:
            p.set_texture("gAlbedo", "gbuffer_albedo")
            p.set_texture("gNormal", "gbuffer_normal")
            p.set_texture("gMaterial", "gbuffer_material")
            p.set_texture("gEmission", "gbuffer_emission")
            p.set_texture("sceneDepth", "depth")
            p.set_texture("shadowMap", "shadow_map")
            p.write_color("color")
            p.set_clear(color=(0.0, 0.0, 0.0, 1.0))
            p.fullscreen_quad("deferred_lighting")

        graph.injection_point(
            "after_opaque",
            resources={"color", "depth"},
        )

        # ---- Pass 3: Skybox ----
        with graph.add_pass("SkyboxPass") as p:
            p.read("depth")
            p.write_color("color")
            p.draw_skybox()

        graph.injection_point("after_sky", resources={"color", "depth"})

        # ---- Pass 4: Transparent objects (forward rendering) ----
        with graph.add_pass("TransparentPass") as p:
            p.read("depth")
            p.write_color("color")
            p.set_texture("shadowMap", "shadow_map")
            p.draw_renderers(
                queue_range=(2501, 5000),
                sort_mode="back_to_front",
            )

        graph.injection_point(
            "after_transparent",
            resources={"color", "depth"},
        )

        # ---- ScreenUI + post-process injection points ----
        if self.enable_screen_ui:
            graph.screen_ui_section()
        else:
            graph.injection_point("before_post_process", resources={"color"})
            graph.injection_point("after_post_process", resources={"color"})

        graph.set_output("color")
