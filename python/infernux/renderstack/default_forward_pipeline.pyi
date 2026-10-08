"""Type stubs for infernux.renderstack.default_forward_pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from infernux.renderstack.render_pipeline import RenderPipeline
from infernux.renderstack.forward_parameters import DefaultForwardParameters, MSAASamples as MSAASamples

if TYPE_CHECKING:
    from infernux.rendergraph.graph import RenderGraph


class DefaultForwardPipeline(DefaultForwardParameters, RenderPipeline):
    """Standard forward rendering pipeline (default pipeline).

    Defines a standard forward rendering topology::

        ShadowCasterPass -> OpaquePass -> after_opaque -> SkyboxPass -> after_sky
        -> TransparentPass -> after_transparent

    Injection points:

    =============================  ==================================
    Injection Point                 Timing
    =============================  ==================================
    ``after_opaque``               After opaque objects, before skybox
    ``after_sky``                  After skybox, before transparent
    ``after_transparent``          After transparent objects
    =============================  ==================================

    Attributes:
        shadow_resolution: Shadow map resolution (default 4096).
        msaa_samples: Anti-aliasing sample count (X1/Off, X2, X4, or X8; default X4).
    """

    name: str
    shadow_resolution: int
    msaa_samples: MSAASamples

    def define_topology(self, graph: RenderGraph) -> None: ...
