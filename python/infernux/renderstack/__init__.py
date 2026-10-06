"""
Infernux RenderStack Module

Scene-level rendering configuration system.
Provides a composable, per-scene rendering stack where users mount reusable
Effect assets to pipeline-defined EffectStages.

Core classes:
    - **RenderStack**: Scene-singleton component managing pipeline + effects
    - **RenderPipeline**: Topology skeleton with named EffectStages
    - **ResourceBus**: Transient resource handle dictionary
    - **InjectionPoint**: Named slot in the pipeline topology

Architecture::

    Scene
    └── RenderStack (InxComponent)
        ├── selected_pipeline: RenderPipeline
        │   └── define_topology(graph)
        └── effect_slots: [EffectSlot, ...]

Quick start::

    from infernux.renderstack import RenderStack

    # Mount to scene's RenderStack
    stack = game_object.add_component(RenderStack)

See Also:
    - ``docs/design/RenderStack_Design.md`` for the full design document
"""

from infernux.renderstack.injection_point import InjectionPoint
from infernux.renderstack.effect_stage import EffectResourceContract, EffectScope, EffectStage
from infernux.renderstack.effect_slot import EffectSlot
from infernux.renderstack.render_effect import RenderEffect
from infernux.renderstack.render_effect_asset import (
    EffectAssetReference,
    RenderEffectAsset,
    RenderEffectGroupAsset,
    RenderEffectGroupEntry,
    direct_effect_dependencies,
    dump_render_effect_document,
    parse_render_effect_document,
)
from infernux.renderstack.render_effect_compiler import (
    RenderEffectArtifact,
    RenderEffectArtifactRegistry,
    RenderEffectCompileError,
    RenderEffectFeature,
    get_render_effect_feature,
    render_effect_feature,
    register_render_effect_feature,
)
from infernux.renderstack.resource_bus import ResourceBus
from infernux.renderstack.pass_result import BufferHandle, PassResult
from infernux.renderstack.geometry_buffers import (
    GeometryBufferTopologyError,
    GeometryStagePhase,
    geometry_buffer,
)
from infernux.renderstack.render_pass import RenderPass
from infernux.renderstack.render_pipeline import RenderPipeline, RenderPipelineAsset
from infernux.renderstack.pipeline_dsl import (
    Path,
    PipelineBuilder,
    PipelineDefinition,
    Queue,
    QueueSelector,
    compile_queue_segments,
)
from infernux.renderstack.route_policy import RoutePolicy, merge_route_policies
from infernux.renderstack.geometry_pass import GeometryPass
from infernux.renderstack.fullscreen_effect import EffectColorComposition, FullScreenEffect
from infernux.renderstack.bloom_effect import BloomEffect
from infernux.renderstack.pixelation_effect import PixelationEffect, PixelationSampling
from infernux.renderstack.tonemapping_effect import ToneMappingEffect
from infernux.renderstack.vignette_effect import VignetteEffect
from infernux.renderstack.color_adjustments_effect import ColorAdjustmentsEffect
from infernux.renderstack.chromatic_aberration_effect import ChromaticAberrationEffect
from infernux.renderstack.film_grain_effect import FilmGrainEffect
from infernux.renderstack.motion_blur_effect import MotionBlurEffect
from infernux.renderstack.temporal_aa_effect import TemporalAAEffect
from infernux.renderstack.white_balance_effect import WhiteBalanceEffect
from infernux.renderstack.sharpen_effect import SharpenEffect
from infernux.renderstack.render_stack import RenderStack
from infernux.renderstack.render_stack_pipeline import RenderStackPipeline
from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline
from infernux.renderstack.default_forward_plus_pipeline import DefaultForwardPlusPipeline
from infernux.renderstack.default_deferred_pipeline import DefaultDeferredPipeline
from infernux.renderstack.discovery import (
    discover_passes,
    discover_pipelines,
    discovery_import_failures,
    discovery_name_conflicts,
)

__all__ = [
    # Core
    "RenderStack",
    "RenderPipeline",
    "RenderPipelineAsset",
    "Path",
    "PipelineBuilder",
    "PipelineDefinition",
    "Queue",
    "QueueSelector",
    "compile_queue_segments",
    "RoutePolicy",
    "merge_route_policies",
    "RenderStackPipeline",
    "DefaultForwardPipeline",
    "DefaultForwardPlusPipeline",
    "DefaultDeferredPipeline",
    # Injection points
    "InjectionPoint",
    "EffectStage",
    "EffectScope",
    "EffectResourceContract",
    "EffectSlot",
    "RenderEffect",
    "EffectAssetReference",
    "RenderEffectAsset",
    "RenderEffectGroupAsset",
    "RenderEffectGroupEntry",
    "parse_render_effect_document",
    "dump_render_effect_document",
    "direct_effect_dependencies",
    "RenderEffectCompileError",
    "RenderEffectArtifact",
    "RenderEffectArtifactRegistry",
    "RenderEffectFeature",
    "get_render_effect_feature",
    "render_effect_feature",
    "register_render_effect_feature",
    # Resource bus
    "ResourceBus",
    "BufferHandle",
    "PassResult",
    "GeometryBufferTopologyError",
    "GeometryStagePhase",
    "geometry_buffer",
    # Pass base classes
    "RenderPass",
    "GeometryPass",
    "FullScreenEffect",
    "EffectColorComposition",
    # Built-in effects
    "BloomEffect",
    "PixelationEffect",
    "PixelationSampling",
    "ToneMappingEffect",
    "VignetteEffect",
    "ColorAdjustmentsEffect",
    "ChromaticAberrationEffect",
    "FilmGrainEffect",
    "MotionBlurEffect",
    "TemporalAAEffect",
    "WhiteBalanceEffect",
    "SharpenEffect",
    # Discovery
    "discover_pipelines",
    "discover_passes",
    "discovery_import_failures",
    "discovery_name_conflicts",
]
