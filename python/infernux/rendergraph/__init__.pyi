"""Type stubs for infernux.rendergraph."""

from __future__ import annotations

from .graph import BufferHandle, DepthCompare, Format, RenderGraph, RenderPassBuilder, TextureHandle
from .renderer_selection import RendererSelection
from infernux.lib import DrawParameterBlock
from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline

__all__ = [
    "RenderGraph",
    "RendererSelection",
    "DrawParameterBlock",
    "RenderPassBuilder",
    "TextureHandle",
    "BufferHandle",
    "Format",
    "DepthCompare",
    "DefaultForwardPipeline",
]
