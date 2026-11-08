"""Type stubs for infernux.rendergraph."""

from __future__ import annotations

from .graph import BufferHandle, BufferInput, DepthCompare, Format, RenderGraph, RenderPassBuilder, TextureHandle
from .renderer_selection import RendererSelection
from .buffer_source import BufferSource as BufferSource, capture_buffer as capture_buffer
from .events import RenderEvent as RenderEvent, events as events
from infernux.lib import DrawParameterBlock
from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline

def apply_result(request, targets, apply, *, required: bool = False):
    """Apply once before scene render extraction; return a cancellable result consumer."""
    ...

__all__ = [
    "apply_result",
    "BufferSource",
    "capture_buffer",
    "RenderEvent",
    "events",
    "RenderGraph",
    "RendererSelection",
    "DrawParameterBlock",
    "RenderPassBuilder",
    "TextureHandle",
    "BufferHandle",
    "BufferInput",
    "Format",
    "DepthCompare",
    "DefaultForwardPipeline",
]
