"""
Infernux RenderGraph module.

Provides the Python API used to define render-graph topology.
Python describes pass layout, resource connections, and per-pass actions.
C++ compiles the graph, inserts barriers, and manages transient resources.

Usage::

    from infernux.rendergraph import RenderGraph, Format

    graph = RenderGraph("ForwardPipeline")

        graph.create_texture("color", camera_target=True)
    graph.create_texture("depth", format=Format.D32_SFLOAT, camera_target=True)

    with graph.add_pass("OpaquePass") as p:
        p.write_color("color")
        p.write_depth("depth")
        p.set_clear(color=(0.1, 0.1, 0.1, 1.0), depth=0.0)
        p.draw_renderers(queue_range=(0, 2500), sort_mode="front_to_back")

    with graph.add_pass("SkyboxPass") as p:
        p.read("depth")
        p.write_color("color")
        p.draw_skybox()

    with graph.add_pass("TransparentPass") as p:
        p.read("depth")
        p.write_color("color")
        p.draw_renderers(queue_range=(2501, 5000), sort_mode="back_to_front")

    graph.set_output("color")

    # Apply to the engine's scene render graph
    scene_graph = engine.get_scene_render_graph()
    scene_graph.apply_python_graph(graph.build())
"""

from .graph import BufferHandle, BufferInput, RenderGraph, RenderPassBuilder, TextureHandle, WorkBinding, Format, DepthCompare
from .renderer_selection import RendererSelection, DrawParameterBlock
from .buffer_source import BufferSource, capture_buffer
from .events import RenderEvent, events
from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline


def apply_result(request, targets, apply, *, required=False):
    """Apply a managed result once at a scene render extraction boundary.

    Targets identify the native components whose render inputs will change.
    Pending work leaves rendering advancing by default. Set ``required=True``
    when the next extraction depends on this result.
    Registration after extraction affects the following render extraction.
    """
    from infernux.application import Application
    engine = Application._current_engine()
    if engine is None:
        raise RuntimeError('render result application requires an active Engine')
    return engine._apply_result(request, targets, apply, boundary='render', required=required)


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
    "WorkBinding",
    "Format",
    "DepthCompare",
    "DefaultForwardPipeline",
]
