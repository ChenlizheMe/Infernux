"""Isolated graph-builder state for replaying one mounted effect's values."""
from __future__ import annotations

from infernux.rendergraph.graph import BufferHandle, RenderGraph, RenderPassBuilder, TextureHandle
from infernux.renderstack.pass_result import PassResult
from infernux.renderstack.resource_bus import ResourceBus


def _fork(graph, resources):
    # Clone builder-owned Python state, keeping imported GPU owners borrowed.
    # deepcopy would attempt to copy native allocations and renderer handles.
    memo = {}

    def clone(value):
        identity = id(value)
        if identity in memo:
            return memo[identity]
        if isinstance(value, (RenderGraph, RenderPassBuilder, TextureHandle, BufferHandle, PassResult)):
            result = object.__new__(type(value))
            memo[identity] = result
            result.__dict__.update({key: clone(item) for key, item in vars(value).items()})
            if isinstance(result, RenderGraph):
                # These closures compile pipeline stages into the live graph.
                # A value replay records this effect, not the enclosing stack.
                result._injection_callback = None
                result._effect_stage_callback = None
            elif isinstance(result, PassResult):
                # Stage inputs are already materialized by the pipeline's
                # declared requirements. Never call its live provider closure.
                result._materialize = None
            return result
        if isinstance(value, dict):
            result = {}
            memo[identity] = result
            result.update((clone(key), clone(item)) for key, item in value.items())
            return result
        if isinstance(value, list):
            result = []
            memo[identity] = result
            result.extend(clone(item) for item in value)
            return result
        if isinstance(value, tuple):
            result = tuple(clone(item) for item in value)
            memo[identity] = result
            return result
        if isinstance(value, set):
            result = {clone(item) for item in value}
            memo[identity] = result
            return result
        return value

    forked = clone(graph)
    return forked, ResourceBus(clone(resources), graph=forked)


def graph_structure(graph):
    """Exact builder structure, excluding only dynamic parameter values."""
    def freeze(value):
        if isinstance(value, dict):
            return tuple((key, freeze(item)) for key, item in value.items())
        if isinstance(value, (tuple, list)):
            return tuple(freeze(item) for item in value)
        if isinstance(value, set):
            return frozenset(value)
        return value

    return (
        tuple(freeze(vars(handle)) for handle in graph._textures),
        tuple(freeze(vars(handle)) for handle in graph._buffers),
        tuple(freeze({
            key: tuple(value) if key == "_push_constants" else value
            for key, value in vars(render_pass).items()
            if key not in {"_graph", "_parameter_block"}
        }) for render_pass in graph._passes),
        tuple(graph._topology),
        graph._output, graph._linear_output, graph._msaa_samples,
        graph._temporal_jitter,
    )


class EffectParameterContext:
    """Frozen inputs captured before setup, independent for each mount."""

    def __init__(self, graph, bus):
        self._graph, frozen_bus = _fork(graph, bus.snapshot())
        self._resources = frozen_bus.snapshot()
        self._first_pass = graph.pass_count

    def record(self, source, feature, binding_id):
        graph, bus = _fork(self._graph, self._resources)
        with graph.name_scope(f"effects/{binding_id}"):
            feature.instantiate(source).setup_passes(graph, bus)
        return graph._passes[self._first_pass:], graph_structure(graph)
