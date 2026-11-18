"""Isolated graph-builder state for replaying one mounted effect's values."""
from __future__ import annotations

from itertools import compress

from infernux.lib import GraphTextureDesc
from infernux.rendergraph.graph import BufferHandle, RenderGraph, RenderPassBuilder, TextureHandle, WorkBinding
from infernux.renderstack.pass_result import PassResult
from infernux.renderstack.resource_bus import ResourceBus


class _NewEffectInput(Exception):
    """A parameter edit first requested an input absent from the recorded graph."""


def _require_graph_rebuild(*_):
    raise _NewEffectInput


def _fork(graph, resources):
    # Clone builder-owned Python state, keeping imported GPU owners borrowed.
    # deepcopy would attempt to copy native allocations and renderer handles.
    memo = {}

    def clone(value):
        identity = id(value)
        if identity in memo:
            return memo[identity]
        if isinstance(value, (RenderGraph, RenderPassBuilder, TextureHandle, BufferHandle, WorkBinding, PassResult)):
            if isinstance(value, TextureHandle):
                result = GraphTextureDesc.__new__(type(value))
                GraphTextureDesc.__init__(result, value)
            else:
                result = object.__new__(type(value))
            memo[identity] = result
            result.__dict__.update({key: clone(item) for key, item in vars(value).items()})
            if isinstance(result, RenderGraph):
                # These closures compile pipeline stages into the live graph.
                # A value replay records this effect, not the enclosing stack.
                result._injection_callback = None
                result._effect_stage_callback = None
            elif isinstance(result, PassResult):
                # Inputs actually consumed by setup are frozen after its first
                # recording. A replay must never invoke a live graph closure.
                if value._materialize is not None:
                    result._materialize = _require_graph_rebuild
                result._providers = dict.fromkeys(value._providers, _require_graph_rebuild)
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
        if isinstance(value, WorkBinding):
            return ("work_binding", value.name)
        if isinstance(value, dict):
            return tuple((key, freeze(item)) for key, item in value.items())
        if isinstance(value, (tuple, list)):
            return tuple(freeze(item) for item in value)
        if isinstance(value, set):
            return frozenset(value)
        return value

    return (
        tuple((GraphTextureDesc(handle), freeze(vars(handle))) for handle in sorted(graph._textures, key=lambda x: x.name)),
        tuple(freeze(vars(handle)) for handle in sorted(graph._buffers, key=lambda x: x.name)),
        tuple((
            binding.name,
            # Keep the immutable recipe alive as well as comparing identity:
            # equal port layouts do not imply equal prepared computation.
            id(binding._native), binding._native,
            tuple(sorted((name, source.name, source.byte_offset, source.byte_size)
                         for name, source in binding._input_bindings.items())),
        ) for binding in sorted(graph._work_bindings, key=lambda x: x.name)),
        tuple(freeze({
            key: tuple(value) if key == "_parameters" else value
            for key, value in vars(render_pass).items()
            if key not in {"_graph", "_order", "_parameter_block"}
        }) for render_pass in sorted(graph._passes, key=lambda p: p._order)),
        tuple(graph.topology_sequence),
        graph._output, graph._linear_output, graph._msaa_samples,
        graph._temporal_jitter,
    )


class EffectGraphCheckpoint:
    """The mount owns its declarations. Lazy inputs belong to their publishing stage."""

    def __init__(self, graph, scope):
        self.prefix = scope + "/"
        self.order = (*graph._order_prefix, graph._topology_serial)
        self.first_pass = graph.pass_count
        self.first_topology = len(graph._topology)
        self.options = {key: getattr(graph, key) for key in (
            "_output", "_linear_output", "_msaa_samples", "_temporal_jitter")}

    def passes(self, graph):
        return [p for p in graph._passes[self.first_pass:] if p.name.startswith(self.prefix)]

    def discard_mount(self, graph):
        for items in (graph._passes, graph._textures, graph._buffers, graph._work_bindings):
            items[:] = [item for item in items if not item.name.startswith(self.prefix)]
        # Descendants of earlier publications can be appended during setup.
        # Compare their publication position at the depth of this mount.
        additions = graph._topology[self.first_topology:]
        keep = [order[:len(self.order)] <= self.order for order, _, _ in additions]
        for kind, declarations in (("ip", graph._injection_points_list),
                                   ("effect_stage", graph._effect_stages_list)):
            flags = [retain for (_, entry_kind, _), retain in zip(additions, keep) if entry_kind == kind]
            if flags:
                declarations[-len(flags):] = compress(declarations[-len(flags):], flags)
        graph._topology[self.first_topology:] = compress(additions, keep)
        graph._pass_results = {
            name: result for name, result in graph._pass_results.items()
            if not result._name_scopes or not (result._name_scopes[-1] + "/").startswith(self.prefix)
        }
        for key, value in self.options.items():
            setattr(graph, key, value)


class EffectParameterContext:
    """Frozen external inputs, with this mount's own declarations removed."""

    def __init__(self, graph, resources, mount):
        self._graph, frozen_bus = _fork(graph, resources)
        mount.discard_mount(self._graph)
        self._resources = frozen_bus.snapshot()
        self._mount = mount

    def record(self, source, feature, binding_id):
        graph, bus = _fork(self._graph, self._resources)
        try:
            with graph.name_scope(f"effects/{binding_id}"):
                feature.instantiate(source).setup_passes(graph, bus)
        except _NewEffectInput:
            return None
        return self._mount.passes(graph), graph_structure(graph)
