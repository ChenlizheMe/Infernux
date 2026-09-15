"""Camera output contracts specialize topology, never author parameters."""

import pytest
import re
from pathlib import Path
from types import SimpleNamespace

from Infernux.core.asset_ref import RenderEffectRef
from Infernux.core.assets import AssetManager
from Infernux.rendergraph.graph import RenderGraph
from Infernux.renderstack.default_deferred_pipeline import DefaultDeferredPipeline
from Infernux.renderstack.render_effect import RenderEffect
from Infernux.renderstack.render_effect_asset import RenderEffectAsset
from Infernux.renderstack.render_pipeline import RenderPipeline
from Infernux.renderstack.render_stack import RenderStack
from Infernux.renderstack.render_stack_pipeline import RenderStackPipeline


def _stack(pipeline=None):
    stack = RenderStack()
    if pipeline is not None:
        stack._pipeline = pipeline
        pipeline._render_stack = stack
    stack.add_effect_slot("final", RenderEffectRef(effect=RenderEffect(
        RenderEffectAsset(feature_type="infernux.post.temporal_aa"))))
    return stack


def _assert_motion_samples(description, samples):
    assert description.msaa_samples == samples
    assert description.temporal_jitter is True
    textures = {texture.name: texture for texture in description.textures}
    motion_passes = [p for p in description.passes if p.name.endswith("/Motion")]
    assert motion_passes
    for p in motion_passes:
        color = textures[dict(p.write_colors)[0]]
        assert color.samples == samples
        assert bool(p.resolve_color) == (samples > 1)
        if p.resolve_color:
            assert textures[p.resolve_color].samples == 1
        for name in p.read_textures:
            texture = textures[name]
            if texture.is_depth:
                assert (texture.samples or description.msaa_samples) == samples


class _DslPipeline(RenderPipeline):
    name = "View samples DSL test"

    def define(self, pipeline):
        pipeline.frame(hdr=True, msaa=4)
        pipeline.opaque().forward()
        pipeline.effects("final")
        pipeline.screen_ui()


@pytest.mark.parametrize("samples", [1, 2, 4, 8])
@pytest.mark.parametrize("declarative", [False, True])
def test_taa_motion_and_resolves_follow_camera_target(samples, declarative):
    stack = _stack(_DslPipeline() if declarative else None)
    _assert_motion_samples(stack.build_graph(output_samples=samples), samples)
    # Building a target variant cannot rewrite the author's screen quality.
    _assert_motion_samples(stack.build_graph(), 4)
    if not declarative:
        assert int(stack.pipeline.msaa_samples) == 4


def test_graph_returns_effective_samples_before_topology_is_authored():
    graph = RenderGraph(output_samples=1)
    assert graph.set_msaa_samples(4) == 1
    assert graph.set_msaa_samples(0) == 1
    assert RenderGraph().set_msaa_samples(4) == 4
    with pytest.raises(ValueError, match="output_samples"):
        RenderGraph(output_samples=3)


def test_deferred_rejects_unsupported_multisample_target_at_construction():
    with pytest.raises(ValueError, match="single-sample Camera target"):
        DefaultDeferredPipeline().define_topology(RenderGraph(output_samples=4))


class _Context:
    def __init__(self, samples, graph_id):
        self.output_samples = samples
        self.graph_instance_id = graph_id
        self.description = None
        self.applied = []
        self.parameter_updates = []

    def setup_camera_properties(self, camera):
        pass

    def cull(self, camera):
        return object()

    def apply_graph(self, description):
        self.description = description
        self.applied.append(description.source_revision)

    def submit_culling(self, culling):
        pass

    def render_compiled(self, camera, revision):
        return self.description is not None and self.description.source_revision == revision

    def render_with_graph(self, camera, description):
        self.apply_graph(description)

    def update_parameter_blocks(self, updates):
        self.parameter_updates.extend(updates)


def test_alternating_views_reuse_variants_and_invalidate_together(monkeypatch):
    monkeypatch.setattr(AssetManager, "refresh_pending", staticmethod(lambda: False))
    stack = _stack()
    contexts = [_Context(0, 1), _Context(1, 2), _Context(4, 3), _Context(1, 4)]
    for _ in range(4):
        for context in contexts:
            stack.render(context, object())
            _assert_motion_samples(context.description, context.output_samples or 4)
    assert all(len(context.applied) == 1 for context in contexts)
    assert contexts[1].description is contexts[3].description
    assert contexts[0].description is not contexts[2].description
    assert len(stack._graph_states) == 3
    old_revisions = [context.description.source_revision for context in contexts]
    stack.invalidate_graph()
    assert all(state.description is None for state in stack._graph_states.values())
    for context, old_revision in zip(contexts, old_revisions):
        stack.render(context, object())
        assert context.description.source_revision != old_revision
        assert len(context.applied) == 2


def test_effect_parameter_uploads_belong_to_variant_and_native_view(monkeypatch):
    monkeypatch.setattr(AssetManager, "refresh_pending", staticmethod(lambda: False))
    stack = RenderStack()
    effect = RenderEffect(RenderEffectAsset(feature_type="infernux.post.tonemapping"))
    stack.add_effect_slot("final", RenderEffectRef(effect=effect))
    contexts = [_Context(1, 10), _Context(4, 11), _Context(1, 12)]
    for _ in range(2):
        for context in contexts:
            stack.render(context, object())
    for context in contexts:
        context.parameter_updates.clear()
    effect.set_float("exposure", 2.0)
    for context in contexts:
        stack.render(context, object())
        assert any(dict(update.values).get("exposure") == 2.0
                   for update in context.parameter_updates)
        assert len(context.applied) == 1


def test_no_stack_default_uses_output_contract_cache():
    pipeline = RenderStackPipeline()
    contexts = [_Context(1, 10), _Context(4, 11)]
    for _ in range(3):
        for context in contexts:
            pipeline._render_default(context, object())
            assert context.description.msaa_samples == context.output_samples
    assert all(len(context.applied) == 1 for context in contexts)
    assert len(pipeline._default_graphs) == 2


def test_deserialization_invalidates_every_output_variant():
    stack = RenderStack()
    for samples in (0, 1, 4):
        description = stack.build_graph(output_samples=samples)
        stack._graph_state.description = description
    assert all(state.description is not None for state in stack._graph_states.values())
    stack.on_after_deserialize()
    assert all(state.description is None for state in stack._graph_states.values())


def test_standalone_pipeline_builds_once_per_output_contract():
    pipeline = _DslPipeline()

    class Context(_Context):
        def is_graph_revision_current(self, revision):
            return self.render_compiled(None, revision)

    contexts = [Context(1, 20), Context(4, 21)]
    for _ in range(3):
        for context in contexts:
            pipeline.render(context, object())
            assert context.description.msaa_samples == context.output_samples
    assert all(len(context.applied) == 1 for context in contexts)
    pipeline.dispose()
    assert pipeline._standalone_graphs == {}


@pytest.mark.parametrize("samples", [0, 1, 2, 4, 8])
def test_bilingual_msaa_example_builds_for_each_output_contract(samples):
    from Infernux import rendergraph

    guide = Path(__file__).parents[2] / "docs" / "learn" / "rendergraph-advanced.md"
    blocks = re.findall(r"```python\n(.*?)\n```", guide.read_text(encoding="utf-8"), re.DOTALL)
    examples = [block for block in blocks if block.startswith("samples = graph.set_msaa_samples(4)")]
    assert len(examples) == 2
    assert examples[0] == examples[1]
    graph = RenderGraph(output_samples=samples)
    exec(examples[0], {"graph": graph, "inx": SimpleNamespace(rendergraph=rendergraph)})
    description = graph.build()
    effective = samples or 4
    assert description.msaa_samples == effective
    route = next(p for p in description.passes if p.name == "Route")
    assert bool(route.resolve_color) == (effective > 1)
