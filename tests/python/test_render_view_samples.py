"""Camera output contracts specialize topology, never author parameters."""

import pytest
import re
from pathlib import Path

from infernux.core.asset_ref import RenderEffectRef
from infernux.core.assets import AssetManager
from infernux.renderstack.render_effect import RenderEffect
from infernux.renderstack.render_effect_asset import RenderEffectAsset
from infernux.renderstack.render_pipeline import RenderPipeline
from infernux.renderstack.render_stack import RenderStack
from infernux.renderstack.render_stack_pipeline import RenderStackPipeline


def _stack(pipeline=None):
    stack = RenderStack()
    if pipeline is not None:
        stack._pipeline = pipeline
        pipeline._render_stack = stack
    stack.add_effect_slot(
        "final", RenderEffectRef(effect=RenderEffect(RenderEffectAsset(feature_type="infernux.post.temporal_aa")))
    )
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


@pytest.mark.parametrize("samples,declarative", [(1, False), (4, False), (1, True), (8, True)])
def test_taa_motion_and_resolves_follow_camera_target(samples, declarative):
    stack = _stack(_DslPipeline() if declarative else None)
    _assert_motion_samples(stack.build_graph(output_samples=samples), samples)
    # Building a target variant cannot rewrite the author's screen quality.
    _assert_motion_samples(stack.build_graph(), 4)
    if not declarative:
        assert int(stack.pipeline.msaa_samples) == 4


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
        assert any(dict(update.values).get("exposure") == 2.0 for update in context.parameter_updates)
        assert len(context.applied) == 1
