"""Committing an effect must preserve pixels or explicitly resample its extent."""
import pytest

from infernux.components.fields import serialized_field
from infernux.core.asset_ref import RenderEffectRef
from infernux.rendergraph.graph import Format
from infernux.renderstack import FullScreenEffect, RenderPipeline, RenderStack, render_effect_feature
from infernux.renderstack.render_effect import RenderEffect
from infernux.renderstack.render_effect_asset import RenderEffectAsset


@render_effect_feature('test.copy_output_extent', topology_parameters=('divisor', 'fixed_size'))
class ExtentEffect(FullScreenEffect):
    name = 'Output Extent Probe'
    injection_point = 'final'
    divisor: int = serialized_field(default=0)
    fixed_size: bool = serialized_field(default=False)

    def setup_passes(self, graph, bus):
        output = graph.create_texture(
            'probe', format=Format.RGBA16_SFLOAT,
            size=(64, 48) if self.fixed_size else None,
            size_divisor=self.divisor,
        )
        with graph.add_pass('ResizeProbe') as render_pass:
            render_pass.set_texture('_SourceTex', bus.get('color'))
            render_pass.write_color(output)
            render_pass.fullscreen_quad('Fullscreen Blit')
        bus.set('color', output)


@pytest.mark.parametrize(('parameters', 'shader'), [
    ({}, 'Fullscreen Copy'),
    ({'divisor': 2}, 'Fullscreen Blit'),
    ({'fixed_size': True}, 'Fullscreen Blit'),
])
def test_effect_commit_uses_the_declared_output_extent(parameters, shader):
    class Pipeline(RenderPipeline):
        name = 'Extent Test'

        def define(self, pipeline):
            pipeline.frame(hdr=True)
            pipeline.opaque().forward()
            pipeline.effects('final')

    stack = RenderStack()
    stack._pipeline = Pipeline()
    stack._pipeline._render_stack = stack
    effect = RenderEffect(RenderEffectAsset(feature_type='test.copy_output_extent', parameters=parameters))
    stack.add_effect_slot('final', RenderEffectRef(effect=effect))
    description = stack.build_graph()
    assert not stack.effect_compile_errors
    commit = next(p for p in description.passes if p.name == 'effects/final/Commit')
    assert commit.commands[0].shader_name == shader
