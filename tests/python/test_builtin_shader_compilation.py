"""Compile builtin Python shader stages and their real target resource contracts."""
from pathlib import Path
import json

import pytest

from infernux.lib import _Infernux as native
from infernux._compiler.shader import _prepare_source


SHADERS = Path(__file__).parents[2] / 'python/infernux/resources/shaders'
STAGES = sorted((*SHADERS.glob('*.frag.py'), *SHADERS.glob('*.vert.py')))


@pytest.mark.parametrize('path', STAGES, ids=lambda path:path.name)
def test_builtin_authored_stage_compiles(path):
    text = path.read_text(encoding='utf-8')
    vertex = None
    source_path = str(path)
    if path.name.endswith('.vert.py'):
        vertex = (str(path), text)
        text = '''import infernux.shader as shader
class VertexProbe(shader.frag):
    def main(self, ctx: shader.mesh_input) -> shader.vec4:
        return shader.vec4(ctx.uv0, 0.0, 1.0)
'''
        source_path = str(SHADERS / 'vertex_probe.frag.py')
    source, libraries, schema, ranges = _prepare_source(text, source_path, 'builtin_stage', None, vertex_source=vertex)
    compiled = native._compile_shader_program(source, str(path), 'builtin_stage', libraries, ('spirv', 'wgsl'),
                                              source_ranges=ranges)
    assert {target['target'] for target in compiled['targets']} == {'spirv', 'wgsl'}
    for target in compiled['targets']:
        assert target['entries']
        for entry in target['entries']:
            assert entry['code']
            if target['target'] == 'spirv':
                assert entry['code'][:4] == b'\x03\x02\x23\x07'
                assert len(entry['code']) >= 20 and len(entry['code']) % 4 == 0
        if json.loads(schema)['domain'] != 'particle':
            continue
        entries = {entry['name']: entry for entry in target['entries']}
        for suffix in ('main', 'soft_depth', 'soft_color', 'depth', 'picking', 'motion'):
            bindings = {tuple(binding) for binding in entries['fragment_' + suffix]['bindings']}
            assert bindings.isdisjoint({(1, 1), (1, 2), (1, 3)}), (path.name, target['target'], suffix, bindings)
        if path.name in {'particle_lit.frag.py', 'particle_six_way_smoke.frag.py'}:
            for suffix in ('', '_soft_depth', '_soft_color'):
                bindings = {tuple(binding) for binding in entries['fragment_forward_plus' + suffix]['bindings']}
                assert {(1, 1), (1, 2), (1, 3)} <= bindings, (path.name, target['target'], suffix, bindings)
