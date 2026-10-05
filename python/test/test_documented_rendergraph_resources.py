"""Execute bilingual resource examples against the native graph description."""
from pathlib import Path
import re

import pytest
import infernux as inx
from infernux.lib import GraphBufferUsage


def _resource_examples(language):
    source = Path(__file__).parents[2] / 'docs/learn/rendergraph-advanced.md'
    english, chinese = source.read_text(encoding='utf-8').split('<!-- language:zh -->', 1)
    text = english if language == 'en' else chinese
    anchor = 'resource-usage' if language == 'en' else 'resource-usage_1'
    section = text.split('{#' + anchor + '}', 1)[1].split('\n## ', 1)[0]
    return re.findall(r'```python\n(.*?)```', section, re.S)


@pytest.mark.parametrize('language', ('en', 'zh'))
@pytest.mark.parametrize('samples', (1, 2, 4, 8))
def test_documented_resolve_uses_effective_camera_samples(language, samples):
    graph = inx.rendergraph.RenderGraph('Documented resolve', output_samples=samples)
    source = next(code for code in _resource_examples(language) if 'route_msaa' in code)
    namespace = {'inx': inx, 'graph': graph}
    exec(compile(source, 'rendergraph-advanced.md#resource-usage', 'exec'), namespace)
    description = graph.build()
    assert namespace['samples'] == samples
    assert description.msaa_samples == samples
    assert description.output_texture == 'route_color'
    route, = description.passes
    assert route.name == 'Route'
    assert route.write_depth == 'depth'
    if samples == 1:
        assert route.write_colors == [(0, 'route_color')]
        assert route.resolve_color == ''
    else:
        assert route.write_colors == [(0, 'route_msaa')]
        assert route.resolve_color == 'route_color'


@pytest.mark.parametrize('language', ('en', 'zh'))
def test_documented_buffer_creation_declares_all_requested_usages(language):
    graph = inx.rendergraph.RenderGraph('Documented usages')
    source = next(code for code in _resource_examples(language) if 'draw_data =' in code)
    namespace = {'graph': graph}
    exec(compile(source, 'rendergraph-advanced.md#resource-usage', 'exec'), namespace)
    buffer = namespace['draw_data']
    assert buffer.byte_size == 65536
    assert buffer.usage == (int(GraphBufferUsage.STORAGE) | int(GraphBufferUsage.INDIRECT) | int(GraphBufferUsage.TRANSFER_DESTINATION))


@pytest.mark.parametrize('usage', ('storage', 'indirect', 'transfer'))
def test_buffer_read_requires_corresponding_creation_flag(usage):
    graph = inx.rendergraph.RenderGraph('Invalid buffer contract')
    buffer = graph.create_buffer('data', 32, storage=False,
                                 indirect=usage != 'indirect',
                                 transfer_source=usage != 'transfer')
    graph.create_texture('color', camera_target=True)
    graph.add_pass('Use').write_color('color').read_buffer(buffer, usage).set_clear(color=(0, 0, 0, 1))
    graph.set_output('color')
    with pytest.raises(ValueError, match='usage|requires|not created'):
        graph.build()


@pytest.mark.parametrize('usage', ('storage', 'transfer'))
def test_buffer_write_requires_corresponding_creation_flag(usage):
    graph = inx.rendergraph.RenderGraph('Invalid buffer write')
    buffer = graph.create_buffer('data', 32, storage=False, indirect=True)
    graph.create_texture('color', camera_target=True)
    graph.add_pass('Use').write_color('color').write_buffer(buffer, usage).set_clear(color=(0, 0, 0, 1))
    graph.set_output('color')
    with pytest.raises(ValueError, match='usage|requires|not created'):
        graph.build()
