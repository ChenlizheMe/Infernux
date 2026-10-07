"""RenderGraph authoring declarations match the shipped runtime entry points."""
import ast
import importlib
import inspect
from pathlib import Path

import pytest

from infernux import rendergraph
from infernux.rendergraph import graph as graph_module


def _stub(module):
    return ast.parse(Path(module.__file__).with_suffix('.pyi').read_text(encoding='utf-8'))


def _method(name):
    cls = next(node for node in _stub(graph_module).body
               if isinstance(node, ast.ClassDef) and node.name == 'RenderPassBuilder')
    return next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)


@pytest.mark.parametrize('name', ['draw_shadow_casters', 'draw_screen_ui'])
def test_pass_builder_stub_has_the_runtime_parameters(name):
    declared = [arg.arg for arg in _method(name).args.args]
    actual = list(inspect.signature(getattr(rendergraph.RenderPassBuilder, name)).parameters)
    assert declared == actual


def test_screen_ui_selector_declaration_matches_runtime_string_contract():
    selector = next(arg for arg in _method('draw_screen_ui').args.args if arg.arg == 'list')
    assert ast.unparse(selector.annotation) == 'str'


def test_rendergraph_stub_exports_all_public_runtime_symbols():
    tree = _stub(rendergraph)
    exports = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '__all__' for t in node.targets))
    assert set(exports) == set(rendergraph.__all__)
    imports = {alias.asname or alias.name: (node, alias.name)
               for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names}
    for name in exports:
        declaration, imported_name = imports[name]
        module = importlib.import_module('.' * declaration.level + declaration.module, rendergraph.__package__)
        assert getattr(module, imported_name) is getattr(rendergraph, name)


def test_pixel_format_members_are_typed_as_the_runtime_enum():
    from infernux.lib import _Infernux as native

    stub = Path(graph_module.__file__).parents[1] / 'lib' / '_Infernux.pyi'
    cls = next(node for node in ast.parse(stub.read_text(encoding='utf-8')).body
               if isinstance(node, ast.ClassDef) and node.name == 'PixelFormat')
    members = {node.target.id: ast.unparse(node.annotation) for node in cls.body
               if isinstance(node, ast.AnnAssign)}
    assert set(members) == set(native.PixelFormat.__members__)
    for name, annotation in members.items():
        assert annotation == 'PixelFormat'
        assert isinstance(getattr(native.PixelFormat, name), native.PixelFormat)


@pytest.mark.parametrize('selector,expected', [('camera', 0), ('overlay', 1)])
def test_public_shadow_and_screen_ui_calls_build_real_native_description(selector, expected):
    graph = rendergraph.RenderGraph('Typed public graph')
    depth = graph.create_texture('shadow', format=rendergraph.Format.D32_SFLOAT, size=(32, 32))
    color = graph.create_texture('color', camera_target=True)
    graph.add_pass('shadow').write_depth(depth).set_clear(depth=1.0).draw_shadow_casters(queue_range=(0, 2999), light_index=0)
    graph.add_pass('ui').write_color(color).draw_screen_ui(list=selector)
    graph.set_output(color)
    built = graph.build()
    assert len(built.passes) == 2
    assert built.passes[1].commands[0].screen_ui_list == expected
