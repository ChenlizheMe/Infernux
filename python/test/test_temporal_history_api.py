"""Public TAA declarations project the existing runtime API, not a second effect."""
import ast
import inspect
from pathlib import Path

from Infernux import renderstack
from Infernux.rendergraph.graph import RenderGraph
from Infernux.renderstack.temporal_aa_effect import TemporalAAEffect


def test_temporal_effect_is_exported_by_runtime_and_stub():
    assert renderstack.TemporalAAEffect is TemporalAAEffect
    assert 'TemporalAAEffect' in renderstack.__all__
    tree = ast.parse(Path(renderstack.__file__).with_suffix('.pyi').read_text(encoding='utf-8'))
    assert any(isinstance(n, ast.ImportFrom) and n.module == TemporalAAEffect.__module__
               and any(a.name == a.asname == 'TemporalAAEffect' for a in n.names) for n in tree.body)
    names = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '__all__' for t in n.targets))
    assert 'TemporalAAEffect' in ast.literal_eval(names)
    stub = Path(inspect.getfile(TemporalAAEffect)).with_suffix('.pyi')
    declaration = next(n for n in ast.parse(stub.read_text(encoding='utf-8')).body
                       if isinstance(n, ast.ClassDef) and n.name == 'TemporalAAEffect')
    floats = {n.target.id for n in declaration.body if isinstance(n, ast.AnnAssign)
              and isinstance(n.annotation, ast.Name) and n.annotation.id == 'float'}
    assert floats == {'feedback', 'motion_rejection', 'depth_rejection'}
    assert floats == set(TemporalAAEffect.__annotations__)


def test_temporal_history_stub_matches_keyword_only_runtime_signature():
    stub = Path(inspect.getfile(RenderGraph)).with_suffix('.pyi')
    declaration = next(n for n in ast.parse(stub.read_text(encoding='utf-8')).body
                       if isinstance(n, ast.ClassDef) and n.name == 'RenderGraph')
    method = next(n for n in declaration.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'create_temporal_history')
    signature = inspect.signature(RenderGraph.create_temporal_history)
    positional = [n for n, p in signature.parameters.items() if p.kind == p.POSITIONAL_OR_KEYWORD]
    keywords = [n for n, p in signature.parameters.items() if p.kind == p.KEYWORD_ONLY]
    assert [arg.arg for arg in method.args.args] == positional == ['self', 'name']
    assert [arg.arg for arg in method.args.kwonlyargs] == keywords == ['format', 'size', 'size_divisor']
    assert method.args.vararg is None and method.args.kwarg is None
    assert ast.unparse(method.returns) == signature.return_annotation
