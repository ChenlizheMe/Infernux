"""One typed ShaderLib definition can become the existing NN scalar row."""
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from infernux import nn
from infernux._compiler import slang
from infernux._compiler.diagnostics import ShaderSourceError
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor
from infernux.engine.project_context import using_project_root
from infernux.nn import _primitives as primitive


@pytest.fixture
def library(tmp_path):
    root = tmp_path / 'project'
    scripts = root / 'Assets' / 'Scripts'
    shaders = root / 'Assets' / 'Shaders'
    scripts.mkdir(parents=True)
    shaders.mkdir()
    caller = scripts / 'network.py'
    caller.write_text(
        'from infernux import compute\ncurve = compute.library("..Shaders.curve")\n', encoding='utf-8')
    leaf = shaders / 'leaf.shaderlib.py'
    leaf.write_text('import infernux.shader as shader\n'
                    'def bend(value: float) -> float:\n'
                    '    return shader.sin(value)\n', encoding='utf-8')
    entry = shaders / 'curve.shaderlib.py'
    entry.write_text('import infernux.shader as shader\nfrom .leaf import bend\n'
                     'def curve(value: float) -> float:\n'
                     '    return bend(value) + value * value * 0.125\n', encoding='utf-8')
    module = ModuleType('Scripts.network')
    # Cooked scripts may keep a project-relative filename in their bytecode.
    exec(compile(caller.read_text(encoding='utf-8'), 'Assets/Scripts/network.py', 'exec'), vars(module))
    return root, module.curve, entry, leaf


@pytest.mark.parametrize('target', ['spirv', 'wgsl'])
def test_nn_shaderlib_freezes_transitive_source_and_compiles_supported_derivatives(library, target):
    root, reference, entry, leaf = library
    with using_project_root(root):
        model = nn.SlangUnary(reference, entry_point='curve')
        original = model.options
        leaf.write_text(leaf.read_text(encoding='utf-8').replace('shader.sin', 'shader.cos'), encoding='utf-8')
        changed = nn.SlangUnary(reference, entry_point='curve')
        assert changed.options != original
        assert model.options == original
        modules = {Path(path): source for _, path, source in original[2]}
        assert set(modules) == {entry, leaf}
        assert all('[Differentiable]' in source for source in modules.values())
        assert 'return sin(' in modules[leaf]
        descriptor = _gpu_buffer_descriptor((17,), np.float32)
        for mode in ('none', 'reverse'):
            artifact = slang.compile_kernel(
                primitive.copy_values, (descriptor, descriptor), target=target,
                autodiff_mode=mode, gradient_parameters=(0, 1) if mode == 'reverse' else (),
                _scalar_slang=original)
            assert artifact.target == target and artifact.tasks
        tangent = slang.compile_kernel(primitive.add, (descriptor,) * 3, target=target,
                                       _scalar_slang=original, _scalar_slang_jvp=True)
        assert tangent.target == target and tangent.tasks


@pytest.mark.parametrize('source,line,code', [
    ('def curve(value: float) -> shader.vec2:\n    return shader.vec2(value)\n', 2, 'author.type'),
    ('def curve(value: shader.vec2) -> float:\n    return value.x\n', 2, 'author.type'),
])
def test_nn_shaderlib_reports_original_typed_signature(library, source, line, code):
    root, reference, entry, leaf = library
    entry.write_text('import infernux.shader as shader\n' + source, encoding='utf-8')
    with using_project_root(root), pytest.raises(ShaderSourceError) as failure:
        nn.SlangUnary(reference, entry_point='curve')
    diagnostic, = failure.value.diagnostics
    assert Path(diagnostic['path']) == entry
    assert (diagnostic['line'], diagnostic['code']) == (line, code)


def test_nn_shaderlib_rejects_stage_only_function_at_its_source(library):
    root, reference, entry, leaf = library
    leaf.write_text('import infernux.shader as shader\n'
                    'def bend(value: float) -> float:\n'
                    '    return shader.ddx(value)\n', encoding='utf-8')
    with using_project_root(root), pytest.raises(ShaderSourceError) as failure:
        nn.SlangUnary(reference, entry_point='curve')
    diagnostic, = failure.value.diagnostics
    assert Path(diagnostic['path']) == leaf
    assert (diagnostic['line'], diagnostic['code']) == (3, 'author.stage')


def test_nn_raw_slang_definition_keeps_existing_syntax():
    source = '[Differentiable] public float forward(float value) { return sin(value); }'
    assert nn.SlangUnary(source).options == ('forward', source)
