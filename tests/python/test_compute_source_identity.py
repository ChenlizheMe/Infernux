"""A cache hit must describe the current kernel and its actual compile-time inputs."""
import ast
import linecache
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from infernux import compute
from infernux._compiler import slang as frontend
from infernux._compiler.compile_inputs import snapshot_compilation_values
from infernux.lib import _Infernux
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor
from infernux.engine.project_context import using_project_root


def load_author(path, source, name='Scripts.Work', **bindings):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
    linecache.clearcache()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(bindings)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def load_kernel(path, source, **bindings):
    return load_author(path, source, **bindings).solve.function


@pytest.mark.parametrize('kind', ['scalar', 'helper', 'numpy', 'sequence', 'attribute'])
def test_warm_and_cold_compilation_agree_after_constant_edit(tmp_path, engine, monkeypatch, kind):
    descriptor = _gpu_buffer_descriptor((4,), np.int32)
    path = tmp_path / 'warm/Assets/Scripts/Work.py'
    artifacts = []
    latest = None
    for value in (1, 2):
        setup, expression = f'GAIN = {value}\n', 'GAIN'
        if kind == 'numpy':
            setup = f'import numpy as np\nGAIN = np.int32({value})\n'
        elif kind == 'sequence':
            setup, expression = f'GAINS = [{value}, 99]\n', 'GAINS[0]'
        elif kind == 'attribute':
            setup, expression = f'from types import SimpleNamespace\nconfig = SimpleNamespace(gain={value})\n', 'config.gain'
        elif kind == 'helper':
            setup += '@inx.compute.function\ndef helper(value):\n    return value + GAIN\n'
            expression = 'helper(0)'
        source = ('import infernux as inx\n' + setup +
                  '@inx.compute.kernel\ndef solve(values):\n'
                  '    i = inx.compute.index(values)\n' + f'    values[i] += {expression}\n')
        latest = load_kernel(path, source)
        with using_project_root(tmp_path / 'warm'):
            artifacts.append(frontend.compile_kernel(latest, (descriptor,)))
    with using_project_root(tmp_path / 'cold'):
        cold = frontend.compile_kernel(latest, (descriptor,))
    assert artifacts[0].tasks != cold.tasks
    assert artifacts[-1].tasks == cold.tasks
    assert len(list((tmp_path / 'warm/Library/Artifacts/Compute').glob('*.inxgpu'))) == 2

    # Public launch must select the latest warmed artifact without recompiling.
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    values = compute.buffer(shape=4, dtype=np.int32, device='gpu', data=[10, 20, 30, 40])
    declaration = compute.Kernel(latest)
    try:
        with using_project_root(tmp_path / 'warm'), monkeypatch.context() as guard:
            guard.setattr(_Infernux, '_compile_shader_program', lambda *args, **kwargs: pytest.fail('public launch must use the warmed current artifact'))
            compute.launch(declaration, (values,))
        used = next(iter(declaration._executables.values()))
        assert used.artifact.tasks == artifacts[-1].tasks
        assert len(list((tmp_path / 'warm/Library/Artifacts/Compute').glob('*.inxgpu'))) == 2
        snapshot = values.get_data()
        try:
            np.testing.assert_array_equal(snapshot.numpy(), [12, 22, 32, 42])
        finally:
            snapshot.close()
    finally:
        declaration._release_engine_resources()
        values.close()
        host._release_lease()


def test_unused_and_shadowed_globals_do_not_invalidate_compilation(tmp_path):
    descriptor = _gpu_buffer_descriptor((4,), np.int32)
    path = tmp_path / 'Assets/Scripts/Work.py'
    compiled = []
    for value in (1, 2):
        source = (f'import infernux as inx\nGAIN = {value}\nUNUSED = {value}\n'
                  '@inx.compute.kernel\ndef solve(values):\n'
                  '    i = inx.compute.index(values)\n    GAIN = 5\n    values[i] += GAIN\n')
        function = load_kernel(path, source)
        with using_project_root(tmp_path):
            compiled.append(frontend.compile_kernel(function, (descriptor,)))
    assert compiled[0].tasks == compiled[1].tasks
    assert len(list((tmp_path / 'Library/Artifacts/Compute').glob('*.inxgpu'))) == 1


@pytest.mark.parametrize('accepted_module_code', [False, True])
def test_first_preparation_uses_retained_source_after_disk_edit(tmp_path, accepted_module_code):
    from infernux.engine.script_source import prepare_module_code

    path = tmp_path / 'Assets' / 'Scripts' / 'Work.py'
    source = ('import infernux as inx\n'
              '@inx.compute.function\ndef gain(value):\n    return value + 2\n'
              '@inx.compute.kernel\ndef solve(values):\n'
              '    i = inx.compute.index(values)\n    values[i] = gain(values[i]) + 3\n')
    if accepted_module_code:
        retained = ModuleType('Scripts.Work')
        accepted = prepare_module_code(retained.__name__, str(path), source=source)
    else:
        retained = load_author(path, source)
    current = load_author(path, source.replace('+ 2', '+ 5').replace('+ 3', '+ 7'))
    if accepted_module_code:
        accepted.execute(retained.__dict__)

    params = (_gpu_buffer_descriptor((4,), np.int32),)
    with using_project_root(tmp_path):
        first = frontend.prepare_kernel(retained.solve.function, params)
        second = frontend.prepare_kernel(current.solve.function, params)
    assert '(value + 2)' in first.generated_source and '+ 3)' in first.generated_source
    assert '(value + 5)' in second.generated_source and '+ 7)' in second.generated_source
    assert first.artifact_key != second.artifact_key


def test_gpu_continuation_preparation_preserves_frozen_source(tmp_path, engine, monkeypatch):
    path = tmp_path / 'Assets' / 'Scripts' / 'Work.py'
    source = ('import infernux as inx\n@inx.compute.kernel\ndef solve(values, output):\n'
              '    i = inx.compute.index(values)\n    output[i] = values[i] * 2.0\n')
    retained = load_author(path, source).solve
    path.write_text(source.replace('2.0', '5.0'), encoding='utf-8')
    linecache.clearcache()

    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    values = compute.buffer(shape=4, dtype=np.float32, device='gpu', data=[1, 2, 3, 4])
    output = compute.buffer(shape=4, dtype=np.float32, device='gpu')
    continuation = None
    try:
        with using_project_root(tmp_path):
            continuation = compute.prepare_continuation(retained, (values, output))
            assert continuation.inputs == ('values',) and continuation.outputs == ('output',)
            compute.launch(continuation.program, (values, output))
        snapshot = output.get_data()
        try:
            np.testing.assert_array_equal(snapshot.numpy(), [2, 4, 6, 8])
        finally:
            snapshot.close()
    finally:
        if continuation is not None:
            continuation.program._release_engine_resources()
        output.close()
        values.close()
        host._release_lease()


def test_identical_code_retains_frozen_helper_module_identity(tmp_path):
    source = ('import infernux as inx\n'
              '@inx.compute.function\ndef apply(value):\n    return value + 2\n'
              '@inx.compute.function\ndef unused(value):\n    return value + 7\n')
    kernel_source = ('import infernux as inx\n@inx.compute.kernel\ndef solve(values):\n'
                     '    i = inx.compute.index(values)\n    values[i] = provider.apply(values[i])\n')
    params = (_gpu_buffer_descriptor((4,), np.int32),)
    artifacts = []
    with using_project_root(tmp_path):
        for name in ('Scripts.MathA', 'Scripts.MathB'):
            provider = load_author(tmp_path / 'Math.py', source, name)
            kernel = load_kernel(tmp_path / 'Work.py', kernel_source, provider=provider)
            artifact = frontend.compile_kernel(kernel, params)
            artifacts.append(artifact)
            entry, helper = artifact.diagnostic_locations
            assert entry['role'] == 'entry' and entry['function'] == 'Scripts.Work.solve'
            assert helper['role'] == 'helper' and helper['function'] == name + '.apply'
            assert (helper['line'], helper['column']) == (3, 1)
            assert frontend.compile_kernel(kernel, params).diagnostic_locations == artifact.diagnostic_locations
    assert artifacts[0].identity != artifacts[1].identity
    assert artifacts[0].tasks == artifacts[1].tasks


@pytest.mark.parametrize(('expression', 'phase', 'column'), [
    ('host_only(value)', 'author', 12),
    ('inx.compute.vec2(value, value, value, value, value)', 'compile', 5),
])
def test_public_prepare_preserves_transitive_helper_diagnostic_ranges(tmp_path, engine, monkeypatch,
                                                              expression, phase, column):
    leaf_path = tmp_path / 'Leaf.py'
    leaf = load_author(leaf_path,
                       'import infernux as inx\n@inx.compute.function\ndef apply(value):\n'
                       f'    return {expression}\n', 'Scripts.Leaf')
    provider = load_author(tmp_path / 'Shared.py',
                           'import infernux as inx\n@inx.compute.function\ndef apply(value):\n'
                           '    return leaf.apply(value)\n', 'Scripts.Shared', leaf=leaf)
    kernel = load_author(tmp_path / 'Work.py',
                         'import infernux as inx\n@inx.compute.kernel\ndef solve(values):\n'
                         '    i = inx.compute.index(values)\n    values[i] = provider.apply(values[i])\n',
                         provider=provider).solve
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    values = compute.buffer(shape=4, dtype=np.int32, device='gpu')
    try:
        with using_project_root(tmp_path), pytest.raises(compute.KernelCompilationError) as error:
            compute.prepare(kernel, (values,))
        assert any(Path(item['path']) == leaf_path and item['phase'] == phase and item['line'] == 4
                   and item['column'] == column and item['end_column'] > column
                   for item in error.value.diagnostics), error.value.diagnostics
    finally:
        values.close()
        host._release_lease()


def test_compile_values_snapshot_mutable_inputs_once():
    definition = ast.parse('def kernel(value):\n    return value + DATA[0] + SETTINGS.offset\n').body[0]
    from types import SimpleNamespace
    data = np.array([3, 4], dtype=np.int32)
    settings = SimpleNamespace(offset=5.)
    globals_map = {'DATA': data, 'SETTINGS': settings, 'UNUSED': object()}
    identities = snapshot_compilation_values([definition], globals_map)
    data[0] = 99
    settings.offset = 100.
    assert identities['DATA'][-1] == np.array([3, 4], dtype=np.int32).tobytes().hex()
    assert identities['SETTINGS.offset'] == ['float', (5.).hex()]
    tree = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    exec(compile(tree, '<frozen-compile-inputs>', 'exec'), globals_map)
    assert globals_map['kernel'](1) == 9
