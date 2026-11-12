"""A cache hit must describe the current kernel and its actual compile-time inputs."""
import ast
import linecache

import numpy as np
import pytest

from infernux import compute
from infernux._compiler import slang as frontend
from infernux._compiler.compile_inputs import snapshot_compilation_values
from infernux.lib import _Infernux
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor
from infernux.engine.project_context import using_project_root


def load_kernel(path, source, module='Scripts.Work'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf-8')
    linecache.clearcache()
    namespace = {'__name__': module, '__file__': str(path)}
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace['solve'].function


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
