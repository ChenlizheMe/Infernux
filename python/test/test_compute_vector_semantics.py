"""Public Web cook must preserve accepted work-item expression semantics."""
import textwrap

import numpy as np
import pytest

import infernux as inx
from infernux._compiler.source_metadata import embed_compute_sources
from infernux.engine.game_builder import GameBuilder
from infernux._compiler.taichi import frontend
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor
from infernux.engine.project_context import using_project_root


CASES = [
    ('unconditional', 'out[i] += 1', [11, 21, 31, 41], True),
    ('masked_array', 'if i < 2:\n    out[i] = values[i] + 1', [2, 3, 30, 40], True),
    ('masked_constant', 'if i < 2:\n    out[i] = 7', [7, 7, 30, 40], True),
    ('masked_increment', 'if i < 2:\n    out[i] += 3', [13, 23, 30, 40], True),
    ('inactive_gather', 'if i < 2:\n    out[i] = values[indices[i]]', [1, 2, 30, 40], True),
    ('uniform_true', 'if enabled:\n    out[i] = values[i]\nelse:\n    out[i] = values[i] * 2', [1, 2, 3, 4], True),
    ('uniform_false', 'if enabled:\n    out[i] = values[i]\nelse:\n    out[i] = values[i] * 2', [2, 4, 6, 8], False),
    ('vector_not', 'if not (i < 2):\n    out[i] = values[i]', [10, 20, 3, 4], True),
    ('chained_comparison', 'if 0 <= i < 2:\n    out[i] = values[i]', [1, 2, 30, 40], True),
    ('ordered_while', 'j = 0\nwhile j < 2:\n    out[i] += 1\n    j += 1', [12, 22, 32, 42], True),
]

EXTRA = [
    ('nested_gather', 'if i < 2:\n    if values[indices[i]] > 0:\n        out[i] = 7', [7, 7, 30, 40], True),
    ('empty_mask', 'if i < 0:\n    out[i] = values[indices[i]]', [10, 20, 30, 40], True),
    ('merged_local', 'x = 0\nif i < 2:\n    x = values[indices[i]]\nelse:\n    x = 3\nout[i] = x * 2', [2, 4, 6, 6], True),
    ('lazy_choice', 'out[i] = values[indices[i]] if i < 2 else -1', [1, 2, -1, -1], True),
    ('lazy_and', 'if i < 2 and values[indices[i]] > 0:\n    out[i] = 9', [9, 9, 30, 40], True),
    ('lazy_or', 'if i >= 2 or values[indices[i]] == 1:\n    out[i] = 7', [7, 20, 7, 7], True),
    ('lazy_chain', 'if i < 2 <= values[indices[i]]:\n    out[i] = 7', [10, 7, 30, 40], True),
    ('local_mask_merge', 'x = 1\nif i < 2:\n    x += values[indices[i]]\nif i > 0:\n    out[i] = x', [10, 3, 1, 1], True),
    ('uniform_short_circuit', 'if enabled and i < 2:\n    out[i] = values[indices[i]]', [10, 20, 30, 40], False),
    ('uniform_choice', 'x = 1 if enabled else 2\nout[i] = x', [2, 2, 2, 2], False),
    ('masked_divide', 'if i > 0:\n    out[i] = 12 / i', [10, 12, 6, 4], True),
    ('loop_local', 'total = 0\nfor k in range(2):\n    if i < 2:\n        total += values[indices[i]]\nout[i] = total', [2, 4, 0, 0], True),
    ('and_value', 'out[i] = i < 2 and 7', [7, 7, 0, 0], True),
    ('or_value', 'out[i] = i < 2 or 7', [1, 1, 7, 7], True),
    ('masked_atomic_scalar', 'if i < 2:\n    inx.compute.atomic_add(out[0], 3)', [16, 20, 30, 40], True),
    ('masked_atomic_gather', 'if i < 2:\n    inx.compute.atomic_add(out[indices[i]], values[indices[i]])', [11, 22, 30, 40], True),
    ('empty_atomic', 'if i < 0:\n    inx.compute.atomic_add(out[indices[i]], values[indices[i]])', [10, 20, 30, 40], True),
]


def kernel_source(body):
    return ('import infernux as inx\n@inx.compute.kernel\n'
            'def solve(domain, out, values, indices, enabled):\n'
            '    i = inx.compute.index(domain)\n' + textwrap.indent(body, '    ') + '\n')


def execute(source, enabled):
    namespace = {'__name__': 'VectorProbe'}
    exec(compile(source, '<vector-probe>', 'exec'), namespace)
    buffers = [inx.buffer(shape=4, dtype=np.int32, device='gpu'),
               inx.buffer(shape=4, dtype=np.int32, device='gpu', data=[10, 20, 30, 40]),
               inx.buffer(shape=4, dtype=np.int32, device='gpu', data=[1, 2, 3, 4]),
               inx.buffer(shape=4, dtype=np.int32, device='gpu', data=[0, 1, 99, 99])]
    try:
        inx.compute.launch(namespace['solve'], (*buffers, enabled))
        return buffers[1].numpy(copy=True).tolist(), namespace['solve']._cpu_vectorized
    finally:
        for value in buffers:
            value.close()


@pytest.mark.parametrize('case,body,expected,enabled', CASES, ids=[case[0] for case in CASES])
def test_game_builder_preserves_work_item_results(tmp_path, monkeypatch, case, body, expected, enabled):
    monkeypatch.setenv('INFERNUX_WEB_RUNTIME', '1')
    source = kernel_source(body)
    reference, _ = execute(embed_compute_sources(source), enabled)
    assert reference == expected
    builder = GameBuilder(str(tmp_path), str(tmp_path / 'output'), game_name='VectorProbe')
    builder._runtime_platform = 'web'
    cooked = builder._cook_compute_source(source)
    actual, vectorized = execute(cooked, enabled)
    assert actual == reference
    assert vectorized == (case != 'ordered_while')


@pytest.mark.parametrize('case,body,expected,enabled', CASES + EXTRA, ids=[case[0] for case in CASES + EXTRA])
def test_web_batches_match_real_native_dispatch(tmp_path, engine, monkeypatch, case, body, expected, enabled):
    source = kernel_source(body)
    namespace = {'__name__': 'VectorNativeProbe'}
    exec(compile(embed_compute_sources(source), '<vector-native-probe>', 'exec'), namespace)
    descriptors = tuple(_gpu_buffer_descriptor((4,), np.int32) for _ in range(4)) + (enabled,)
    with using_project_root(tmp_path):
        artifact = frontend.compile_kernel(namespace['solve'].function, descriptors)
    host = engine._acquire_compute_host()
    monkeypatch.setattr(inx.compute, '_native_compute_host', lambda: host)
    buffers = [inx.buffer(shape=4, dtype=np.int32, device='gpu'),
               inx.buffer(shape=4, dtype=np.int32, device='gpu', data=[10, 20, 30, 40]),
               inx.buffer(shape=4, dtype=np.int32, device='gpu', data=[1, 2, 3, 4]),
               inx.buffer(shape=4, dtype=np.int32, device='gpu', data=[0, 1, 99, 99])]
    executable = inx.compute._KernelExecutable(artifact, host, (*buffers, enabled))
    try:
        executable.launch((*buffers, enabled))
        readback = buffers[1].get_data()
        try:
            native = readback.numpy().copy()
            np.testing.assert_array_equal(native, expected)
        finally:
            readback.close()
    finally:
        executable.close()
        for buffer in buffers:
            buffer.close()
        host._release_lease()
    monkeypatch.setenv('INFERNUX_WEB_RUNTIME', '1')
    builder = GameBuilder(str(tmp_path), str(tmp_path / 'output'), game_name='VectorNativeProbe')
    builder._runtime_platform = 'web'
    actual, vectorized = execute(builder._cook_compute_source(source), enabled)
    np.testing.assert_array_equal(actual, native)
    assert vectorized == (case != 'ordered_while')


@pytest.mark.parametrize('body,expected', [
    ('v = values[i]\nif i % 2 == 0:\n    v[1] += 2\nout[i] = v * scales[i]',
     [[0, 3, 2], [6, 8, 10], [18, 27, 24], [36, 40, 44], [60, 75, 70]]),
    ('out[i] = values[i] * 2 if i < 2 else values[i] + 1',
     [[0, 2, 4], [6, 8, 10], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
    ('out[i] = values[0] + scales[i]',
     [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]]),
])
def test_vector_lanes_and_scalar_broadcast_match_gpu(tmp_path, engine, monkeypatch, body, expected):
    source = ('import infernux as inx\n@inx.compute.kernel\ndef solve(domain, out, values, scales):\n'
              '    i = inx.compute.index(domain)\n' + textwrap.indent(body, '    ') + '\n')
    namespace = {'__name__': 'VectorLaneProbe'}
    exec(compile(embed_compute_sources(source), '<vector-lane-native>', 'exec'), namespace)
    descriptors = (_gpu_buffer_descriptor((5,), np.int32),
                   _gpu_buffer_descriptor((5,), inx.vector3), _gpu_buffer_descriptor((5,), inx.vector3),
                   _gpu_buffer_descriptor((5,), np.float32))
    with using_project_root(tmp_path):
        artifact = frontend.compile_kernel(namespace['solve'].function, descriptors)
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    host = engine._acquire_compute_host()
    monkeypatch.setattr(inx.compute, '_native_compute_host', lambda: host)
    for backend in ('gpu', 'web'):
        if backend == 'web':
            monkeypatch.setenv('INFERNUX_WEB_RUNTIME', '1')
        buffers = [inx.buffer(shape=5, dtype=np.int32, device='gpu'),
                   inx.buffer(shape=5, dtype=inx.vector3, device='gpu'),
                   inx.buffer(shape=5, dtype=inx.vector3, device='gpu', data=values),
                   inx.buffer(shape=5, dtype=np.float32, device='gpu', data=[1, 2, 3, 4, 5])]
        executable = None
        try:
            if backend == 'gpu':
                executable = inx.compute._KernelExecutable(artifact, host, tuple(buffers))
                executable.launch(tuple(buffers))
            else:
                builder = GameBuilder(str(tmp_path), str(tmp_path / 'output'), game_name='VectorLaneProbe')
                builder._runtime_platform = 'web'
                exec(compile(builder._cook_compute_source(source), '<vector-lane-web>', 'exec'), namespace)
                assert namespace['solve']._cpu_vectorized
                inx.compute.launch(namespace['solve'], tuple(buffers))
            for buffer, wanted in ((buffers[1], expected), (buffers[2], values)):
                snapshot = buffer.get_data()
                try:
                    np.testing.assert_allclose(snapshot.numpy(), wanted, rtol=1e-6)
                finally:
                    snapshot.close()
        finally:
            if executable is not None:
                executable.close()
            for buffer in buffers:
                buffer.close()
            if backend == 'gpu':
                host._release_lease()


def test_class_scope_and_author_symbols_do_not_capture_generated_helpers(tmp_path, monkeypatch):
    source = '''import infernux as inx
import infernux.compute as _inx_cpu_compute
_inx_cpu_lanes = 7
class Controller:
    @staticmethod
    @inx.compute.kernel
    def solve(domain, out, _inx_cpu_np):
        i = inx.compute.index(domain)
        if i < 2:
            out[i] = _inx_cpu_lanes + _inx_cpu_np
'''
    builder = GameBuilder(str(tmp_path), str(tmp_path / 'output'), game_name='HygienicProbe')
    builder._runtime_platform = 'web'
    namespace = {'__name__': 'HygienicProbe'}
    exec(compile(builder._cook_compute_source(source), '<hygienic-web>', 'exec'), namespace)
    monkeypatch.setenv('INFERNUX_WEB_RUNTIME', '1')
    domain = inx.buffer(shape=4, dtype=np.int32, device='gpu')
    out = inx.buffer(shape=4, dtype=np.int32, device='gpu')
    try:
        inx.compute.launch(namespace['Controller'].solve, (domain, out, 3))
        np.testing.assert_array_equal(out.numpy(), [10, 10, 0, 0])
    finally:
        domain.close()
        out.close()


def test_source_less_web_reads_only_the_declared_domain(tmp_path, monkeypatch):
    import importlib.machinery
    import importlib.util
    import py_compile

    source = ('import infernux as inx\n@inx.compute.kernel\ndef solve(out, domain):\n'
              '    i = inx.compute.index(domain)\n    if i >= 0:\n        out[i] += 3\n')
    builder = GameBuilder(str(tmp_path), str(tmp_path / 'output'), game_name='BytecodeProbe')
    builder._runtime_platform = 'web'
    path = tmp_path / 'Work.py'
    path.write_text(builder._cook_compute_source(source), encoding='utf-8')
    bytecode = path.with_suffix('.pyc')
    py_compile.compile(str(path), cfile=str(bytecode), doraise=True)
    path.unlink()
    loader = importlib.machinery.SourcelessFileLoader('BytecodeProbe', str(bytecode))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader('BytecodeProbe', loader))
    loader.exec_module(module)
    monkeypatch.setenv('INFERNUX_WEB_RUNTIME', '1')
    domain = inx.buffer(shape=2, dtype=np.int32, device='gpu')
    out = inx.buffer(shape=5, dtype=np.int32, device='gpu')
    try:
        monkeypatch.setattr(frontend, '_function_source', lambda *_: pytest.fail('cooked code must not inspect source'))
        assert module.solve._cpu_vectorized
        inx.compute.launch(module.solve, (out, domain))
        np.testing.assert_array_equal(out.numpy(), [3, 3, 0, 0, 0])
    finally:
        domain.close()
        out.close()


@pytest.mark.parametrize('body,expected', [
    ('out[indices[i]] = values[i]', [2, 4, 30, 40]),
    ('if i > 0:\n    out[i] = out[i - 1] + 1', [10, 11, 12, 13]),
])
def test_visible_cross_item_writes_keep_ordered_execution(tmp_path, monkeypatch, body, expected):
    source = kernel_source(body)
    builder = GameBuilder(str(tmp_path), str(tmp_path / 'output'), game_name='OrderedProbe')
    builder._runtime_platform = 'web'
    namespace = {'__name__': 'OrderedProbe'}
    exec(compile(builder._cook_compute_source(source), '<ordered-web>', 'exec'), namespace)
    monkeypatch.setenv('INFERNUX_WEB_RUNTIME', '1')
    buffers = [inx.buffer(shape=4, dtype=np.int32, device='gpu'),
               inx.buffer(shape=4, dtype=np.int32, device='gpu', data=[10, 20, 30, 40]),
               inx.buffer(shape=4, dtype=np.int32, device='gpu', data=[1, 2, 3, 4]),
               inx.buffer(shape=4, dtype=np.int32, device='gpu', data=[0, 0, 1, 1])]
    try:
        assert not namespace['solve']._cpu_vectorized
        inx.compute.launch(namespace['solve'], (*buffers, True))
        np.testing.assert_array_equal(buffers[1].numpy(), expected)
    finally:
        for value in buffers:
            value.close()
