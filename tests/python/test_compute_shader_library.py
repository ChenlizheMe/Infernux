"""Exercise the same authored ShaderLib through graphics and numerical Slang."""
import json
import linecache
import py_compile
from importlib.machinery import SourcelessFileLoader
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from infernux import compute
from infernux.application import Application
from infernux._compiler import slang as frontend
from infernux._compiler.shader import read_shader
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor, stage_compute_artifacts
from infernux.engine.game_builder import GameBuilder
from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
from infernux.engine.project_context import (
    set_runtime_asset_query, set_runtime_asset_resolver, using_project_root,
)
from infernux.lib import _Infernux


KERNEL = '''import infernux as inx
forces = inx.compute.library(".forces")

@inx.compute.function
def adjust(value):
    return forces.damp(value)

@inx.compute.kernel
def solve(values):
    i = inx.compute.index(values)
    values[i] = adjust(values[i])
'''

LIBRARY = '''import infernux.shader as shader
from . import physical as p, display as d

def damp(value: float) -> float:
    return p.scale(value) + d.scale(value)
'''


@pytest.fixture
def project(tmp_path):
    root = tmp_path / 'project'
    scripts = root / 'Assets' / 'Scripts'
    scripts.mkdir(parents=True)
    (scripts / 'Work.py').write_text(KERNEL, encoding='utf-8')
    (scripts / 'forces.shaderlib.py').write_text(LIBRARY, encoding='utf-8')
    (scripts / 'physical.shaderlib.py').write_text(
        'import infernux.shader as shader\ndef scale(value: float) -> float:\n    return value * 2.0\n',
        encoding='utf-8',
    )
    (scripts / 'display.shaderlib.py').write_text(
        'import infernux.shader as shader\ndef scale(value: float) -> float:\n    return value + 1.0\n',
        encoding='utf-8',
    )
    return root, scripts


def load_kernel(path):
    module = ModuleType('Scripts.Work')
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), vars(module))
    return module.solve


def test_transitive_shaderlib_links_graphics_and_compute_targets(project):
    root, scripts = project
    preview = scripts / 'Preview.frag.py'
    preview.write_text('''import infernux.shader as shader
from .forces import damp

class Preview(shader.fullscreen):
    def main(self) -> shader.vec4:
        return shader.vec4(damp(0.25))
''', encoding='utf-8')
    params = (_gpu_buffer_descriptor((4,), np.float32),)
    declaration = load_kernel(scripts / 'Work.py')
    with using_project_root(root):
        compiled = read_shader(preview, 'test_shared_shaderlib').compile()
        assert {program['target'] for program in compiled['targets']} == {'spirv', 'wgsl'}
        for target in ('spirv', 'wgsl'):
            artifact = frontend.compile_kernel(declaration.function, params, target=target)
            assert artifact.target == target and artifact.tasks
            assert [Path(row['path']).name for row in artifact.diagnostic_locations if row['role'] == 'library'] == [
                'physical.shaderlib.py', 'display.shaderlib.py', 'forces.shaderlib.py',
            ]
            assert any('ShaderLib calls require a proven operation bound' in message
                       for message in artifact.continuation.diagnostics)
            if target == 'spirv':
                primal = artifact
                assert artifact.tasks[0].startswith(b'\x03\x02\x23\x07')
            else:
                assert b'@compute' in artifact.tasks[0]
        leaf = scripts / 'physical.shaderlib.py'
        leaf.write_text(leaf.read_text(encoding='utf-8').replace('2.0', '4.0'), encoding='utf-8')
        changed = frontend.compile_kernel(declaration.function, params)
    assert changed.identity != primal.identity and changed.tasks != primal.tasks


@pytest.mark.parametrize(('leaf', 'code', 'path', 'line'), [
    ('return shader.ddx(value)', 'author.stage', 'physical.shaderlib.py', 3),
    ('return shader.frame_time()', 'author.stage', 'physical.shaderlib.py', 3),
    ('return value', 'author.reference', 'Work.py', 6),
])
def test_shaderlib_errors_report_the_original_library_or_call(project, leaf, code, path, line):
    root, scripts = project
    work = scripts / 'Work.py'
    if code == 'author.reference':
        work.write_text(KERNEL.replace('forces.damp', 'forces.missing'), encoding='utf-8')
    (scripts / 'physical.shaderlib.py').write_text(
        f'import infernux.shader as shader\ndef scale(value: float) -> float:\n    {leaf}\n',
        encoding='utf-8',
    )
    with using_project_root(root), pytest.raises(SyntaxError) as failure:
        frontend.compile_kernel(load_kernel(work).function, (_gpu_buffer_descriptor((4,), np.float32),))
    diagnostic, = failure.value.diagnostics
    assert diagnostic['code'] == code and diagnostic['line'] == line
    assert Path(diagnostic['path']) == scripts / path and diagnostic['end_column'] > diagnostic['column']


def test_linked_shaderlib_native_diagnostic_maps_the_library_span(project):
    root, scripts = project
    leaf = scripts / 'physical.shaderlib.py'
    leaf.write_text('''import infernux.shader as shader
def scale(value: float) -> float:
    return shader.vec2(value, value, value, value, value)
''', encoding='utf-8')
    with using_project_root(root), pytest.raises(_Infernux.ShaderCompilationError) as failure:
        frontend.compile_kernel(load_kernel(scripts / 'Work.py').function, (_gpu_buffer_descriptor((4,), np.float32),))
    assert any(Path(row['path']) == leaf and row['phase'] == 'compile' and row['severity'] == 'error'
               and (row['line'], row['column']) == (3, 5) and row['end_column'] > 5
               for row in failure.value.diagnostics), failure.value.diagnostics


def test_reverse_kernel_differentiates_transitive_shaderlib_calls(project, engine, monkeypatch):
    root, scripts = project
    leaf = scripts / 'physical.shaderlib.py'
    source = leaf.read_text(encoding='utf-8').replace('value * 2.0', 'value * value')
    leaf.write_text(source + '\ndef cell(value: float) -> float:\n    return shader.floor(value)\n', encoding='utf-8')
    work = scripts / 'Work.py'
    work.write_text(KERNEL.replace('solve(values):', 'solve(values, output):')
                    .replace('values[i] = adjust', 'output[i] = adjust'), encoding='utf-8')
    declaration = load_kernel(work)
    values = np.linspace(-2.0, 2.0, 67, dtype=np.float32)
    seeds = np.resize(np.array([0.0, 0.5, -1.25], dtype=np.float32), values.size)
    initial_gradients = np.full_like(values, 0.25)
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    inputs = compute.buffer(shape=values.size, dtype=np.float32, device='gpu', data=values)
    outputs = compute.buffer(shape=values.size, dtype=np.float32, device='gpu')
    gradients = compute.buffer(shape=values.size, dtype=np.float32, device='gpu', data=initial_gradients)
    output_seeds = compute.buffer(shape=values.size, dtype=np.float32, device='gpu', data=seeds)
    try:
        with using_project_root(root):
            params = inputs, outputs
            wgsl = frontend.compile_kernel(declaration.function, params, target='wgsl',
                                           autodiff_mode='reverse', gradient_parameters=(0, 1))
            assert b'@compute' in wgsl.tasks[0] and wgsl.gradient_parameters == (0, 1)
            compute.prepare(declaration, params, autodiff_mode='reverse', gradient_parameters=(0, 1))
            compute.launch(declaration, params)
            for _ in range(2):
                compute.launch(declaration, params, autodiff_mode='reverse', gradients={0: gradients, 1: output_seeds})
        for buffer, expected in (
            (outputs, values * values + values + 1.0),
            (gradients, initial_gradients + 2 * seeds * (2 * values + 1.0)),
            (output_seeds, seeds),
        ):
            snapshot = buffer.get_data()
            try:
                np.testing.assert_allclose(snapshot.numpy(), expected, rtol=2e-6, atol=2e-6)
            finally:
                snapshot.close()
    finally:
        declaration._release_engine_resources()
        for buffer in (inputs, outputs, gradients, output_seeds):
            buffer.close()
        host._release_lease()


def test_forward_kernel_writes_primal_and_tangents_through_shaderlib(project, tmp_path, engine, monkeypatch):
    root, scripts = project
    (scripts / 'packing.shaderlib.py').write_text('''import infernux.shader as shader
from .forces import damp

def pack(value: shader.vec2) -> shader.vec4:
    x = value.x
    y = value.y
    return shader.vec4(damp(x * x), x * y, shader.sin(y), x + y)
''', encoding='utf-8')
    work = scripts / 'Work.py'
    work.write_text('''import infernux as inx
packing = inx.compute.library(".packing")

@inx.compute.function
def terms(x, y):
    return packing.pack(inx.compute.vec2(x, y))

@inx.compute.kernel
def solve(x, y, shift, output, auxiliary):
    i = inx.compute.index(x)
    packed = terms(x[i], y[i])
    output[i] = packed.x + packed.z + shift[i]
    auxiliary[i] = packed.y - packed.w
''', encoding='utf-8')
    declaration = load_kernel(work)
    x = np.linspace(-2.0, 2.0, 67, dtype=np.float32)
    y = np.linspace(0.5, 1.5, x.size, dtype=np.float32)
    dx = np.resize(np.array([0.0, 0.5, -1.25], dtype=np.float32), x.size)
    dy = np.resize(np.array([1.0, -0.25, 0.75], dtype=np.float32), x.size)
    shift = np.full_like(x, 0.125)
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    buffers = tuple(compute.buffer(shape=x.size, dtype=np.float32, device='gpu', data=data)
                    for data in (x, y, shift, np.zeros_like(x), np.zeros_like(x), dx, dy, np.full_like(x, 9.0)))
    params = buffers[:5]
    seeds = [value.view(readonly=True) for value in buffers[5:7]]
    tangent_output = buffers[7]
    derivative_buffers = {0: seeds[0], 1: seeds[1], 3: tangent_output}

    def read(buffer):
        snapshot = buffer.get_data()
        try:
            return np.array(snapshot.numpy(), copy=True)
        finally:
            snapshot.close()

    try:
        with using_project_root(root):
            wgsl = frontend.compile_kernel(declaration.function, params, target='wgsl',
                                           autodiff_mode='forward', gradient_parameters=(0, 1, 3))
            assert b'@compute' in wgsl.tasks[0] and wgsl.autodiff_mode == 'forward'
            access = {binding['argument_indices'][0]: binding['access']
                      for binding in wgsl.task_metadata[0]['buffer_bindings'] if binding.get('is_grad')}
            assert access == {0: 1, 1: 1, 3: 2}
            compute.prepare(declaration, params)
            compute.prepare(declaration, params, autodiff_mode='forward', gradient_parameters=(0, 1, 3))
            compute.launch(declaration, params, autodiff_mode='forward', gradients=derivative_buffers)
            np.testing.assert_allclose(read(params[3]), 3 * x * x + 1.0 + np.sin(y) + shift, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(read(params[4]), x * y - x - y, rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(read(tangent_output), 6 * x * dx + np.cos(y) * dy, rtol=2e-6, atol=2e-6)
            np.testing.assert_array_equal(read(seeds[0]), dx)
            np.testing.assert_array_equal(read(seeds[1]), dy)
            compute.launch(declaration, params, autodiff_mode='forward', gradients={3: tangent_output})
            np.testing.assert_array_equal(read(tangent_output), np.zeros_like(x))
            with pytest.raises(ValueError, match='disjoint primal and derivative buffer ranges'):
                compute.launch(declaration, params, autodiff_mode='forward',
                               gradients={**derivative_buffers, 0: params[0]})

        data_root = tmp_path / 'build' / 'Data'
        stage_compute_artifacts(root, (work,), data_root)
        manifest = json.loads((data_root / 'Library/Artifacts/Compute/AotManifest.json').read_text(encoding='utf-8'))
        roles = [json.loads(row['specialization']) for row in manifest['artifacts'] if row['function'] == 'Scripts.Work.solve']
        assert {role['autodiff_mode'] for role in roles} == {'none', 'forward'}
        assert {role['target'] for role in roles if role['autodiff_mode'] == 'forward'} == {'spirv', 'wgsl'}
    finally:
        declaration._release_engine_resources()
        for buffer in (*seeds, *buffers):
            buffer.close()
        host._release_lease()


@pytest.mark.parametrize(('body', 'mask', 'message'), [
    ('if values[i] > 0.0:\n        output[i] = values[i]', (0, 1), 'requires every row to write output'),
    ('output[i] = values[i + 1]', (0, 1), 'reads the current row'),
    ('output[i] = values[i]', (0,), 'requires an output tangent buffer'),
])
def test_forward_kernel_requires_complete_row_outputs_and_tangents(project, body, mask, message):
    root, scripts = project
    work = scripts / 'Work.py'
    work.write_text('import infernux as inx\n@inx.compute.kernel\ndef solve(values, output):\n'
                    f'    i = inx.compute.index(values)\n    {body}\n', encoding='utf-8')
    buffer = _gpu_buffer_descriptor((4,), np.float32)
    with using_project_root(root), pytest.raises(SyntaxError, match=message):
        frontend.compile_kernel(load_kernel(work).function, (buffer, buffer),
                                autodiff_mode='forward', gradient_parameters=mask)


def test_cooked_project_shaderlib_resolves_through_player_assets(project, tmp_path, engine, monkeypatch):
    root, scripts = project
    data = tmp_path / 'Player' / 'Data'
    builder = GameBuilder(str(root), str(tmp_path / 'Player'), game_name='ShaderLibTest')
    builder.include_jit_runtime = False
    authored_work = scripts / 'Work.py'
    runtime_work = data / authored_work.relative_to(root)
    runtime_work.parent.mkdir(parents=True)
    runtime_work.write_text(builder._cook_compute_source(KERNEL), encoding='utf-8')
    bytecode = py_compile.compile(str(runtime_work), cfile=str(runtime_work.with_suffix('.pyc')),
                                  dfile=authored_work.relative_to(root).as_posix(), optimize=2, doraise=True)
    runtime_work.unlink()
    authored_work.unlink()
    artifacts, records = [], []
    for index, path in enumerate(sorted(scripts.glob('*.shaderlib.py'))):
        source_path = path.relative_to(root).as_posix()
        runtime_path = 'Library/Artifacts/Blob/' + source_path
        cooked = data / runtime_path
        cooked.parent.mkdir(parents=True, exist_ok=True)
        payload = path.read_bytes()
        cooked.write_bytes(payload)
        path.unlink()
        guid, artifact_id = f'{index:032x}', f'ShaderLib-{index}'
        artifacts.append({'runtime_artifact_id': artifact_id, 'runtime_path': runtime_path,
                          'asset_guid': guid, 'package': 'Content.inxpkg', 'content_bytes': len(payload)})
        records.append({'guid': guid, 'runtime_path': source_path, 'primary_runtime_artifact_id': artifact_id,
                        'runtime_artifact_ids': [artifact_id]})
    catalog = PlayerRuntimeAssetCatalog.from_documents(str(data), {'artifacts': artifacts}, {'entries': records})
    monkeypatch.setattr(Application, 'is_player', staticmethod(lambda: True))
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    values = compute.buffer(shape=4, dtype=np.float32, device='gpu', data=[0, 1, 2, 3])
    module = ModuleType('Scripts.Work')
    linecache.clearcache()
    try:
        with using_project_root(data):
            set_runtime_asset_query(catalog.query_asset_guids)
            set_runtime_asset_resolver(catalog.resolve_guid)
            SourcelessFileLoader(module.__name__, bytecode).exec_module(module)
            compute.launch(module.solve, (values,))
        result = values.get_data()
        try:
            np.testing.assert_array_equal(result.numpy(), [1, 4, 7, 10])
        finally:
            result.close()
    finally:
        if hasattr(module, 'solve'):
            module.solve._release_engine_resources()
        values.close()
        host._release_lease()
