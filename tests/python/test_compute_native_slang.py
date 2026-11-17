"""Project Slang entries use the public compute runtime and the cooked asset ABI."""

from contextlib import ExitStack
from importlib.machinery import SourcelessFileLoader
import json
from pathlib import Path
import py_compile
from types import ModuleType

import numpy as np
import pytest

from infernux import compute
from infernux.application import Application
from infernux._compiler.gpu_artifact import _read_artifact
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor, stage_compute_artifacts
from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
from infernux.engine.project_context import set_runtime_asset_query, set_runtime_asset_resolver, using_project_root
from infernux.lib import _Infernux


REDUCTION = '''import contribution;

[vk::binding(0, 0)] StructuredBuffer<float> values;
[vk::binding(1, 0)] RWStructuredBuffer<float> sums;
groupshared float partials[4];

[shader("compute")]
[numthreads(4, 1, 1)]
void reduceRows(uint3 thread : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint count, stride;
    values.GetDimensions(count, stride);
    float value = 0.0;
    if (thread.x < count)
        value = forSum(values[thread.x]);
    partials[lane] = value;
    GroupMemoryBarrierWithGroupSync();

    for (uint step = 2; step > 0; step >>= 1) {
        if (lane < step)
            partials[lane] += partials[lane + step];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0) {
        sums.GetDimensions(count, stride);
        if (group.x < count)
            sums[group.x] = partials[0];
    }
}
'''

LIBRARIES = {'contribution': 'Assets/Kernels/contribution.slang', 'scale_value': 'Assets/Kernels/scale_value.slang'}
SOURCE_PATH = 'Assets/Kernels/reduce.slang'
RUNTIME_ENTRY = REDUCTION + '''
[shader("compute")]
[numthreads(4, 1, 1)]
void scaleRows(uint3 thread : SV_DispatchThreadID)
{
    uint count, stride;
    values.GetDimensions(count, stride);
    if (thread.x < count)
        sums[thread.x] = forSum(values[thread.x]);
}
'''
WORK_SOURCE = f'''from infernux import compute
from infernux.compute import kernel as authored
from pathlib import Path

class Run:
    reduce = authored(Path({SOURCE_PATH!r}), entry_point="reduceRows", domain="values", libraries={LIBRARIES!r})

def prepare_work(values, sums):
    return compute.prepare_work(((Run.reduce, (values, sums)),), private_buffers=(sums,),
                                inputs={{"seed": sums}}, outputs={{"result": sums}})
'''
TILED_REDUCTION = (
    'import inx_specialization;\nextern static const int tile;\n'
    + REDUCTION.replace('partials[4];', 'partials[tile];')
    .replace('numthreads(4,', 'numthreads(tile,')
    .replace('uint step = 2;', 'uint step = uint(tile) >> 1;')
)
TILED_WORK_SOURCE = f'''from infernux import compute

class Run:
    narrow = compute.kernel({SOURCE_PATH!r}, entry_point="reduceRows", domain="values", libraries={LIBRARIES!r},
                            constants={{"tile": 4}})
    wide = compute.kernel({SOURCE_PATH!r}, entry_point="reduceRows", domain="values", libraries={LIBRARIES!r},
                          constants={{"tile": 8}})

def prepare_work(values, narrow, wide):
    return compute.prepare_work(((Run.narrow, (values, narrow)), (Run.wide, (values, wide))),
                                private_buffers=(narrow, wide), inputs={{"narrow_seed": narrow, "wide_seed": wide}},
                                outputs={{"narrow": narrow, "wide": wide}})
'''
CALIBRATION_PATH = 'Assets/Kernels/calibration.slang'
CALIBRATION = '''import calibration_math;

[vk::binding(0, 0)] StructuredBuffer<float> realValues;
[vk::binding(1, 0)] RWStructuredBuffer<float> realResult;
[vk::binding(2, 0)] StructuredBuffer<int> integerValues;
[vk::binding(3, 0)] RWStructuredBuffer<int> integerResult;

[shader("compute")]
[numthreads(4, 1, 1)]
void calibrateFloat(uint3 thread : SV_DispatchThreadID)
{
    uint count, stride;
    realValues.GetDimensions(count, stride);
    if (thread.x < count)
        realResult[thread.x] = calibrate<float>(realValues[thread.x], 0.5, 1.25);
}

[shader("compute")]
[numthreads(4, 1, 1)]
void calibrateInt(uint3 thread : SV_DispatchThreadID)
{
    uint count, stride;
    integerValues.GetDimensions(count, stride);
    if (thread.x < count)
        integerResult[thread.x] = calibrate<int>(integerValues[thread.x], 3, 2);
}
'''
CALIBRATION_MODULES = (
    ('calibration_math', 'Assets/Kernels/calibration_math.slang', '''import scalar_math;
public T calibrate<T : IArithmetic>(T value, T scale, T bias)
{
    return multiplyAndAdd<T>(value, scale, bias);
}
'''),
    ('scalar_math', 'Assets/Kernels/scalar_math.slang', '''public T multiplyAndAdd<T : IArithmetic>(T value, T scale, T bias)
{
    return value * scale + bias;
}
'''),
)
CALIBRATION_LIBRARIES = {name: path for name, path, _ in CALIBRATION_MODULES}
CALIBRATION_WORK_SOURCE = f'''from infernux import compute

class Calibration:
    real = compute.kernel({CALIBRATION_PATH!r}, entry_point="calibrateFloat", domain="realValues",
                          libraries={CALIBRATION_LIBRARIES!r})
    integer = compute.kernel({CALIBRATION_PATH!r}, entry_point="calibrateInt", domain="integerValues",
                             libraries={CALIBRATION_LIBRARIES!r})

def prepare_work(real, real_result, integer, integer_result):
    return compute.prepare_work(((Calibration.real, (real, real_result)),
                                 (Calibration.integer, (integer, integer_result))),
                                private_buffers=(real_result, integer_result),
                                inputs={{"real_seed": real_result, "integer_seed": integer_result}},
                                outputs={{"real": real_result, "integer": integer_result}})
'''


@compute.kernel
def python_reduce(values, sums):
    row = compute.index(sums)
    first = 4 * row
    sums[row] = 2.0 * (values[first] + values[first + 1] + values[first + 2] + values[first + 3])


@compute.kernel
def initialize_sums(sums):
    row = compute.index(sums)
    sums[row] = 0.0


@pytest.fixture
def project(tmp_path):
    kernels = tmp_path / 'Assets' / 'Kernels'
    kernels.mkdir(parents=True)
    (kernels / 'reduce.slang').write_text(REDUCTION, encoding='utf-8')
    (kernels / 'contribution.slang').write_text(
        'import scale_value;\npublic float forSum(float value) { return scaleValue(value); }\n', encoding='utf-8')
    (kernels / 'scale_value.slang').write_text(
        'public float scaleValue(float value) { return 2.0 * value; }\n', encoding='utf-8')
    return tmp_path


def _declaration(*, entry_point='reduceRows', domain='values', constants=None):
    return compute.kernel(Path(SOURCE_PATH), entry_point=entry_point, domain=domain, libraries=LIBRARIES,
                          constants=constants)


@pytest.mark.parametrize('target', ['spirv', 'wgsl'])
def test_native_slang_compiles_reflected_group_with_transitive_sources(project, target):
    params = (_gpu_buffer_descriptor((67,), np.float32), _gpu_buffer_descriptor((17,), np.float32))
    with using_project_root(project):
        artifact = _declaration()._compile_artifact(params, None, 'none', (), target)
        leaf = project / LIBRARIES['scale_value']
        leaf.write_text(leaf.read_text(encoding='utf-8').replace('2.0', '3.0'), encoding='utf-8')
        revised = _declaration()._compile_artifact(params, None, 'none', (), target)
        with pytest.raises(compute.KernelCompilationError, match="domain 'missing'"):
            _declaration(domain='missing')._compile_artifact(params, None, 'none', (), target)
        with pytest.raises(compute.KernelCompilationError, match="Entry 'missing'"):
            _declaration(entry_point='missing')._compile_artifact(params, None, 'none', (), target)
        leaf.write_text('public float scaleValue(float value) { return missing; }\n', encoding='utf-8')
        with pytest.raises(compute.KernelCompilationError) as failure:
            _declaration()._compile_artifact(params, None, 'none', (), target)

    task, = artifact.task_metadata
    assert artifact.target == target and artifact.domain_parameter == 0
    assert task['entry_point'] == 'reduceRows' and task['threads_per_group'] == 4
    assert [(binding['binding'], binding['access']) for binding in task['buffer_bindings']] == [(0, 1), (1, 3)]
    assert not artifact.continuation.eligible and not artifact.continuation.fully_written_output_parameters
    assert artifact.identity != revised.identity and artifact.tasks != revised.tasks
    if target == 'spirv':
        assert artifact.tasks[0].startswith(b'\x03\x02\x23\x07')
    else:
        assert b'workgroupBarrier()' in artifact.tasks[0] and b'arrayLength(' in artifact.tasks[0]
    assert any(row['path'].endswith(LIBRARIES['scale_value']) and row['line'] == 1
               for row in failure.value.diagnostics), failure.value.diagnostics


def _typed_transfer(project, storage_type):
    path = 'Assets/Kernels/typed_transfer.slang'
    (project / path).write_text(f'''
[[vk::binding(0, 0)]] StructuredBuffer<{storage_type}> values;
[[vk::binding(1, 0)]] RWStructuredBuffer<{storage_type}> result;
[shader("compute")] [numthreads(4, 1, 1)]
void transfer(uint3 thread : SV_DispatchThreadID)
{{
    uint count, stride;
    values.GetDimensions(count, stride);
    if (thread.x < count)
        result[thread.x] = values[thread.x] * 2.0 + float(stride);
}}
''', encoding='utf-8')
    return compute.kernel(path, entry_point='transfer', domain='values')


@pytest.mark.parametrize('storage_type, dtype, packed_stride', [
    ('float, Std140DataLayout', np.float32, 4),
    ('float2, Std140DataLayout', compute.vec2, 8),
])
def test_native_slang_rejects_mismatched_storage_stride(project, storage_type, dtype, packed_stride):
    kernel = _typed_transfer(project, storage_type)
    params = (_gpu_buffer_descriptor((5,), dtype),) * 2
    with using_project_root(project), pytest.raises(
        compute.KernelCompilationError, match=f'16-byte elements on spirv.*{packed_stride}-byte elements'
    ):
        kernel._compile_artifact(params, None, 'none', (), 'spirv')


@pytest.mark.parametrize('storage_type, dtype, lanes', [
    ('float2', compute.vec2, 2),
    ('float4', compute.vec4, 4),
])
def test_native_slang_launches_packed_vector_buffers(project, engine, monkeypatch, storage_type, dtype, lanes):
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    kernel = _typed_transfer(project, storage_type)
    values = np.arange(5 * lanes, dtype=np.float32).reshape(5, lanes)
    with using_project_root(project), ExitStack() as resources:
        resources.callback(host._release_lease)
        resources.callback(kernel._release_engine_resources)
        source = resources.enter_context(compute.buffer(shape=5, dtype=dtype, device='gpu', data=values))
        result = resources.enter_context(compute.buffer(shape=5, dtype=dtype, device='gpu'))
        web = kernel._compile_artifact((source, result), None, 'none', (), 'wgsl')
        assert [layout['dtype'] for layout in web.argument_layout['parameters']] == [f'vector{lanes}'] * 2
        compute.launch(kernel, (source, result))
        with result.get_data_async().get_data() as readback:
            np.testing.assert_array_equal(readback.numpy(), values * 2 + 4 * lanes)


def test_native_slang_kernel_launch_and_work_share_gpu_resources(project, engine, monkeypatch):
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    values = np.arange(68, dtype=np.float32) - 18
    expected = (2 * values.reshape(17, 4).sum(axis=1)).astype(np.float32)
    with using_project_root(project), ExitStack() as resources:
        resources.callback(host._release_lease)
        kernel = _declaration()
        for declaration in (initialize_sums, python_reduce, kernel):
            resources.callback(declaration._release_engine_resources)
        source = resources.enter_context(compute.buffer(shape=68, dtype=np.float32, device='gpu', data=values))
        inputs = resources.enter_context(source.view(readonly=True))
        output = resources.enter_context(compute.buffer(shape=17, dtype=np.float32, device='gpu'))
        params = inputs, output
        host.reset_statistics()
        for declaration in (python_reduce, kernel):
            output.fill(-500.0)
            compute.prepare(declaration, params)
            compute.launch(declaration, params)
            with output.get_data_async().get_data() as result:
                np.testing.assert_array_equal(result.numpy(), expected)
            work = compute.prepare_work(((initialize_sums, (output,)), (declaration, params)),
                                        private_buffers=(output,), inputs={}, outputs={'sums': output})
            resources.callback(work.close)
            assert work.outputs == ('sums',) and work._native.dispatch_count == 2

        # The unpadded input uses the same target artifact and reflects the tail.
        tail = resources.enter_context(compute.buffer(shape=67, dtype=np.float32, device='gpu', data=values[:67]))
        compute.launch(kernel, (tail, output))
        expected[-1] -= 2 * values[-1]
        with output.get_data() as result:
            np.testing.assert_array_equal(result.numpy(), expected)
        assert kernel._executable(params) is kernel._executable((tail, output))
        assert host.get_statistics()['dispatch_count'] == 3

        with pytest.raises(ValueError, match='prior complete writer or input port'):
            compute.prepare_work(((kernel, params),), private_buffers=(output,), inputs={}, outputs={'sums': output})
        with inputs.view(offset=1) as subrange, pytest.raises(ValueError, match='full GPU buffer allocation'):
            compute.launch(kernel, (subrange, output))
        with output.view(readonly=True) as readonly, pytest.raises(ValueError, match='read-only'):
            compute.launch(kernel, (inputs, readonly))
        with pytest.raises(TypeError, match='whole dispatch'):
            compute.prepare_continuation(kernel, params)


def _freeze_work_sources(project, data, *, work_source=WORK_SOURCE, slang_source=RUNTIME_ENTRY):
    work = project / 'Assets/Scripts/Work.py'
    work.parent.mkdir(parents=True)
    work.write_text(work_source, encoding='utf-8')
    (project / SOURCE_PATH).write_text(slang_source, encoding='utf-8')
    for relative in (SOURCE_PATH, *LIBRARIES.values()):
        destination = data / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((project / relative).read_bytes())
    return work


def _work_artifacts(data):
    root = data / 'Library/Artifacts/Compute'
    manifest = json.loads((root / 'AotManifest.json').read_text(encoding='utf-8'))
    return [root / row['artifact'] for row in manifest['artifacts'] if row['function'] == 'Scripts.Work.Run.reduce']


@pytest.mark.parametrize(('player', 'target'), [('Player/Windows', 'spirv'), ('Web/Player', 'wgsl')])
def test_stage_slang_work_matches_the_frozen_closure_and_player_target(project, player, target):
    data = project / 'build/Data'
    work = _freeze_work_sources(project, data)
    params = (_gpu_buffer_descriptor((67,), np.float32), _gpu_buffer_descriptor((17,), np.float32))
    with using_project_root(project):
        prepared = _declaration()._compile_artifact(params, None, 'none', (), 'spirv')
        _declaration(entry_point='scaleRows')._compile_artifact(params, None, 'none', (), 'spirv')
    work.write_text(WORK_SOURCE.replace('reduceRows', 'scaleRows'), encoding='utf-8')
    leaf_path = LIBRARIES['scale_value']
    new_leaf = 'public float scaleValue(float value) { return 3.0 * value; }\n'
    (project / leaf_path).write_text(new_leaf, encoding='utf-8')

    snapshots = {str(work): WORK_SOURCE}
    result = stage_compute_artifacts(project, (work,), data, target=player, source_snapshots=snapshots)
    path, = _work_artifacts(data)
    header, selected = _read_artifact(path)
    assert result.kernel_count == 1 and result.artifact_count == 4
    assert selected.target == target and selected.native_dispatch
    assert selected.task_metadata[0]['entry_point'] == 'reduceRows'
    assert path.stem == selected.identity
    assert json.loads(header['specialization'])['parameters'] == [['buffer', 'float32', 1]] * 2
    assert (selected.identity == prepared.identity) == (target == 'spirv')

    # A later build with changed imported bytes has no matching editor prepare.
    (data / leaf_path).write_text(new_leaf, encoding='utf-8')
    result = stage_compute_artifacts(project, (work,), data, target=player, source_snapshots=snapshots)
    assert result.kernel_count == 1 and result.artifact_count == 3
    assert not _work_artifacts(data) and not path.exists()


def test_player_launches_the_prepared_slang_work_and_compiles_a_new_entry(project, engine, monkeypatch):
    data = project / 'build/Data'
    work = _freeze_work_sources(project, data)
    author = ModuleType('Scripts.Work')
    exec(compile(WORK_SOURCE, str(work), 'exec'), vars(author))
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    values = np.arange(68, dtype=np.float32) - 18
    with ExitStack() as resources:
        resources.callback(host._release_lease)
        resources.callback(author.Run.reduce._release_engine_resources)
        with using_project_root(project):
            source = resources.enter_context(compute.buffer(shape=68, dtype=np.float32, device='gpu', data=values))
            output = resources.enter_context(compute.buffer(shape=17, dtype=np.float32, device='gpu'))
            prepared = author.prepare_work(source, output)
            prepared.close()
        stage_compute_artifacts(project, (work,), data, target='Player/Windows')
        packed, = _work_artifacts(data)

        # Player resolution uses the packaged Blob identities and Python bytecode.
        artifacts, records = [], []
        for index, relative in enumerate((SOURCE_PATH, *LIBRARIES.values())):
            guid = f'{index:032x}'
            path = f'Library/Artifacts/Blob/{guid}.slang'
            destination = data / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            (data / relative).replace(destination)
            artifacts.append({'runtime_artifact_id': guid, 'runtime_path': path, 'asset_guid': guid,
                              'package': 'Content.inxpkg', 'content_bytes': destination.stat().st_size})
            records.append({'guid': guid, 'runtime_path': relative, 'primary_runtime_artifact_id': guid,
                            'runtime_artifact_ids': [guid]})
        catalog = PlayerRuntimeAssetCatalog.from_documents(str(data), {'artifacts': artifacts}, {'entries': records})
        bytecode = py_compile.compile(str(work), cfile=str(data / 'Work.pyc'), dfile='Assets/Scripts/Work.py', doraise=True)
        player = ModuleType('Scripts.Work')
        SourcelessFileLoader(player.__name__, bytecode).exec_module(player)
        resources.callback(player.Run.reduce._release_engine_resources)

        compile_program = _Infernux._compile_shader_program
        calls = []

        def record_compile(source, path, *args, **kwargs):
            calls.append(path)
            return compile_program(source, path, *args, **kwargs)

        monkeypatch.setattr(_Infernux, '_compile_shader_program', record_compile)
        monkeypatch.setattr(Application, 'is_player', staticmethod(lambda: True))
        with using_project_root(data):
            set_runtime_asset_resolver(catalog.resolve_guid)
            set_runtime_asset_query(catalog.query_asset_guids)
            try:
                staged = player.prepare_work(source, output)
                staged.close()
                compute.launch(player.Run.reduce, (source, output))
                assert player.Run.reduce._executable((source, output)).artifact.identity == packed.stem
                with output.get_data() as result:
                    np.testing.assert_array_equal(result.numpy(), 2 * values.reshape(17, 4).sum(axis=1))
                assert not calls

                cold = _declaration(entry_point='scaleRows')
                resources.callback(cold._release_engine_resources)
                scaled = resources.enter_context(compute.buffer(shape=68, dtype=np.float32, device='gpu'))
                compute.prepare(cold, (source, scaled))
                compute.launch(cold, (source, scaled))
                with scaled.get_data() as result:
                    np.testing.assert_array_equal(result.numpy(), 2 * values)
                assert calls == [SOURCE_PATH]
            finally:
                set_runtime_asset_resolver(None)
                set_runtime_asset_query(None)


def test_native_slang_link_constants_report_the_extern_and_validate_declarations(project):
    (project / SOURCE_PATH).write_text(TILED_REDUCTION, encoding='utf-8')
    with pytest.raises(TypeError, match="'tile' requires a Python int"):
        _declaration(constants={'tile': True})
    with pytest.raises(ValueError, match="'tile' must fit a signed 32-bit int"):
        _declaration(constants={'tile': 1 << 31})
    params = (_gpu_buffer_descriptor((67,), np.float32), _gpu_buffer_descriptor((17,), np.float32))
    with using_project_root(project), pytest.raises(compute.KernelCompilationError) as failure:
        _declaration(constants={'other': 4})._compile_artifact(params, None, 'none', (), 'spirv')
    assert any(row['path'].endswith(SOURCE_PATH) and row['line'] == 2 and "'tile'" in row['message']
               for row in failure.value.diagnostics), failure.value.diagnostics


def test_native_slang_link_constants_are_frozen_across_work_launches_and_player_cook(project, engine, monkeypatch):
    data = project / 'build/Data'
    script = _freeze_work_sources(project, data, work_source=TILED_WORK_SOURCE, slang_source=TILED_REDUCTION)
    author = ModuleType('Scripts.Work')
    exec(compile(TILED_WORK_SOURCE, str(script), 'exec'), vars(author))
    options = {'tile': 4}
    same_tile = _declaration(constants=options)
    options['tile'] = 8

    compile_program = _Infernux._compile_shader_program
    calls = []

    def record_compile(source, path, *args, **kwargs):
        if path == SOURCE_PATH:
            calls.append(kwargs['targets'][0])
        return compile_program(source, path, *args, **kwargs)

    monkeypatch.setattr(_Infernux, '_compile_shader_program', record_compile)
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    values = np.arange(67, dtype=np.float32) - 18
    with using_project_root(project), ExitStack() as resources:
        resources.callback(host._release_lease)
        for declaration in (author.Run.narrow, author.Run.wide, same_tile):
            resources.callback(declaration._release_engine_resources)
        source = resources.enter_context(compute.buffer(shape=67, dtype=np.float32, device='gpu', data=values))
        new_source = resources.enter_context(compute.buffer(shape=65, dtype=np.float32, device='gpu', data=-values[:65]))
        narrow = resources.enter_context(compute.buffer(shape=17, dtype=np.float32, device='gpu'))
        wide = resources.enter_context(compute.buffer(shape=9, dtype=np.float32, device='gpu'))
        prepared = author.prepare_work(source, narrow, wide)
        resources.callback(prepared.close)
        assert prepared._native.dispatch_count == 2
        narrow_code = author.Run.narrow._executable((source, narrow)).artifact
        wide_code = author.Run.wide._executable((source, wide)).artifact
        assert narrow_code.identity != wide_code.identity and narrow_code.tasks != wide_code.tasks
        assert [code.task_metadata[0]['threads_per_group'] for code in (narrow_code, wide_code)] == [4, 8]
        assert calls == ['spirv', 'spirv']

        def launch_and_check(declaration, inputs, output, expected, width):
            compute.launch(declaration, (inputs, output))
            with output.get_data() as actual:
                np.testing.assert_array_equal(actual.numpy(), np.add.reduceat(2 * expected, np.arange(0, len(expected), width)))

        launch_and_check(author.Run.narrow, source, narrow, values, 4)
        launch_and_check(author.Run.wide, source, wide, values, 8)
        source.set_data(3 * values)
        launch_and_check(author.Run.narrow, source, narrow, 3 * values, 4)
        launch_and_check(same_tile, new_source, narrow, -values[:65], 4)
        assert same_tile._executable((new_source, narrow)).artifact.identity == narrow_code.identity
        assert calls == ['spirv', 'spirv']

    cache = data / 'Library/Artifacts/Compute'
    for player, target in (('Player/Windows', 'spirv'), ('Web/Player', 'wgsl')):
        result = stage_compute_artifacts(project, (script,), data, target=player)
        assert result.kernel_count == 2 and result.artifact_count == 5
        rows = json.loads((cache / 'AotManifest.json').read_text(encoding='utf-8'))['artifacts']
        tiles = {row['function']: _read_artifact(cache / row['artifact'])[1] for row in rows
                 if row['function'].startswith('Scripts.Work.Run.')}
        for name, width in (('narrow', 4), ('wide', 8)):
            artifact = tiles[f'Scripts.Work.Run.{name}']
            assert artifact.target == target and artifact.task_metadata[0]['threads_per_group'] == width
        assert len({artifact.identity for artifact in tiles.values()}) == 2
    assert calls == ['spirv', 'spirv', 'wgsl', 'wgsl']


def _write_calibration_project(project):
    work = 'Assets/Scripts/Work.py'
    for path, text in ((work, CALIBRATION_WORK_SOURCE), (CALIBRATION_PATH, CALIBRATION),
                       *((path, text) for _, path, text in CALIBRATION_MODULES)):
        destination = project / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding='utf-8')
    return project / work


def test_selected_slang_entry_reuses_only_the_same_frontend_snapshot():
    def compile_entry(entry_point, *, path='Assets/Cache/first.slang', libraries=CALIBRATION_MODULES,
                      targets=('spirv',), precise=True):
        result = _Infernux._compile_shader_program(CALIBRATION, path, 'inx_kernel', libraries=libraries,
                                                  targets=targets, precise=precise, entry_point=entry_point)
        assert [target['target'] for target in result['targets']] == list(targets)
        assert all([entry['name'] for entry in target['entries']] == [entry_point] for target in result['targets'])
        return result['frontend_reused'], result['targets'][0]['entries'][0]['code']

    reused, code = compile_entry('calibrateFloat')
    assert not reused
    assert compile_entry('calibrateInt')[0]
    ordinary = _Infernux._compile_shader_program(CALIBRATION, 'Assets/Cache/first.slang', 'inx_kernel',
                                                libraries=CALIBRATION_MODULES, targets=['spirv'], precise=True)
    assert 'frontend_reused' not in ordinary
    assert [entry['name'] for entry in ordinary['targets'][0]['entries']] == ['calibrateFloat', 'calibrateInt']
    assert compile_entry('calibrateFloat') == (True, code)

    assert not compile_entry('calibrateFloat', path='Assets/Cache/second.slang')[0]
    assert not compile_entry('calibrateFloat')[0]  # The resident slot retains one snapshot.
    leaf_name, leaf_path, leaf_source = CALIBRATION_MODULES[-1]
    revised = (*CALIBRATION_MODULES[:-1], (leaf_name, leaf_path, leaf_source.replace('value * scale', 'value + scale')))
    reused, revised_code = compile_entry('calibrateFloat', libraries=revised)
    assert not reused and revised_code != code
    assert compile_entry('calibrateInt', libraries=revised)[0]
    assert not compile_entry('calibrateFloat', libraries=revised, precise=False)[0]
    assert not compile_entry('calibrateInt', libraries=revised, precise=False, targets=('spirv', 'wgsl'))[0]
    assert compile_entry('calibrateFloat', libraries=revised, precise=False, targets=('spirv', 'wgsl'))[0]


def test_native_slang_generic_module_prepares_typed_work_and_cooks_both_entries(project, engine, monkeypatch):
    work_script = _write_calibration_project(project)
    data = project / 'build/Data'
    _write_calibration_project(data)
    author = ModuleType('Scripts.Work')
    exec(compile(CALIBRATION_WORK_SOURCE, str(work_script), 'exec'), vars(author))
    compile_program = _Infernux._compile_shader_program
    compilations = []

    def track(source, path, *args, **kwargs):
        result = compile_program(source, path, *args, **kwargs)
        if path == CALIBRATION_PATH:
            entry, = result['targets'][0]['entries']
            compilations.append((kwargs['targets'][0], entry['name'], result['frontend_reused']))
        return result

    monkeypatch.setattr(_Infernux, '_compile_shader_program', track)
    host = engine._acquire_compute_host()
    monkeypatch.setattr(compute, '_native_compute_host', lambda: host)
    real_values = np.arange(67, dtype=np.float32) - 17.5
    int_values = np.arange(67, dtype=np.int32) - 28
    with using_project_root(project), ExitStack() as resources:
        resources.callback(host._release_lease)
        resources.callback(author.Calibration.real._release_engine_resources)
        resources.callback(author.Calibration.integer._release_engine_resources)
        real = resources.enter_context(compute.buffer(shape=67, dtype=np.float32, device='gpu', data=real_values))
        integer = resources.enter_context(compute.buffer(shape=67, dtype=np.int32, device='gpu', data=int_values))
        real_result = resources.enter_context(compute.buffer(shape=67, dtype=np.float32, device='gpu'))
        int_result = resources.enter_context(compute.buffer(shape=67, dtype=np.int32, device='gpu'))
        recipe = author.prepare_work(real, real_result, integer, int_result)
        resources.callback(recipe.close)
        assert recipe._native.dispatch_count == 2
        assert compilations == [('spirv', 'calibrateFloat', False), ('spirv', 'calibrateInt', True)]
        compute.launch(author.Calibration.real, (real, real_result))
        compute.launch(author.Calibration.integer, (integer, int_result))
        with real_result.get_data() as real_data, int_result.get_data() as int_data:
            np.testing.assert_array_equal(real_data.numpy(), real_values * 0.5 + 1.25)
            np.testing.assert_array_equal(int_data.numpy(), int_values * 3 + 2)

    result = stage_compute_artifacts(project, (work_script,), data, target='Web/Player')
    assert result.kernel_count == 2 and result.artifact_count == 5
    assert compilations[2:] == [('wgsl', 'calibrateInt', False), ('wgsl', 'calibrateFloat', True)]
    cache = data / 'Library/Artifacts/Compute'
    rows = json.loads((cache / 'AotManifest.json').read_text(encoding='utf-8'))['artifacts']
    artifacts = {row['function']: _read_artifact(cache / row['artifact'])[1] for row in rows
                 if row['function'].startswith('Scripts.Work.Calibration.')}
    for name, dtype, binding in (('real', 'float32', 0), ('integer', 'int32', 2)):
        artifact = artifacts[f'Scripts.Work.Calibration.{name}']
        assert artifact.target == 'wgsl'
        assert [item['dtype'] for item in artifact.argument_layout['parameters']] == [dtype, dtype]
        assert [item['binding'] for item in artifact.task_metadata[0]['buffer_bindings']] == [binding, binding + 1]
