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


def _declaration(*, entry_point='reduceRows', domain='values'):
    return compute.kernel(Path(SOURCE_PATH), entry_point=entry_point, domain=domain, libraries=LIBRARIES)


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


def _freeze_work_sources(project, data):
    work = project / 'Assets/Scripts/Work.py'
    work.parent.mkdir(parents=True)
    work.write_text(WORK_SOURCE, encoding='utf-8')
    (project / SOURCE_PATH).write_text(RUNTIME_ENTRY, encoding='utf-8')
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
