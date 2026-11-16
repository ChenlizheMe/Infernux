"""Project Slang entries use the public compute runtime and the cooked asset ABI."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from infernux import compute
from infernux.engine.build.compute_aot import _gpu_buffer_descriptor
from infernux.engine.project_context import using_project_root


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
    return compute.kernel(Path('Assets/Kernels/reduce.slang'), entry_point=entry_point, domain=domain, libraries=LIBRARIES)


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
