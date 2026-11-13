"""Real Vulkan regression for the public kernel/index/launch author path."""

from __future__ import annotations

import gc
import linecache
import math
from pathlib import Path
import sys
import tempfile
import traceback
import weakref
from types import ModuleType

import numpy as np

import infernux as inx


@inx.compute.function
def _affine_term(value, scale, bias):
    return value * scale + bias


@inx.compute.kernel
def _helper_kernel(domain_values, output_values, scale, bias):
    i = inx.compute.index(domain_values)
    output_values[i] = _affine_term(domain_values[i], scale, bias)


@inx.compute.function
def _nested_term(value, scale, bias=4):
    result = _affine_term(value, scale, bias)
    if result < 0:
        result = -result
    for increment in range(3):
        result += increment
    return result


@inx.compute.kernel
def _nested_helper_kernel(domain_values, output_values):
    i = inx.compute.index(domain_values)
    output_values[i] = _nested_term(domain_values[i], -2)


@inx.compute.function
def _dynamic_nested_term(value, limit, break_at):
    result = value
    for outer in range(limit):
        for inner in range(limit):
            if inner == 0 or inner == 3:
                continue
            result += outer + inner
            if inner == break_at:
                break
        else:
            result += 7
        if outer == break_at:
            break
    else:
        result += 11
    return result


@inx.compute.kernel
def _dynamic_nested_helper_kernel(domain_values, output_values, limit, break_at):
    i = inx.compute.index(domain_values)
    output_values[i] = _dynamic_nested_term(domain_values[i], limit, break_at)


@inx.compute.function
def _recursive_helper(value):
    return _recursive_helper(value)


@inx.compute.kernel
def _recursive_kernel(domain_values):
    i = inx.compute.index(domain_values)
    domain_values[i] = _recursive_helper(domain_values[i])


@inx.compute.function
def _wave_term(value):
    return math.sqrt(value * value + 1.0) + math.sin(value)


@inx.compute.kernel
def _wave_kernel(values):
    i = inx.compute.index(values)
    values[i] = _wave_term(values[i])


@inx.compute.kernel
def _intrinsic_components(values, sine, root):
    i = inx.compute.index(values)
    sine[i] = math.sin(values[i])
    root[i] = math.sqrt(values[i] * values[i] + 1.0)


@inx.compute.function
def _floor_row(value):
    return math.floor(value)


@inx.compute.kernel
def _literal_math(values):
    i = inx.compute.index(values)
    sine = math.sin(1)
    root = math.sqrt(2)
    quotient = (i + 1) / 2
    identity = math.floor(16777217)
    correction = identity - 16777217
    row = _floor_row(float(i) + 0.25)
    values[row] = sine + root + quotient + correction


def _check_imported_helpers(project, domain, output):
    def load(name, source, **bindings):
        path = project / 'Assets' / (name + '.py')
        path.write_text(source, encoding='utf8')
        module = ModuleType(name)
        module.__dict__.update(bindings)
        exec(compile(source, str(path), 'exec'), module.__dict__)
        return module

    first = load('第一套数学', '''import infernux as inx
import math as numeric
GAIN = 2
@inx.compute.function
def apply(value, /, bias=GAIN):
    return numeric.sqrt(value * value) * GAIN + bias
''')
    second = load('第二套数学', '''import infernux as inx
GAIN = 5
@inx.compute.function
def apply(value):
    return value * GAIN
''')
    middle = load('共享数学', '''import infernux as inx
@inx.compute.function
def combine(value):
    return first.apply(value) + second.apply(value)
''', first=first, second=second)
    owner = load('跨文件入口', '''import infernux as inx
GAIN = 100
@inx.compute.kernel
def run(domain, output):
    i = inx.compute.index(domain)
    output[i] = library.combine(domain[i]) + GAIN
''', library=middle)
    old, edited = owner.run, None
    try:
        inx.compute.launch(old, params=(domain, output))
        np.testing.assert_array_equal(output.get_data().numpy(), np.arange(17) * 7 + 102)
        first.GAIN = 3
        edited = inx.compute.kernel(old.function)
        inx.compute.launch(edited, params=(domain, output))
        np.testing.assert_array_equal(output.get_data().numpy(), np.arange(17) * 8 + 102)
        inx.compute.launch(old, params=(domain, output))
        np.testing.assert_array_equal(output.get_data().numpy(), np.arange(17) * 7 + 102)
        print('IMPORTED_HELPERS_GPU_OK own_scopes defaults aliases frozen_old_program', flush=True)
    finally:
        old._release_engine_resources()
        if edited is not None:
            edited._release_engine_resources()


def main() -> int:
    # The embedded compiler must not claim or temporarily replace another
    # package's import namespace. GPU execution below still uses the real RHI.
    occupied = {name: ModuleType(name) for name in ("taichi", "taichi.lang", "taichi._lib")}
    sys.modules.update(occupied)
    with tempfile.TemporaryDirectory(prefix="infernux-public-kernel-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        values = points = velocities = upload_only = declaration = affine = integrate = None
        view_source = view = read_only_view = affine_executable = affine_native = None
        velocities_readonly = None
        rows = matrix = gathered = gather_rows = None
        explicit_domain = explicit_output = local_loop_output = atomic_total = None
        explicit_domain_kernel = local_loops = atomic_reduce = None
        wave_values = wave_repeat = intrinsic_source = sine_values = root_values = None
        failure = None
        try:
            try:
                frontend.init_renderer(64, 64, str(project))
            except (OSError, RuntimeError) as exception:
                print(f"Public GPU kernel test skipped: {exception}")
                return 77
            inx.compute.statistics(reset=True)

            source = np.arange(257, dtype=np.int32)
            with inx.compute._record_commands():
                upload_only = inx.buffer(shape=source.size, dtype=np.int32, device="gpu", data=source)
            np.testing.assert_array_equal(upload_only.get_data().numpy(), source)
            values = inx.buffer(shape=source.size, dtype=np.int32, device="gpu", data=source)

            @inx.compute.kernel
            def affine(payload, scale, bias):
                i = inx.compute.index(payload)
                payload[i] = payload[i] * scale + bias

            declaration = affine
            inx.compute.statistics(reset=True)
            # Cold preparation must not retain unrelated project globals.
            globals()['_compiler_lifetime_probe'] = ModuleType('compiler_lifetime_probe')
            globals_reference = weakref.ref(globals()['_compiler_lifetime_probe'])
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                try:
                    inx.compute.prepare(affine, params=(values, 3, 11))
                finally:
                    del globals()['_compiler_lifetime_probe']
                assert globals_reference() is None, 'Compiler retained unrelated project globals'
            finally:
                if gc_was_enabled:
                    gc.enable()
            assert not any(name.startswith('infernux._compiler.taichi') for name in sys.modules)
            assert not any(name.startswith('<infernux-kernel:') for name in linecache.cache)
            @inx.compute.kernel
            def invalid_lowering(payload):
                i = inx.compute.index(payload)
                payload[i] = undefined_compiler_value  # noqa: F821 -- compiler source diagnostic probe

            try:
                inx.compute.prepare(invalid_lowering, params=(values,))
            except inx.compute.KernelCompilationError as exception:
                assert 'undefined_compiler_value' in str(exception)
                assert 'payload[i] = undefined_compiler_value' in str(exception)
            else:
                raise AssertionError('Invalid compiler AST unexpectedly accepted')
            assert not any(name.startswith('<infernux-kernel:') for name in linecache.cache)
            prepared_statistics = inx.compute.statistics()
            assert prepared_statistics.submission_count == 0
            assert prepared_statistics.dispatch_count == 0
            with inx.compute._record_commands():
                inx.compute.launch(affine, params=(values, 3, 11))
                inx.compute.launch(affine, params=(values, 2, -5))
            expected_values = (source * 3 + 11) * 2 - 5
            np.testing.assert_array_equal(values.get_data().numpy(), expected_values)

            # Drop the process-local executable and make private frontend
            # initialization impossible. Preparing the same specialization
            # must reconstruct its Infernux pipeline from the project cache.
            from infernux.lib import _Infernux as compiler_service
            affine._release_engine_resources()
            compile_program = compiler_service._compile_shader_program
            compiler_service._compile_shader_program = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cached GPU artifact tried to compile source")
            )
            try:
                inx.compute.prepare(affine, params=(values, 3, 11))
            finally:
                compiler_service._compile_shader_program = compile_program
            readback = values.get_data_async(offset=31, count=67)
            assert isinstance(readback.done, bool)
            values.close()
            np.testing.assert_array_equal(readback.get_data().numpy(), expected_values[31:98])

            view_source = inx.buffer(shape=source.size, dtype=np.int32, device="gpu", data=source)
            view = view_source.view(offset=17, count=23)
            assert view.description.ownership == "borrowed"
            assert view.description.byte_offset == 17 * np.dtype(np.int32).itemsize
            assert view.description.resource_index == view_source.description.resource_index
            assert view.description.resource_generation == view_source.description.resource_generation
            read_only_view = view.view(readonly=True)
            assert read_only_view.description.access == "read_only"
            try:
                inx.compute.launch(affine, params=(read_only_view, 1, 0))
            except ValueError as exception:
                assert "read-only" in str(exception)
            else:
                raise AssertionError("GPU kernel wrote a read-only inx.buffer view")
            read_only_view.close()
            inx.compute.launch(affine, params=(view, -2, 7))
            expected_view_source = source.copy()
            expected_view_source[17:40] = expected_view_source[17:40] * -2 + 7
            np.testing.assert_array_equal(view_source.get_data().numpy(), expected_view_source)
            view_source.close()
            np.testing.assert_array_equal(view.get_data().numpy(), expected_view_source[17:40])
            affine_executable = next(iter(affine._executables.values()))
            affine_native = affine_executable.tasks[0][0]
            groups_with_view = int(affine_native.cached_binding_group_count)
            view.close()
            affine_native.wait()
            inx.compute._flush_commands()
            assert int(affine_native.pending_dispatch_count) == 0
            assert int(affine_native.cached_binding_group_count) == groups_with_view - 1

            # Rebuilding a logical buffer with another capacity creates a new
            # RHI generation. Closed generations must leave the kernel cache
            # after their exact submission completes instead of accumulating
            # until engine shutdown.
            stable_group_count = int(affine_native.cached_binding_group_count)
            for capacity in (7, 19, 41, 83):
                replacement = inx.buffer(
                    shape=capacity, dtype=np.int32, device="gpu",
                    data=np.arange(capacity, dtype=np.int32),
                )
                inx.compute.launch(affine, params=(replacement, 1, capacity))
                replacement.close()
                affine_native.wait()
                inx.compute._flush_commands()
                assert int(affine_native.cached_binding_group_count) == stable_group_count

            point_source = np.arange(129 * 3, dtype=np.float32).reshape(129, 3) * 0.01
            velocity_source = np.full((129, 3), (0.5, -0.25, 1.0), dtype=np.float32)
            points = inx.buffer(shape=129, dtype=inx.vector3, device="gpu", data=point_source)
            velocities = inx.buffer(shape=129, dtype=inx.vector3, device="gpu", data=velocity_source)
            velocities_readonly = velocities.view(readonly=True)

            @inx.compute.kernel
            def integrate(position_values, velocity_values, delta_time):
                i = inx.compute.index(position_values)
                position_values[i] = position_values[i] + velocity_values[i] * delta_time

            inx.compute.launch(integrate, params=(points, velocities_readonly, 0.125))
            integrate_executable = next(iter(integrate._executables.values()))
            declared_access = {}
            for task in integrate_executable.artifact.task_metadata:
                for binding in task["buffer_bindings"]:
                    if binding["resource_kind"] != "external_buffer":
                        continue
                    parameter = int(binding["argument_indices"][0])
                    declared_access[parameter] = declared_access.get(parameter, 0) | int(binding["access"])
            assert declared_access == {0: 3, 1: 1}
            np.testing.assert_allclose(
                points.get_data().numpy(), point_source + velocity_source * 0.125, rtol=1e-6, atol=1e-6
            )

            rows = inx.buffer(shape=4, dtype=np.int32, device="gpu", data=[3, 1, 0, 2])
            matrix_source = np.arange(12, dtype=np.float32).reshape(4, 3)
            matrix = inx.buffer(shape=(4, 3), dtype=float, device="gpu", data=matrix_source)
            gathered = inx.buffer(shape=4, dtype=float, device="gpu")

            @inx.compute.kernel
            def gather_rows(row_order, source_values, output_values, factor):
                i = inx.compute.index(row_order)
                row = row_order[i]
                total = 0.0
                for axis in range(3):
                    total += source_values[row, axis] * factor
                if total > 0.0:
                    output_values[i] = total

            inx.compute.launch(gather_rows, params=(rows, matrix, gathered, 0.5))
            np.testing.assert_allclose(
                gathered.get_data().numpy(), matrix_source[[3, 1, 0, 2]].sum(axis=1) * 0.5,
                rtol=1e-6, atol=1e-6,
            )

            # The index declaration is metadata for the whole function, not a
            # point where execution suddenly becomes parallel.  Statements on
            # either side execute once for each item in the explicitly named
            # domain.  The output view has the exact pointwise domain length,
            # and the rest of its larger backing buffer must stay untouched.
            explicit_domain = inx.buffer(
                shape=17, dtype=np.int32, device="gpu", data=np.arange(17, dtype=np.int32)
            )
            explicit_output = inx.buffer(
                shape=41, dtype=np.int32, device="gpu", data=np.full(41, -1, dtype=np.int32)
            )

            @inx.compute.kernel
            def explicit_domain_kernel(output_values, domain_values, bias):
                adjusted = bias * 2 + 1
                i = inx.compute.index(domain_values)
                output_values[i] = domain_values[i] + adjusted

            with explicit_output.view(count=17) as output_rows:
                inx.compute.launch(explicit_domain_kernel, params=(output_rows, explicit_domain, 4))
            expected_explicit = np.full(41, -1, dtype=np.int32)
            expected_explicit[:17] = np.arange(17, dtype=np.int32) + 9
            np.testing.assert_array_equal(explicit_output.get_data().numpy(), expected_explicit)
            assert next(iter(explicit_domain_kernel._executables.values())).artifact.domain_parameter == 1

            # Ordinary control flow remains serial inside one work item.  It
            # must not inherit Taichi's historical top-level-loop offloading.
            local_loop_output = inx.buffer(
                shape=17, dtype=np.int32, device="gpu", data=np.zeros(17, dtype=np.int32)
            )

            @inx.compute.kernel
            def local_loops(domain_values, output_values):
                i = inx.compute.index(domain_values)
                total = 0
                cursor = 0
                while cursor < 4:
                    total += i + cursor
                    cursor += 1
                for extra in range(3):
                    total += extra
                output_values[i] = total

            inx.compute.launch(local_loops, params=(explicit_domain, local_loop_output))
            expected_local = np.asarray([4 * i + 9 for i in range(17)], dtype=np.int32)
            np.testing.assert_array_equal(local_loop_output.get_data().numpy(), expected_local)

            @inx.compute.kernel
            def loop_control(domain_values, output_values, limit):
                i = inx.compute.index(domain_values)
                total = 0
                for j in range(-2, 6):
                    if j == 0:
                        continue
                    if j > limit:
                        break
                    total = total * 3 + i + j
                for outer in range(3):
                    for inner in range(2):
                        total += outer * 2 + inner
                for empty in range(0):
                    total += 10000
                output_values[i] = total

            for limit in (0, 3, 7):
                inx.compute.launch(loop_control, params=(explicit_domain, local_loop_output, limit))
                expected_control = []
                for i in range(17):
                    total = 0
                    for j in range(-2, 6):
                        if j == 0:
                            continue
                        if j > limit:
                            break
                        total = total * 3 + i + j
                    expected_control.append(total + 15)
                np.testing.assert_array_equal(local_loop_output.get_data().numpy(), expected_control)

            @inx.compute.kernel
            def dynamic_integer_loops(domain_values, output_values, limit):
                i = inx.compute.index(domain_values)
                total = 0
                for j in range(limit):
                    if j == 1:
                        continue
                    total += i + j
                    if j >= 4:
                        break
                cursor = 0
                while cursor < limit:
                    if cursor == 1 or cursor == 2:
                        cursor += 1
                        continue
                    total += cursor
                    if cursor >= 3:
                        break
                    cursor += 1
                output_values[i] = total

            for limit in (0, 1, 2, 3, 4, 5, 7):
                inx.compute.launch(
                    dynamic_integer_loops,
                    params=(explicit_domain, local_loop_output, limit),
                )
                expected_dynamic = []
                for i in range(17):
                    total = 0
                    for j in range(limit):
                        if j == 1:
                            continue
                        total += i + j
                        if j >= 4:
                            break
                    cursor = 0
                    while cursor < limit:
                        if cursor == 1 or cursor == 2:
                            cursor += 1
                            continue
                        total += cursor
                        if cursor >= 3:
                            break
                        cursor += 1
                    expected_dynamic.append(total)
                np.testing.assert_array_equal(
                    local_loop_output.get_data().numpy(), expected_dynamic
                )

            @inx.compute.kernel
            def dynamic_negative_range(domain_values, output_values, start, stop):
                i = inx.compute.index(domain_values)
                total = 0
                for j in range(start, stop, -2):
                    total += i + j
                output_values[i] = total

            for start, stop in ((7, -2), (2, 2), (-3, -8)):
                inx.compute.launch(
                    dynamic_negative_range,
                    params=(explicit_domain, local_loop_output, start, stop),
                )
                expected_negative = np.asarray([
                    sum(i + j for j in range(start, stop, -2)) for i in range(17)
                ], dtype=np.int32)
                np.testing.assert_array_equal(
                    local_loop_output.get_data().numpy(), expected_negative
                )

            @inx.compute.kernel
            def invalid_return(domain_values):
                i = inx.compute.index(domain_values)
                return i

            try:
                inx.compute.launch(invalid_return, params=(explicit_domain,))
            except TypeError as exception:
                assert isinstance(exception, inx.compute.KernelCompilationError)
                assert "cannot return" in str(exception)
            else:
                raise AssertionError("GPU kernel accepted a per-work-item Python return value")

            captured_value = 3

            @inx.compute.kernel
            def invalid_closure(domain_values):
                i = inx.compute.index(domain_values)
                domain_values[i] = captured_value

            try:
                inx.compute.launch(invalid_closure, params=(explicit_domain,))
            except TypeError as exception:
                assert isinstance(exception, inx.compute.KernelCompilationError)
                assert "closure" in str(exception)
            else:
                raise AssertionError("GPU kernel accepted a captured Python closure value")

            try:
                inx.compute.launch(_recursive_kernel, params=(explicit_domain,))
            except TypeError as exception:
                assert isinstance(exception, inx.compute.KernelCompilationError)
                assert "recursion" in str(exception)
            else:
                raise AssertionError("GPU kernel accepted a recursive compute helper")

            atomic_total = inx.buffer(
                shape=1, dtype=np.int32, device="gpu", data=np.zeros(1, dtype=np.int32)
            )

            @inx.compute.kernel
            def atomic_reduce(domain_values, total):
                i = inx.compute.index(domain_values)
                inx.compute.atomic_add(total[0], domain_values[i] + 1)

            inx.compute.launch(atomic_reduce, params=(explicit_domain, atomic_total))
            np.testing.assert_array_equal(
                atomic_total.get_data().numpy(),
                np.asarray([sum(range(1, 18))], dtype=np.int32),
            )

            @inx.compute.kernel
            def atomic_fetch(domain_values, total, tickets, increment):
                i = inx.compute.index(domain_values)
                tickets[i] = inx.compute.atomic_add(total[0], increment)
                if i == 0:
                    total[1] = total[2] + 1

            for dtype, increment in ((np.int32, -1), (np.uint32, 1), (np.float32, .25)):
                backing = inx.buffer(shape=5, dtype=dtype, device="gpu",
                                     data=np.asarray([11, 0, 7, 33, 44], dtype=dtype))
                total_view = backing.view(offset=1, count=3)
                tickets = inx.buffer(shape=17, dtype=dtype, device="gpu")
                try:
                    inx.compute.launch(atomic_fetch, params=(explicit_domain, total_view, tickets, increment))
                    np.testing.assert_array_equal(backing.get_data().numpy(),
                                                  np.asarray([11, 17 * increment, 34, 33, 44], dtype=dtype))
                    np.testing.assert_array_equal(np.sort(tickets.get_data().numpy()),
                                                  np.sort(np.arange(17, dtype=dtype) * increment))
                finally:
                    tickets.close()
                    total_view.close()
                    backing.close()

            inx.compute.launch(_helper_kernel, params=(explicit_domain, local_loop_output, 3, -2))
            np.testing.assert_array_equal(
                local_loop_output.get_data().numpy(),
                np.arange(17, dtype=np.int32) * 3 - 2,
            )
            _check_imported_helpers(project, explicit_domain, local_loop_output)
            inx.compute.launch(_nested_helper_kernel, params=(explicit_domain, local_loop_output))
            np.testing.assert_array_equal(
                local_loop_output.get_data().numpy(),
                np.abs(np.arange(17, dtype=np.int32) * -2 + 4) + 3,
            )
            for limit, break_at in ((0, 2), (2, 5), (5, 2), (6, 5)):
                inx.compute.launch(
                    _dynamic_nested_helper_kernel,
                    params=(explicit_domain, local_loop_output, limit, break_at),
                )
                expected_nested = []
                for value in range(17):
                    result = value
                    for outer in range(limit):
                        for inner in range(limit):
                            if inner == 0 or inner == 3:
                                continue
                            result += outer + inner
                            if inner == break_at:
                                break
                        else:
                            result += 7
                        if outer == break_at:
                            break
                    else:
                        result += 11
                    expected_nested.append(result)
                np.testing.assert_array_equal(
                    local_loop_output.get_data().numpy(), expected_nested
                )
            # Intrinsics in helpers must resolve to the same private compiler,
            # including when a public Taichi namespace is already occupied.
            wave_source = np.linspace(-2.0, 2.0, 33, dtype=np.float32)
            wave_values = inx.buffer(shape=33, dtype=np.float32, device="gpu", data=wave_source)
            inx.compute.launch(_wave_kernel, params=(wave_values,))
            wave_actual = wave_values.get_data().numpy().copy()
            intrinsic_source = inx.buffer(shape=33, dtype=np.float32, device="gpu", data=wave_source)
            sine_values = inx.buffer(shape=33, dtype=np.float32, device="gpu")
            root_values = inx.buffer(shape=33, dtype=np.float32, device="gpu")
            inx.compute.launch(_intrinsic_components, params=(intrinsic_source, sine_values, root_values))
            sine_actual = sine_values.get_data().numpy().copy()
            root_actual = root_values.get_data().numpy().copy()
            # Vulkan's GLSL.std.450 single-precision Sin contract permits an
            # absolute error of 2^-11 on [-pi, pi]. CPU libm equality is not
            # its contract. Keep arithmetic/root and composition checks tight.
            # https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#precision-of-glsl-std-450-instructions
            np.testing.assert_allclose(sine_actual, np.sin(wave_source.astype(np.float64)),
                                       rtol=0, atol=2.0 ** -11)
            np.testing.assert_allclose(root_actual, np.sqrt(wave_source * wave_source + 1.0),
                                       rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(
                wave_actual, root_actual + sine_actual,
                rtol=2e-6, atol=2e-6,
            )
            wave_repeat = inx.buffer(shape=33, dtype=np.float32, device="gpu", data=wave_source)
            inx.compute.launch(_wave_kernel, params=(wave_repeat,))
            np.testing.assert_array_equal(wave_repeat.get_data().numpy(), wave_actual)
            inx.compute.launch(_literal_math, params=(root_values,))
            np.testing.assert_allclose(root_values.get_data().numpy(),
                                       math.sin(1) + math.sqrt(2) + (np.arange(33) + 1) / 2,
                                       rtol=0, atol=2.0 ** -11 + 2e-6)
            statistics = inx.compute.statistics()
            assert statistics.submission_count > 0
            assert statistics.dispatch_count >= 7
            assert statistics.upload_request_count > 0
            assert statistics.upload_bytes >= source.nbytes
            assert statistics.readback_request_count > 0
            assert statistics.readback_bytes >= source.nbytes
            assert statistics.staging_allocation_count > 0
            assert statistics.host_map_count > 0
            assert statistics.native_boundary_count > 0
            assert statistics.wait_count > 0
            assert statistics.cpu_submit_ms > 0.0
            assert statistics.wait_ms >= 0.0
            assert statistics.pending_submission_count >= 0
            assert statistics.gpu_profile_available and statistics.gpu_profile_serial > 0
            assert statistics.gpu_time_ms is not None and statistics.gpu_time_ms >= 0.0

            inx.compute.set_profiling_enabled(False)
            inx.compute.launch(affine, params=(rows, 1, 0))
            rows.get_data()
            unprofiled = inx.compute.statistics()
            assert not unprofiled.gpu_profile_available and unprofiled.gpu_time_ms is None

            inx.compute.set_profiling_enabled(True)
            inx.compute.launch(affine, params=(rows, 1, 0))
            rows.get_data()
            profiled = inx.compute.statistics()
            assert profiled.gpu_profile_available
            assert profiled.gpu_profile_serial > 0
            assert profiled.gpu_time_ms is not None and profiled.gpu_time_ms >= 0.0
            inx.compute.set_profiling_enabled(False)
            assert not any(name.startswith('infernux._compiler.taichi') for name in sys.modules)
            assert {name: module for name, module in sys.modules.items()
                    if name == "taichi" or name.startswith("taichi.")} == occupied
        except BaseException:
            failure = traceback.format_exc()
        finally:
            try:
                # Keep the module-level-style Python references alive while the
                # engine releases compiler/RHI resources at its shutdown boundary.
                # This is the editor/plugin-reload ownership case, not a GC test.
                inx.compute._release_engine_resources()
                affine_executable = affine_native = None
                gc.collect()
                for resource in (
                    values, points, velocities, upload_only, rows, matrix, gathered,
                    explicit_domain, explicit_output, local_loop_output, atomic_total,
                    view_source, view, read_only_view, velocities_readonly,
                    wave_values,
                    wave_repeat,
                    intrinsic_source,
                    sine_values,
                    root_values,
                ):
                    if resource is not None:
                        assert resource.closed
                native.cleanup()
            except BaseException:
                if failure is None:
                    failure = traceback.format_exc()
            affine = integrate = gather_rows = declaration = None
            explicit_domain_kernel = local_loops = atomic_reduce = None
            values = points = velocities = upload_only = rows = matrix = gathered = None
            explicit_domain = explicit_output = local_loop_output = atomic_total = None
            view_source = view = read_only_view = velocities_readonly = None
            gc.collect()
        if failure is not None:
            raise RuntimeError(failure)
    print("INFERNUX_PUBLIC_GPU_KERNEL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
