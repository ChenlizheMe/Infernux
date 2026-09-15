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

import Infernux as inx


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
        wave_values = None
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
            from Infernux._compiler.taichi import frontend as compiler_frontend
            compiler_vendor = compiler_frontend._load_vendor()
            compiler_module = sys.modules[compiler_vendor.__name__ + '.lang.kernel_impl']
            buffer_type = compiler_vendor.types.external_buffer
            ensure_compiled = compiler_module.Kernel.ensure_compiled
            compiler_context = compiler_vendor.lang.impl.get_runtime()
            observed_signatures = []
            compiler_references = []

            def compile_from_metadata(compiler, *arguments):
                observed_signatures.append(arguments)
                for argument, declaration in zip(arguments, compiler.arguments):
                    if isinstance(declaration.annotation, buffer_type):
                        assert argument is declaration.annotation, 'Buffer schema was replaced by runtime data'
                    else:
                        assert argument is None, 'Scalar values must remain runtime arguments'
                key = ensure_compiled(compiler, *arguments)
                compiler_references.append((weakref.ref(compiler), weakref.ref(compiler.compiled_kernels[key])))
                assert not hasattr(compiler, 'grad')
                return key

            compiler_module.Kernel.ensure_compiled = compile_from_metadata
            globals()['_compiler_lifetime_probe'] = ModuleType('compiler_lifetime_probe')
            globals_reference = weakref.ref(globals()['_compiler_lifetime_probe'])
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                try:
                    inx.compute.prepare(affine, params=(values, 3, 11))
                finally:
                    compiler_module.Kernel.ensure_compiled = ensure_compiled
                    del globals()['_compiler_lifetime_probe']
                assert globals_reference() is None, 'Generated compiler namespace retained project globals'
            finally:
                if gc_was_enabled:
                    gc.enable()
            assert len(observed_signatures) == 1
            assert all(reference() is None for pair in compiler_references for reference in pair), \
                'Completed compilation still retains its Python/native kernel'
            assert not hasattr(compiler_context, 'kernels')
            assert not any(name.startswith('<infernux-kernel:') for name in linecache.cache)
            assert not hasattr(compiler_module, 'data_oriented')
            assert not hasattr(compiler_module, '_inside_class')
            @inx.compute.kernel
            def invalid_lowering(payload):
                i = inx.compute.index(payload)
                payload[i] = undefined_compiler_value

            try:
                inx.compute.prepare(invalid_lowering, params=(values,))
            except inx.compute.KernelCompilationError as exception:
                assert 'undefined_compiler_value' in str(exception)
                assert 'payload[i] = undefined_compiler_value' in str(exception)
            else:
                raise AssertionError('Invalid compiler AST unexpectedly accepted')
            assert not any(name.startswith('<infernux-kernel:') for name in linecache.cache)
            assert compiler_context.current_kernel is None
            assert compiler_context.compiling_callable is None
            assert compiler_context.inside_kernel is False
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
            from Infernux._compiler.taichi import frontend as compiler_frontend
            affine._release_engine_resources()
            load_vendor = compiler_frontend._load_vendor
            compiler_frontend._load_vendor = lambda: (_ for _ in ()).throw(
                AssertionError("cached GPU artifact tried to initialize the compiler")
            )
            try:
                inx.compute.prepare(affine, params=(values, 3, 11))
            finally:
                compiler_frontend._load_vendor = load_vendor
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
            # domain.  A larger output first in the argument list also proves
            # that launch never guesses its domain from argument order/size.
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

            inx.compute.launch(explicit_domain_kernel, params=(explicit_output, explicit_domain, 4))
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

            inx.compute.launch(_helper_kernel, params=(explicit_domain, local_loop_output, 3, -2))
            np.testing.assert_array_equal(
                local_loop_output.get_data().numpy(),
                np.arange(17, dtype=np.int32) * 3 - 2,
            )
            inx.compute.launch(_nested_helper_kernel, params=(explicit_domain, local_loop_output))
            np.testing.assert_array_equal(
                local_loop_output.get_data().numpy(),
                np.abs(np.arange(17, dtype=np.int32) * -2 + 4) + 3,
            )
            # Intrinsics in helpers must resolve to the same private compiler,
            # including when a public Taichi namespace is already occupied.
            wave_source = np.linspace(-2.0, 2.0, 33, dtype=np.float32)
            wave_values = inx.buffer(shape=33, dtype=np.float32, device="gpu", data=wave_source)
            inx.compute.launch(_wave_kernel, params=(wave_values,))
            np.testing.assert_allclose(
                wave_values.get_data().numpy(),
                np.sqrt(wave_source * wave_source + 1.0) + np.sin(wave_source),
                rtol=2e-6, atol=2e-6,
            )
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
            assert not statistics.gpu_profile_available
            assert statistics.gpu_time_ms is None

            inx.compute.set_profiling_enabled(True)
            inx.compute.launch(affine, params=(rows, 1, 0))
            rows.get_data()
            profiled = inx.compute.statistics()
            assert profiled.gpu_profile_available
            assert profiled.gpu_profile_serial > 0
            assert profiled.gpu_time_ms is not None and profiled.gpu_time_ms >= 0.0
            inx.compute.set_profiling_enabled(False)
            from Infernux._compiler.taichi import frontend as compiler_frontend
            compiler_native = compiler_frontend.load_native()
            assert not hasattr(compiler_native, "_attach_engine_compute_host")
            assert not hasattr(compiler_native, "_EngineComputeDevice")
            assert not hasattr(compiler_native, "AotModuleBuilder")
            assert not hasattr(compiler_native, "GraphBuilder")
            assert not hasattr(compiler_native, "LoadedAotModule")
            assert not hasattr(compiler_native, "SparseMatrix")
            assert not hasattr(compiler_native, "SparseMatrixBuilder")
            assert not hasattr(compiler_native, "SparseSolver")
            assert not hasattr(compiler_native, "KernelProfilerQueryResult")
            assert not hasattr(compiler_native, "KernelProfileTracedRecord")
            assert not hasattr(compiler_native.Program, "create_ndarray")
            assert not hasattr(compiler_native.Program, "create_texture")
            assert not hasattr(compiler_native.Program, "launch_kernel")
            assert not hasattr(compiler_native.Program, "materialize_runtime")
            assert not hasattr(compiler_native, "Ndarray")
            assert not hasattr(compiler_native, "ArgPack")
            assert not hasattr(compiler_native, "Texture")
            assert not hasattr(compiler_native, "SNodeTree")
            assert not hasattr(compiler_native, "SNodeRegistry")
            assert not hasattr(compiler_native, "SNode")
            assert not hasattr(compiler_native, "Axis")
            assert not hasattr(compiler_native, "Mesh")
            assert not hasattr(compiler_native, "MeshPtr")
            assert not hasattr(compiler_native, "finalize_snode_tree")
            assert not hasattr(compiler_native, "trigger_crash")
            assert not hasattr(compiler_native, "test_threading")
            prefix = compiler_frontend._VENDOR_NAME
            assert compiler_frontend._vendor.__name__ == prefix
            private_modules = {"taichi" + name[len(prefix):] for name in sys.modules
                               if name == prefix or name.startswith(prefix + ".")}
            assert "taichi.types.quant" not in private_modules
            assert not any(name == "taichi._snode" or name.startswith("taichi._snode.") for name in private_modules)
            assert not any(name == "taichi.lang.simt" or name.startswith("taichi.lang.simt.") for name in private_modules)
            assert "taichi.lang.mesh" not in private_modules
            assert "taichi.lang.snode" not in private_modules
            # Taichi field/ndarray modules are gone.  The one private marker
            # module has no storage API; ownership remains with inx.buffer.
            assert "taichi.lang._ndarray" not in private_modules
            assert "taichi.lang.field" not in private_modules
            assert "taichi.types.ndarray_type" not in private_modules
            assert "taichi.lang._texture" not in private_modules
            assert "taichi.types.texture_type" not in private_modules
            assert prefix + ".lang._compiler_types" not in sys.modules
            assert prefix + ".lang.runtime_ops" not in sys.modules
            matrix_types = sys.modules[prefix + ".lang.matrix"]
            assert not hasattr(matrix_types, "MatrixNdarray")
            assert not hasattr(matrix_types, "VectorNdarray")
            for numeric_type in (matrix_types.Matrix, matrix_types.Vector, matrix_types.MatrixType, matrix_types.VectorType):
                assert not hasattr(numeric_type, "ndarray")
                assert not hasattr(numeric_type, "field")
            struct_types = sys.modules[prefix + ".lang.struct"]
            assert not hasattr(struct_types, "StructField")
            assert not hasattr(struct_types.Struct, "field")
            assert not hasattr(struct_types.StructType, "field")
            assert not hasattr(matrix_types, "MatrixField")
            compiler_impl = compiler_frontend._vendor.lang.impl
            for name in ("ndarray", "field", "root", "FieldsBuilder", "create_field_member", "axes", "Axis"):
                assert not hasattr(compiler_impl, name)
            for name in ("materialize", "sync", "materialized", "compiler_only", "global_vars",
                         "unfinalized_fields_builder", "_finalize_root_fb_for_aot"):
                assert not hasattr(compiler_impl.get_runtime(), name)
            # Small numeric values remain compiler expressions, not storage
            # proxies. Exercise their host construction and mutation as well.
            vector_value = matrix_types.Vector([1.0, 2.0, 3.0])
            vector_value[1] = 7.0
            assert vector_value.to_list() == [1.0, 7.0, 3.0]
            vector_value._set_entries([4.0, 5.0, 6.0])
            np.testing.assert_array_equal(vector_value.to_numpy(), [4.0, 5.0, 6.0])
            np.testing.assert_allclose(vector_value.norm(), np.sqrt(77.0))
            matrix_value = matrix_types.Matrix([[1, 2], [3, 4]])
            matrix_value[0, 1] = 9
            assert matrix_value.to_list() == [[1, 9], [3, 4]]
            matrix_value._set_entries([[5, 6], [7, 8]])
            np.testing.assert_array_equal(matrix_value.to_numpy(), [[5, 6], [7, 8]])
            np.testing.assert_array_equal(matrix_value.transpose().to_numpy(), [[5, 7], [6, 8]])
            np.testing.assert_allclose(
                matrix_value.inverse().to_numpy(), np.linalg.inv([[5.0, 6.0], [7.0, 8.0]])
            )
            assert matrix_types.Matrix([[], []]).to_list() == [[], []]
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
