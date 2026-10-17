"""Tests for CPU HIR compilation and adaptive serial/parallel execution."""

from __future__ import annotations

import inspect

import pytest
import numpy as np

import infernux.jit as jit
import infernux._jit_kernels as jit_kernels


@pytest.fixture(autouse=True)
def jit_project_cache(tmp_path):
    from infernux.engine.project_context import using_project_root

    with using_project_root(str(tmp_path)):
        yield


def _positive_value(value):
    return value if value > 0.0 else 0.0


def _carried_branch(source, output, n):
    value = 7
    for i in range(n):
        if source[i] > 0:
            value = source[i]
        output[i] = value
    return output


def _carried_previous(source, output, n):
    value = 7
    for i in range(n):
        output[i] = value
        value = source[i]
    return output


def _last_local(source, output, n):
    value = 7
    for i in range(n):
        value = source[i]
        output[i] = value
    return value


def _last_index(source, output, n):
    i = -1
    for i in range(n):
        output[i] = source[i]
    return i


def _mutated_index(source, output, n):
    for i in range(n):
        i = 0
        output[i] = source[i]
    return output


def _mixed_reduction(source, output, n):
    total = 1.0
    for i in range(n):
        total += source[i]
        total *= 1.01
    return total


def _reset_reduction(source, output, n):
    total = 1
    for i in range(n):
        total = 1
        total += source[i]
    return total


@pytest.mark.parametrize("function", [_carried_branch, _last_local, _mutated_index, _mixed_reduction])
def test_scalar_flow_selects_legal_serial_code_before_any_execution(function):
    compiled = jit.compile(function)
    assert compiled.parallel is compiled.serial
    with pytest.raises(ValueError, match="parallel_policy='required' rejected"):
        jit.compile(function, parallel_policy="required")
    for count in (0, 1, 257):
        source = np.arange(count, dtype=np.int64) - count // 2
        expected = np.full(count, -99, dtype=np.int64)
        actual = expected.copy()
        expected_result = function(source, expected, count)
        result = compiled(source, actual, count)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(result, expected_result)


def test_all_paths_define_private_scalar_can_still_run_parallel():
    @jit.compile(parallel_policy="required")
    def kernel(source, output):
        for i in range(len(source)):
            if source[i] > 0:
                value = source[i]
            else:
                value = -source[i]
            value = value * 2
            output[i] = value

    source = np.arange(257, dtype=np.int64) - 128
    output = np.zeros_like(source)
    assert kernel.parallel is not kernel.serial
    kernel(source, output)
    np.testing.assert_array_equal(output, np.abs(source) * 2)


def test_repeated_conditional_reduction_runs_in_parallel():
    @jit.compile(parallel_policy="required")
    def kernel(source):
        total = 1
        for i in range(len(source)):
            total += source[i]
            if source[i] > 0:
                total += 2
        return total

    assert kernel.parallel is not kernel.serial
    for count in (0, 1, 257):
        source = np.arange(count, dtype=np.int64) - count // 2
        assert kernel(source) == 1 + source.sum() + 2 * np.count_nonzero(source > 0)


class TestPublicJitCompile:
    @pytest.mark.parametrize("auto_parallel", [False, True])
    @pytest.mark.parametrize(
        "source",
        [
            "def kernel(value): return value * FACTOR",
            "def leaf(value): return value * FACTOR\n"
            "def helper(value): return leaf(value)\n"
            "def kernel(value): return helper(value)",
        ],
    )
    def test_republished_module_constants_do_not_reuse_old_machine_code(self, auto_parallel, source):
        def publish(factor):
            namespace = {"__name__": "jit_publication_constants", "FACTOR": factor}
            exec(source, namespace)
            return jit.compile(auto_parallel=auto_parallel)(namespace["kernel"])

        first = publish(2)
        assert first(3) == 6
        second = publish(5)
        assert second(3) == 15
        assert first(3) == 6

    @pytest.mark.parametrize("auto_parallel", [False, True])
    def test_republished_helper_body_selects_a_new_publication(self, auto_parallel):
        def publish(increment):
            namespace = {"__name__": "jit_helper_generation"}
            exec(
                f"def helper(value): return value + {increment}\ndef kernel(value): return helper(value)",
                namespace,
            )
            return jit.compile(auto_parallel=auto_parallel)(namespace["kernel"])

        first = publish(2)
        assert first(3) == 5
        second = publish(7)
        assert second(3) == 10
        # A dependency update publishes new code; it neither mutates nor
        # redirects calls through the old specialization.
        assert first(3) == 5

    @pytest.mark.parametrize("auto_parallel", [False, True])
    def test_republished_array_constant_tracks_data_hidden_by_numpy_repr(self, auto_parallel):
        def publish(middle):
            values = np.zeros(4096, dtype=np.float64)
            values[2048] = middle
            namespace = {"__name__": "jit_array_constants", "VALUES": values}
            exec("def kernel(index): return VALUES[index]", namespace)
            return jit.compile(auto_parallel=auto_parallel)(namespace["kernel"])

        first = publish(2.0)
        assert first(2048) == 2.0
        second = publish(5.0)
        assert second(2048) == 5.0
        assert first(2048) == 2.0

    @pytest.mark.parametrize("auto_parallel", [False, True])
    def test_same_vector_retains_aliasing_between_positional_and_keyword_arguments(self, auto_parallel):
        import infernux as inx

        @jit.compile(auto_parallel=auto_parallel)
        def modify(first, second):
            first.x += 2.0
            second.x *= 3.0
            return first.x

        value = inx.vector3(1.0, 2.0, 3.0)
        assert modify(value, second=value) == pytest.approx(9.0)
        assert value.x == pytest.approx(9.0)

    def test_serial_warmup_does_not_mutate_author_data(self):
        import infernux as inx

        @jit.compile(auto_parallel=False)
        def modify(values):
            values[0] += 4.0

        source = np.zeros(3, dtype=np.float32)
        storage = inx.buffer(shape=3, dtype=np.float32, device="cpu", data=source)
        jit.warmup(modify, source)
        jit.warmup(modify, storage)
        np.testing.assert_array_equal(source, np.zeros(3))
        np.testing.assert_array_equal(storage.numpy(), np.zeros(3))
        modify(storage)
        assert storage.numpy()[0] == pytest.approx(4.0)

    @pytest.mark.parametrize("auto_parallel", [False, True])
    def test_vector_partial_writes_survive_runtime_error_without_replay(self, auto_parallel):
        import infernux as inx

        @jit.compile(auto_parallel=auto_parallel)
        def modify(value):
            value.x += 2.0
            raise ValueError("authored failure")

        value = inx.vector3(1.0, 2.0, 3.0)
        with pytest.raises(ValueError, match="authored failure"):
            modify(value)
        assert value.x == pytest.approx(3.0)

    def test_failed_serial_warmup_preserves_inputs_and_propagates_error(self):
        @jit.compile(auto_parallel=False)
        def fail(values):
            values[0] += 1.0
            raise ValueError("warmup failure")

        values = np.zeros(2, dtype=np.float32)
        with pytest.raises(ValueError, match="warmup failure"):
            jit.warmup(fail, values)
        np.testing.assert_array_equal(values, [0.0, 0.0])

    def test_missing_cpu_runtime_is_an_error_not_a_python_fallback(self, monkeypatch):
        monkeypatch.setattr(jit, "_BACKEND_JIT_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="bundled Numba/llvmlite"):
            jit.compile(lambda value: value)

    def test_cpu_buffer_is_direct_jit_storage_and_gpu_is_explicitly_rejected(self):
        import infernux as inx

        values = inx.buffer(
            shape=4,
            dtype=inx.vector3,
            device="cpu",
            data=np.arange(12, dtype=np.float32).reshape(4, 3),
        )

        @jit.compile(auto_parallel=False)
        def scale(items, factor):
            for index in range(len(items)):
                items[index, 0] *= factor
            return items

        assert scale(values, 2.0) is values
        np.testing.assert_array_equal(
            values.numpy(),
            [[0, 1, 2], [6, 4, 5], [12, 7, 8], [18, 10, 11]],
        )

        gpu = object.__new__(inx.Buffer)
        gpu._device = "gpu"
        gpu._closed = False
        with pytest.raises(RuntimeError, match="GPU buffers require"):
            scale(gpu, 2.0)

    def test_public_engine_vectors_use_typed_cpu_jit_values(self):
        import infernux as inx

        @jit.compile(auto_parallel=False)
        def move(value, amount):
            value.x += amount
            value.y *= 2.0
            return value

        source = inx.vector3(1.0, 2.0, 3.0)
        result = move(source, 4.0)

        assert isinstance(result, inx.vector3)
        assert (result.x, result.y, result.z) == pytest.approx((5.0, 4.0, 3.0))
        assert (source.x, source.y, source.z) == pytest.approx((5.0, 4.0, 3.0))

        @jit.compile(auto_parallel=False)
        def length_squared(value):
            return value.x * value.x + value.y * value.y

        assert length_squared(inx.vector2(3.0, 4.0)) == pytest.approx(25.0)


class TestAutoParallelNjit:
    def test_aliasing_cannot_reuse_an_independent_array_parallel_decision(self, monkeypatch):
        from infernux.jit_runtime import StaticCostDecision

        monkeypatch.setattr(
            jit_kernels,
            "static_cost_decision",
            lambda *args, **kwargs: StaticCostDecision("parallel", "high", 10_000_000, "test workload"),
        )
        calls = []
        serial = lambda *args: calls.append("serial")
        parallel = lambda *args: calls.append("parallel")
        dispatcher = jit_kernels._build_auto_parallel_dispatcher(serial, serial, parallel)
        first, second = np.zeros(16), np.zeros(16)
        dispatcher(first, second)
        dispatcher(first, first)
        dispatcher(first, first.view())
        dispatcher(first, second)
        assert calls == ["parallel", "serial", "serial", "parallel"]
        dispatcher._infernux_warmup(first, first)
        assert calls[-1] == "serial" and "alias" in dispatcher.last_diagnostic

        required = jit_kernels._build_auto_parallel_dispatcher(serial, serial, parallel, parallel_policy="required")
        with pytest.raises(ValueError, match="aliased"):
            required(first, first)
        with pytest.raises(ValueError, match="aliased"):
            required._infernux_warmup(first, first)
        assert len(calls) == 5

    @staticmethod
    def _fake_numba_njit(*factory_args, **factory_kwargs):
        def _compile(fn, *, mode: str):
            state = {"calls": 0}

            def _compiled(*args, **kwargs):
                state["calls"] += 1
                if mode == "parallel" and kwargs.pop("_force_parallel_fail", False):
                    raise RuntimeError("parallel failed")
                value = fn(*args, **kwargs)
                if mode == "parallel" and state["calls"] > 1:
                    value += 0
                elif mode == "serial" and state["calls"] > 1:
                    for _ in range(2000):
                        value += 0
                return value

            _compiled.mode = mode
            _compiled.state = state
            return _compiled

        mode = "parallel" if factory_kwargs.get("parallel") else "serial"

        if factory_args and callable(factory_args[0]) and len(factory_args) == 1:
            return _compile(factory_args[0], mode=mode)

        def _decorator(fn):
            return _compile(fn, mode=mode)

        return _decorator

    def test_auto_parallel_builds_dual_variants(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)

        @jit_kernels.njit(cache=True, auto_parallel=True)
        def burn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        assert burn.py(5) == 10
        assert getattr(burn, "auto_parallel", False) is True
        assert burn.serial.mode == "serial"
        assert burn.parallel.mode == "parallel"
        assert burn.selected_mode == "serial"
        assert "eligible" in burn.last_diagnostic

    def test_parallel_runtime_failure_is_not_replayed_through_serial(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)

        @jit_kernels.njit(
            cache=True,
            auto_parallel=True,
            parallel_policy="required",
        )
        def burn(n: int, _force_parallel_fail: bool = False) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        with pytest.raises(RuntimeError, match="parallel failed"):
            burn(5, _force_parallel_fail=True)
        assert burn.selected_mode == "parallel"
        assert burn.serial.state["calls"] == 0

    @pytest.mark.skipif(not jit_kernels.JIT_AVAILABLE, reason="Numba is unavailable")
    def test_real_numba_hir_kernel_is_correct_and_warmup_isolated(self):
        @jit_kernels.njit(auto_parallel=True)
        def scale(out, values):
            for i in range(values.shape[0]):
                out[i] = values[i] * 2.0

        source = np.arange(4096, dtype=np.float32)
        output = np.zeros_like(source)
        jit_kernels.warmup(scale, output, source)
        # Warmup must use clones and leave live gameplay buffers untouched.
        assert np.count_nonzero(output) == 0
        scale(output, source)
        np.testing.assert_allclose(output, source * 2.0)
        assert scale.parallel is not scale.serial
        assert len(scale.decisions) == 1

    @pytest.mark.skipif(not jit_kernels.JIT_AVAILABLE, reason="Numba is unavailable")
    def test_real_numba_unsafe_dependency_stays_serial(self):
        @jit_kernels.njit(auto_parallel=True)
        def prefix(values):
            for i in range(1, values.shape[0]):
                values[i] = values[i - 1] + 1

        values = np.zeros(32, dtype=np.int64)
        assert prefix.parallel is prefix.serial
        assert "depends on another loop iteration" in prefix.last_diagnostic
        prefix(values)
        np.testing.assert_array_equal(values, np.arange(32, dtype=np.int64))

    @pytest.mark.skipif(not jit_kernels.JIT_AVAILABLE, reason="Numba is unavailable")
    def test_real_numba_strided_range_is_validated_before_selection(self):
        @jit_kernels.njit(auto_parallel=True)
        def fill_odd(values):
            for i in range(1, values.shape[0], 2):
                values[i] = i

        values = np.zeros(64, dtype=np.int64)
        jit_kernels.warmup(fill_odd, values)
        assert fill_odd.parallel is fill_odd.serial
        assert "step size of 1" in fill_odd.last_diagnostic
        fill_odd(values)
        np.testing.assert_array_equal(values[1::2], np.arange(1, 64, 2))
