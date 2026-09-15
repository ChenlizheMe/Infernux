"""Tests for CPU HIR compilation and adaptive serial/parallel execution."""

from __future__ import annotations

import pytest
import numpy as np

import Infernux.jit as jit
import Infernux._jit_kernels as jit_kernels


def _positive_value(value):
    return value if value > 0.0 else 0.0


class TestPublicJitCompile:
    def test_legacy_numba_decorator_is_not_public(self):
        import Infernux as inx

        assert not hasattr(jit, "njit")
        assert not hasattr(jit, "prange")
        assert not hasattr(inx, "njit")

    def test_defaults_to_engine_auto_parallel_policy(self, monkeypatch):
        calls = []

        def fake_njit(*args, **kwargs):
            calls.append((args, kwargs))
            if args:
                return args[0]
            return lambda fn: fn

        monkeypatch.setattr(jit, "_njit", fake_njit)

        @jit.compile
        def direct(value):
            return value + 1

        @jit.compile(cache=True)
        def configured(value):
            return value * 2

        assert direct(2) == 3 and configured(3) == 6
        assert calls[0][1] == {"auto_parallel": True}
        assert calls[1][1] == {"cache": True, "auto_parallel": True}

    def test_rejects_non_callable_direct_argument(self):
        with pytest.raises(TypeError, match="expects a callable"):
            jit.compile(3)

    def test_missing_cpu_runtime_is_an_error_not_a_python_fallback(self, monkeypatch):
        monkeypatch.setattr(jit, "JIT_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="bundled Numba/llvmlite"):
            jit.compile(lambda value: value)

    def test_cpu_buffer_is_direct_jit_storage_and_gpu_is_explicitly_rejected(self):
        import Infernux as inx

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
        import Infernux as inx

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

    def test_public_cpu_jit_freezes_numeric_input_and_output_surface(self):
        @jit.compile(auto_parallel=False)
        def reduce_values(values):
            total = 0.0
            for index in range(len(values)):
                total += _positive_value(values[index])
            return total

        values = np.asarray([-2.0, 1.5, 3.25], dtype=np.float32)
        assert reduce_values(values) == pytest.approx(4.75)

        with pytest.raises(TypeError, match="numeric scalars"):
            reduce_values(object())
        with pytest.raises(TypeError, match="object arrays"):
            reduce_values(np.asarray([object()], dtype=object))

        @jit.compile(auto_parallel=False)
        def typed_list_result(value):
            return [value]

        with pytest.raises(TypeError, match="must return"):
            typed_list_result(3)


class TestAutoParallelNjit:
    def test_aliasing_cannot_reuse_an_independent_array_parallel_decision(self, monkeypatch):
        from Infernux.jit_runtime import StaticCostDecision

        monkeypatch.setattr(jit_kernels, "static_cost_decision", lambda *args, **kwargs:
                            StaticCostDecision("parallel", "high", 10_000_000, "test workload"))
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

    def test_no_jit_build_exposes_stable_serial_metadata(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", False)

        @jit_kernels.njit(auto_parallel=True)
        def kernel(value):
            return value + 1

        assert kernel(4) == 5
        assert kernel.selected_mode == "serial"
        assert kernel.serial is kernel.parallel is kernel
        assert "unavailable" in kernel.last_diagnostic

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
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
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

    def test_warmup_can_pin_serial_variant(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)
        monkeypatch.setattr(
            jit_kernels,
            "_benchmark_callable",
            lambda fn, *_args, **_kwargs: 1.0 if getattr(fn, "mode", "") == "serial" else 2.0,
        )

        @jit_kernels.njit(cache=True, auto_parallel=True)
        def burn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        # Keep the input in the static model's gray zone so warmup timing,
        # rather than a decisive AST estimate, owns the final choice.
        jit_kernels.warmup(burn, 100_000)
        assert burn.selected_mode == "serial"
        assert burn(5) == 10

    def test_warmup_can_pin_parallel_after_compile_cost(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)
        monkeypatch.setattr(
            jit_kernels,
            "_benchmark_callable",
            lambda fn, *_args, **_kwargs: 2.0 if getattr(fn, "mode", "") == "serial" else 1.0,
        )

        @jit_kernels.njit(cache=True, auto_parallel=True)
        def burn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        jit_kernels.warmup(burn, 100_000)
        assert burn.selected_mode == "parallel"
        assert burn(5) == 10

    def test_parallel_runtime_failure_is_not_replayed_through_serial(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
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

    def test_try_build_auto_parallel_variant_rewrites_range_to_prange(self, monkeypatch):
        used = {"prange": False}

        def _fake_prange(*args):
            used["prange"] = True
            return range(*args)

        monkeypatch.setattr(jit_kernels, "prange", _fake_prange)

        def burn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        rewritten = jit_kernels._try_build_auto_parallel_variant(burn)
        assert rewritten is not None
        assert rewritten(5) == 10
        assert used["prange"] is True

    def test_public_compile_embedded_source_supplies_parallel_impl_without_sidecar(self):
        source = (
            "from Infernux import jit\n"
            "@jit.compile\n"
            "def burn(n):\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        total += i\n"
            "    return total\n"
        )
        embedded = jit_kernels.build_auto_parallel_embedded_source(source)
        assert embedded is not None
        assert "__infernux_parallel_burn_" in embedded
        assert "_parallel_impl=" in embedded
        assert "__infernux_prange" in embedded

    @pytest.mark.parametrize(("imports", "decorator"), [
        ("import infernux as inx", "inx.jit.compile"),
        ("import Infernux.jit as cpu", "cpu.compile"),
        ("from Infernux import jit as cpu", "cpu.compile"),
        ("from Infernux.jit import compile as optimize", "optimize"),
    ])
    def test_public_jit_compile_import_forms_embed_parallel_impl(self, imports, decorator):
        source = (
            f"{imports}\n@{decorator}(cache=True)\n"
            "def fill(values):\n"
            "    for i in range(len(values)):\n"
            "        values[i] = i * 3\n"
        )
        assert jit_kernels.auto_parallel_declarations(source) == (("fill", "auto"),)
        embedded = jit_kernels.build_auto_parallel_embedded_source(source)
        assert embedded is not None
        namespace = {}
        exec(compile(embedded, "<cooked-jit-compile>", "exec"), namespace)
        values = np.zeros(8, dtype=np.int32)
        namespace["fill"](values)
        np.testing.assert_array_equal(values, np.arange(8) * 3)
        assert namespace["fill"].compiler_fingerprint == (
            namespace["__infernux_jit_manifest__"]["fill"]["hir_fingerprint"]
        )

    def test_bare_public_jit_compile_decorator_embeds_parallel_impl(self):
        source = (
            "from Infernux import jit\n"
            "@jit.compile\n"
            "def fill(values):\n"
            "    for i in range(len(values)):\n"
            "        values[i] = i + 1\n"
        )
        embedded = jit_kernels.build_auto_parallel_embedded_source(source)
        assert embedded is not None
        namespace = {}
        exec(compile(embedded, "<cooked-bare-jit-compile>", "exec"), namespace)
        values = np.zeros(5, dtype=np.int32)
        namespace["fill"](values)
        np.testing.assert_array_equal(values, np.arange(5) + 1)

    def test_public_jit_compile_can_explicitly_disable_auto_parallel_embedding(self):
        source = (
            "from Infernux import jit\n"
            "@jit.compile(auto_parallel=False)\n"
            "def fill(values):\n"
            "    for i in range(len(values)):\n"
            "        values[i] = i\n"
        )
        assert jit_kernels.auto_parallel_declarations(source) == ()
        assert jit_kernels.build_auto_parallel_embedded_source(source) is None

    def test_try_build_auto_parallel_variant_rewrites_mult_reduction(self, monkeypatch):
        used = {"prange": False}

        def _fake_prange(*args):
            used["prange"] = True
            return range(*args)

        monkeypatch.setattr(jit_kernels, "prange", _fake_prange)

        def product(n: int) -> int:
            acc = 1
            for i in range(1, n + 1):
                acc *= i
            return acc

        rewritten = jit_kernels._try_build_auto_parallel_variant(product)
        assert rewritten is not None
        assert rewritten(5) == 120
        assert used["prange"] is True

    def test_try_build_auto_parallel_variant_rewrites_indexed_array_store(self, monkeypatch):
        used = {"prange": False}

        def _fake_prange(*args):
            used["prange"] = True
            return range(*args)

        monkeypatch.setattr(jit_kernels, "prange", _fake_prange)

        def fill(arr):
            for i in range(len(arr)):
                arr[i] = i * 2

        rewritten = jit_kernels._try_build_auto_parallel_variant(fill)
        assert rewritten is not None
        data = [0] * 5
        rewritten(data)
        assert data == [0, 2, 4, 6, 8]
        assert used["prange"] is True

    def test_try_build_auto_parallel_variant_allows_continue(self, monkeypatch):
        used = {"prange": False}

        def _fake_prange(*args):
            used["prange"] = True
            return range(*args)

        monkeypatch.setattr(jit_kernels, "prange", _fake_prange)

        def evens(n: int) -> int:
            total = 0
            for i in range(n):
                if i % 2 != 0:
                    continue
                total += i
            return total

        rewritten = jit_kernels._try_build_auto_parallel_variant(evens)
        assert rewritten is not None
        assert rewritten(6) == 6  # 0 + 2 + 4
        assert used["prange"] is True

    def test_build_embedded_source_handles_mult_reduction(self):
        source = (
            "from Infernux import jit\n"
            "@jit.compile\n"
            "def product(n):\n"
            "    acc = 1\n"
            "    for i in range(1, n + 1):\n"
            "        acc *= i\n"
            "    return acc\n"
        )
        embedded = jit_kernels.build_auto_parallel_embedded_source(source)
        assert embedded is not None
        assert "__infernux_prange" in embedded

    def test_build_embedded_source_handles_indexed_store(self):
        source = (
            "from Infernux import jit\n"
            "@jit.compile\n"
            "def fill(arr):\n"
            "    for i in range(len(arr)):\n"
            "        arr[i] = i * 2\n"
        )
        embedded = jit_kernels.build_auto_parallel_embedded_source(source)
        assert embedded is not None
        assert "__infernux_prange" in embedded
        assert "__infernux_jit_manifest__" in embedded
        assert "_parallel_fingerprint" in embedded

        namespace = {}
        exec(embedded, namespace)
        entry = namespace["__infernux_jit_manifest__"]["fill"]
        assert len(entry["hir_fingerprint"]) == 64
        assert entry["loop_ids"]
        assert entry["operation_cost"] > 0
        assert namespace["fill"].compiler_fingerprint == entry["hir_fingerprint"]
        assert namespace["fill"].static_operation_cost == entry["operation_cost"]

    def test_required_policy_rejects_loop_carried_dependency(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)

        with pytest.raises(ValueError, match="parallel_policy='required' rejected"):
            @jit_kernels.njit(auto_parallel=True, parallel_policy="required")
            def prefix(values):
                for i in range(1, len(values)):
                    values[i] = values[i - 1] + 1

    def test_warmup_decisions_are_per_shape_bucket(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)
        monkeypatch.setattr(
            jit_kernels,
            "_benchmark_callable",
            lambda fn, *_args, **_kwargs: 2.0 if getattr(fn, "mode", "") == "serial" else 1.0,
        )

        @jit_kernels.njit(auto_parallel=True)
        def fill(values):
            for i in range(len(values)):
                values[i] = i
            return len(values)

        small = [0] * 16
        large = [0] * 100_000
        jit_kernels.warmup(fill, small)
        assert fill.selected_mode == "parallel"
        assert fill(small) == len(small)
        # Lists do not carry ndarray shape metadata, so integer array tests
        # use NumPy below to exercise distinct trip-count buckets.
        import numpy as np

        small_array = np.zeros(16, dtype=np.int64)
        large_array = np.zeros(100_000, dtype=np.int64)
        jit_kernels.warmup(fill, small_array)
        assert fill.selected_mode == "serial"
        assert "static HIR cost" in fill.last_diagnostic
        jit_kernels.warmup(fill, large_array)
        assert fill.selected_mode == "parallel"
        fill(large_array)
        assert fill.selected_mode == "parallel"

    def test_static_decision_skips_repeated_timing_for_decisive_shape(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)
        monkeypatch.setattr(
            jit_kernels,
            "_benchmark_callable",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not benchmark")),
        )

        @jit_kernels.njit(auto_parallel=True)
        def fill(values):
            for i in range(values.shape[0]):
                values[i] = i

        values = np.zeros(10_000_000, dtype=np.float32)
        jit_kernels.warmup(fill, values)
        assert fill.selected_mode == "parallel"
        assert "static HIR cost selected parallel" in fill.last_diagnostic

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
