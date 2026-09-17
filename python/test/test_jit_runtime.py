from __future__ import annotations

import numpy as np

from Infernux.jit_runtime import (
    BoundedLRU,
    calls_equivalent,
    clone_call_arguments,
    compiler_fingerprint,
    runtime_signature,
    static_cost_decision,
)


def test_bounded_lru_evicts_oldest_and_promotes_reads():
    cache = BoundedLRU[str, int](2)
    cache["a"] = 1
    cache["b"] = 2
    assert cache.get("a") == 1
    cache["c"] = 3
    assert "a" in cache
    assert "b" not in cache
    assert cache.get("c") == 3


def test_preparation_does_not_hold_arrays_until_cyclic_gc():
    import gc
    import weakref
    enabled = gc.isenabled()
    gc.disable()
    try:
        source = np.arange(32, dtype=np.float32)
        original_ref = weakref.ref(source)
        arguments, _ = clone_call_arguments((source,), {})
        clone_ref = weakref.ref(arguments[0])
        del source, arguments
        assert original_ref() is None
        assert clone_ref() is None
    finally:
        if enabled:
            gc.enable()


def test_compiler_fingerprint_tracks_constants_defaults_and_dependencies():
    def helper(value):
        return value + 1

    def kernel(value=3):
        return helper(value) + 7

    before = compiler_fingerprint(kernel, {"fastmath": False})

    def replacement(value):
        return value + 2

    helper.__code__ = replacement.__code__
    after_dependency = compiler_fingerprint(kernel, {"fastmath": False})
    after_option = compiler_fingerprint(kernel, {"fastmath": True})

    assert before != after_dependency
    assert after_dependency != after_option


def test_compiler_fingerprint_tracks_cpu_feature_configuration(monkeypatch):
    def kernel(value):
        return value + 1

    monkeypatch.setenv("NUMBA_CPU_FEATURES", "+sse2")
    first = compiler_fingerprint(kernel)
    monkeypatch.setenv("NUMBA_CPU_FEATURES", "+avx2")
    second = compiler_fingerprint(kernel)

    assert first != second


def test_compiler_fingerprint_tracks_transitive_helper_closure_values():
    def make_leaf(offset):
        def leaf(value):
            return value + offset
        return leaf

    leaf = make_leaf(2)

    def helper(value):
        return leaf(value)

    def kernel(value):
        return helper(value)

    first = compiler_fingerprint(kernel)
    leaf = make_leaf(9)
    assert compiler_fingerprint(kernel) != first


def test_compiler_fingerprint_handles_recursive_helpers_without_identity_keys():
    source = "def left(x): return right(x - 1) if x > 0 else FACTOR\n" \
             "def right(x): return left(x)\n"

    def publish(factor):
        namespace = {"__name__": "recursive_jit_dependencies", "FACTOR": factor}
        exec(source, namespace)
        return namespace["left"]

    assert compiler_fingerprint(publish(2)) == compiler_fingerprint(publish(2))
    assert compiler_fingerprint(publish(2)) != compiler_fingerprint(publish(7))


def test_compiler_fingerprint_ignores_unreferenced_module_values():
    def publish(unused):
        namespace = {"__name__": "jit_relevant_constants", "FACTOR": 2, "UNUSED": unused}
        exec("def kernel(value): return value * FACTOR", namespace)
        return namespace["kernel"]

    assert compiler_fingerprint(publish(3)) == compiler_fingerprint(publish(9))


def test_compiler_fingerprint_tracks_helper_defaults():
    def helper(value, offset=2):
        return value + offset

    def kernel(value):
        return helper(value)

    first = compiler_fingerprint(kernel)
    helper.__defaults__ = (5,)
    assert compiler_fingerprint(kernel) != first


def test_compiler_fingerprint_tracks_only_referenced_module_attributes():
    from types import ModuleType

    settings = ModuleType("authored_settings")
    settings.nested = ModuleType("authored_settings.nested")
    settings.nested.scale = 2
    settings.unused = 1
    namespace = {"__name__": "authored_kernel", "settings": settings}
    exec("def kernel(value): return value * settings.nested.scale", namespace)
    function = namespace["kernel"]
    before = compiler_fingerprint(function)
    settings.unused = 99
    assert compiler_fingerprint(function) == before
    settings.nested.scale = 5
    assert compiler_fingerprint(function) != before


def test_compiler_fingerprint_tracks_module_captured_in_closure():
    from types import ModuleType

    settings = ModuleType("closed_settings")
    settings.scale = 2

    def kernel(value):
        return value * settings.scale

    before = compiler_fingerprint(kernel)
    settings.scale = 5
    assert compiler_fingerprint(kernel) != before


def test_runtime_signature_separates_shape_dtype_layout_and_threads():
    small = np.zeros((16, 2), dtype=np.float32)
    large = np.zeros((100_000, 2), dtype=np.float32)
    f_order = np.asfortranarray(small)

    small_key = runtime_signature((small,), {}, thread_count=4)
    assert small_key != runtime_signature((large,), {}, thread_count=4)
    assert small_key != runtime_signature((small.astype(np.float64),), {}, thread_count=4)
    assert small_key != runtime_signature((f_order,), {}, thread_count=4)
    assert small_key != runtime_signature((small,), {}, thread_count=8)


def test_static_cost_model_prefers_ast_work_before_timing():
    small = np.zeros((32, 3), dtype=np.float32)
    large = np.zeros((10_000_000, 3), dtype=np.float32)

    small_decision = static_cost_decision(
        (small,), {}, operation_cost=12, thread_count=8
    )
    large_decision = static_cost_decision(
        (large,), {}, operation_cost=12, thread_count=8
    )

    assert (small_decision.mode, small_decision.confidence) == ("serial", "high")
    assert (large_decision.mode, large_decision.confidence) == ("parallel", "high")
    assert large_decision.work_units > small_decision.work_units


def test_static_cost_model_leaves_unknown_trip_count_for_measurement():
    decision = static_cost_decision(([0] * 16,), {}, operation_cost=12, thread_count=8)
    assert (decision.mode, decision.confidence) == ("serial", "gray")


def test_clone_and_equivalence_include_mutated_array_arguments():
    values = np.arange(8, dtype=np.float32)
    left_args, left_kwargs = clone_call_arguments((values,), {"scale": 2.0})
    right_args, right_kwargs = clone_call_arguments((values,), {"scale": 2.0})

    left_args[0][:] *= 2
    right_args[0][:] *= 2
    assert calls_equivalent(None, left_args, left_kwargs, None, right_args, right_kwargs)
    assert np.array_equal(values, np.arange(8, dtype=np.float32))

    right_args[0][3] = -1
    assert not calls_equivalent(None, left_args, left_kwargs, None, right_args, right_kwargs)


def test_clone_rejects_unknown_mutable_object():
    class Unknown:
        pass

    try:
        clone_call_arguments((Unknown(),), {})
    except TypeError as exc:
        assert "cannot isolate" in str(exc)
    else:
        raise AssertionError("unknown mutable object must not be benchmarked")


def test_clone_preserves_aliases_across_positional_keyword_and_nested_arguments():
    values = np.arange(8, dtype=np.int32)
    nested = [values, {"shared": values}]
    args, kwargs = clone_call_arguments((values, nested), {"again": values, "nested": nested})
    assert args[0] is kwargs["again"] is args[1][0] is args[1][1]["shared"]
    assert args[1] is kwargs["nested"] and args[1] is not nested
    args[0][:] += 7
    np.testing.assert_array_equal(values, np.arange(8))
    np.testing.assert_array_equal(kwargs["again"], np.arange(8) + 7)


def test_clone_preserves_overlapping_strided_and_reinterpreted_views():
    values = np.arange(16, dtype=np.int32)
    views = (values[2:10], values[4:12], values[::-2], values.view(np.uint8))
    args, _ = clone_call_arguments(views, {})
    assert np.shares_memory(args[0], args[1])
    assert np.shares_memory(args[1], args[2])
    assert np.shares_memory(args[0], args[3])
    assert args[2].strides == views[2].strides
    args[0][2] = 777
    assert args[1][0] == 777
    assert args[3].view(np.int32)[4] == 777
    np.testing.assert_array_equal(values, np.arange(16))


def test_clone_preserves_fortran_layout_and_readonly_views():
    values = np.asfortranarray(np.arange(12).reshape(3, 4))
    view = values[:, 1:]
    view.flags.writeable = False
    args, _ = clone_call_arguments((values, view), {})
    assert args[0].flags.f_contiguous and args[1].strides == view.strides
    assert np.shares_memory(*args) and not args[1].flags.writeable
    np.testing.assert_array_equal(args[1], view)


def test_clone_rejects_shared_external_storage_it_cannot_isolate():
    import pytest

    storage = bytearray(32)
    first = np.frombuffer(storage, dtype=np.int32)
    second = np.frombuffer(storage, dtype=np.int32, offset=4)
    with pytest.raises(TypeError, match="independent storage owners"):
        clone_call_arguments((first, second), {})
    with pytest.raises(TypeError, match="object-array"):
        clone_call_arguments((np.array([{}], dtype=object),), {})
