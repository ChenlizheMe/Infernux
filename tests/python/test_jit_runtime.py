from __future__ import annotations

import numpy as np

from infernux.jit_runtime import (
    BoundedLRU,
    calls_equivalent,
    clone_call_arguments,
    compiler_fingerprint,
    runtime_signature,
    static_cost_decision,
)


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
