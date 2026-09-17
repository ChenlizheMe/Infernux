"""Parallel storage legality through the real CPU backend and cooked code."""

import inspect

import numpy as np
import pytest

from Infernux import jit
from Infernux import _jit_kernels as kernels
from Infernux.jit_runtime import (
    StaticCostDecision, array_arguments_alias, clone_call_arguments,
)


def _shared_update(first, second):
    for i in range(len(first)):
        first[i] += 1
        second[i] += first[i]


def _increment(values):
    for i in range(len(values)):
        values[i] += 1


@pytest.mark.parametrize("view", ["same", "view", "reverse", "strided"])
def test_same_layout_shared_arguments_run_real_parallel_kernel(view):
    storage = np.arange(8192, dtype=np.float64)
    first = {"same": storage, "view": storage, "reverse": storage[::-1], "strided": storage[::2]}[view]
    second = first if view == "same" else first.view()
    expected = 2 * (first.copy() + 1)
    compiled = jit.compile(_shared_update, parallel_policy="required")
    jit.warmup(compiled, first, second=second)
    prepared_signatures = tuple(compiled.parallel.signatures)
    np.testing.assert_array_equal(storage, np.arange(8192))
    compiled(first, second=second)
    np.testing.assert_array_equal(first, expected)
    assert compiled.selected_mode == "parallel"
    assert prepared_signatures and tuple(compiled.parallel.signatures) == prepared_signatures


def test_cpu_buffer_same_storage_uses_the_same_parallel_proof():
    from Infernux.compute import buffer

    values = buffer(shape=4096, dtype=np.float32, data=np.arange(4096, dtype=np.float32), device="cpu")
    compiled = jit.compile(_shared_update, parallel_policy="required")
    compiled(values, values)
    np.testing.assert_array_equal(values.numpy(copy=False), 2 * (np.arange(4096) + 1))
    assert compiled.selected_mode == "parallel"


@pytest.mark.parametrize("layout", ["zero_stride", "partial_element", "overlapping_rows"])
def test_layout_guard_rejects_internal_overlap(layout):
    storage = np.zeros(32, dtype=np.int64)
    shape, strides = {
        "zero_stride": ((16,), (0,)),
        "partial_element": ((16,), (4,)),
        "overlapping_rows": ((4, 4), (8, 8)),
    }[layout]
    view = np.lib.stride_tricks.as_strided(storage, shape=shape, strides=strides)
    assert array_arguments_alias((view,), {})


@pytest.mark.parametrize("view", [
    np.arange(24).reshape(4, 6),
    np.arange(24).reshape(4, 6).T,
    np.arange(24).reshape(4, 6)[::-1, ::2],
    np.arange(24)[::3],
    np.empty(0),
])
def test_nonoverlapping_layouts_are_admitted(view):
    assert not array_arguments_alias((view,), {})


def test_layout_proof_never_admits_overlapping_bytes_in_small_strided_corpus():
    for row in (-24, -16, -8, 0, 4, 8, 16, 24):
        for column in (-24, -16, -8, 0, 4, 8, 16, 24):
            view = np.ndarray((3, 3), dtype=np.int64, buffer=bytearray(512),
                              offset=256, strides=(row, column))
            byte_addresses = [i * row + j * column + byte
                              for i in range(3) for j in range(3) for byte in range(8)]
            if not array_arguments_alias((view,), {}):
                assert len(set(byte_addresses)) == len(byte_addresses), (row, column)


def test_overlap_cannot_reuse_parallel_layout_decision_or_modify_required_input(monkeypatch):
    monkeypatch.setattr(kernels, "static_cost_decision", lambda *args, **kwargs:
                        StaticCostDecision("parallel", "high", 10_000_000, "test workload"))
    compiled = jit.compile(_increment)
    ordinary = np.zeros(4096)
    compiled(ordinary)
    assert compiled.selected_mode == "parallel"
    storage = np.zeros(1)
    overlap = np.lib.stride_tricks.as_strided(storage, shape=(4096,), strides=(0,))
    jit.warmup(compiled, overlap)
    assert storage[0] == 0
    compiled(overlap)
    assert compiled.selected_mode == "serial"
    assert storage[0] == 4096
    compiled(ordinary)
    assert compiled.selected_mode == "parallel"
    np.testing.assert_array_equal(ordinary, 2)

    required = jit.compile(_increment, parallel_policy="required")
    for call in (required, lambda value: jit.warmup(required, value)):
        with pytest.raises(ValueError, match="overlapping"):
            call(overlap)
    assert storage[0] == 4096


def test_alias_proof_rejects_shifted_and_reinterpreted_views_before_execution():
    compiled = jit.compile(_shared_update, parallel_policy="required")
    storage = np.arange(4097, dtype=np.float64)
    for first, second in ((storage[:-1], storage[1:]), (storage, storage.view(np.int64))):
        with pytest.raises(ValueError, match="aliased"):
            compiled(first, second)
    np.testing.assert_array_equal(storage, np.arange(4097))


def test_stride_trick_warmup_copy_preserves_shared_owner_and_zero_stride():
    storage = np.arange(8)
    view = np.lib.stride_tricks.as_strided(storage[2:], shape=(4,), strides=(0,))
    args, _ = clone_call_arguments((view, storage), {})
    assert args[0].strides == (0,)
    assert np.shares_memory(*args)
    args[0][1] = 99
    assert args[1][2] == 99
    np.testing.assert_array_equal(storage, np.arange(8))


def test_warmup_does_not_silently_densify_an_external_overlapping_owner():
    view = np.ndarray((16,), dtype=np.int64, buffer=bytearray(8), strides=(0,))
    with pytest.raises(TypeError, match="non-contiguous storage owner"):
        clone_call_arguments((view,), {})
    np.testing.assert_array_equal(view, 0)


@pytest.mark.parametrize("values", [
    np.empty((0, 6))[:, ::-2], np.empty((3, 0))[::-1],
    np.empty(0)[::-1], np.empty((3, 0, 2)).transpose(2, 1, 0),
])
def test_empty_warmup_views_keep_layout_without_out_of_owner_offsets(values):
    args, _ = clone_call_arguments((values, values), {})
    assert args[0] is args[1] and args[0] is not values
    assert args[0].shape == values.shape
    assert args[0].strides == values.strides
    assert args[0].dtype == values.dtype


def test_cooked_alias_proof_needs_no_runtime_source_lookup(monkeypatch):
    source = "from Infernux import jit\n@jit.compile(parallel_policy='required')\n" + inspect.getsource(_shared_update)
    embedded = kernels.build_auto_parallel_embedded_source(source)
    assert embedded is not None

    def unavailable(*args, **kwargs):
        raise OSError("authored source is not distributed")

    monkeypatch.setattr(inspect, "getsource", unavailable)
    namespace = {"__name__": "cooked_alias_fixture"}
    exec(compile(embedded, "<cooked-alias>", "exec"), namespace)
    entry = namespace["__infernux_jit_manifest__"]["_shared_update"]
    assert entry["alias_pairs"] == (("first", "second"),)
    values = np.arange(4096, dtype=np.float64)
    namespace["_shared_update"](values, values.view())
    np.testing.assert_array_equal(values, 2 * (np.arange(4096) + 1))
    assert namespace["_shared_update"].selected_mode == "parallel"
