"""Multidimensional row-local CPU work through HIR and real native code."""

import inspect

import numpy as np
import pytest

from Infernux import jit
from Infernux import _jit_kernels as kernels
from Infernux.jit_hir import build_hir


def _advance(positions, velocities, dt):
    for i in range(len(positions)):
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt


def _row_mix(values):
    for i in range(len(values)):
        values[i, 0] += values[i, -1]
        values[i, 1] = values[i, 0] * 2


@pytest.mark.parametrize("write,read", [
    ("i, 0", "i, 1"), ("0, i", "1, i"),
    ("i, 0, 1", "i, 1, 0"), ("i, i", "i, i"),
    ("i, -1", "i, 0"),
])
def test_row_local_scalar_accesses_are_eligible(write, read):
    hir = build_hir(f"def update(values, n):\n    for i in range(n):\n        values[{write}] += values[{read}]\n")
    assert hir.eligible_loops, hir.diagnostics


@pytest.mark.parametrize("bounds,write,read", [
    ("1, n", "i, 0", "i-1, 0"),
    ("n", "i, 0", "0, i"),
    ("n", "0, 0", "i, 0"),
    ("n", "i, :", "i, 0"),
    ("n", "i, indices[i]", "i, 0"),
    ("-1, n", "i, 0", "i, 1"),
    ("n, -1, -1", "i, 0", "i, 1"),
])
def test_unproven_row_accesses_do_not_get_parallel_permission(bounds, write, read):
    hir = build_hir(f"def update(values, indices, n):\n    for i in range({bounds}):\n        values[{write}] += values[{read}]\n")
    assert not hir.eligible_loops


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided", "reversed"])
@pytest.mark.parametrize("count", [0, 1, 1024])
def test_real_parallel_rows_preserve_strided_numpy_semantics(layout, count):
    original = np.arange(count * 6, dtype=np.float64).reshape(count, 6)
    positions = {
        "contiguous": original[:, :3].copy(),
        "fortran": np.asfortranarray(original[:, :3]),
        "strided": original[:, ::2],
        "reversed": original[::-1, ::-2],
    }[layout]
    velocities = positions.copy() + 1
    expected = positions.copy() + velocities * 0.125
    function = jit.compile(_advance, parallel_policy="required")
    jit.warmup(function, positions, velocities, 0.125)
    before = positions.copy()
    function(positions, velocities, 0.125)
    np.testing.assert_array_equal(positions, expected)
    np.testing.assert_array_equal(before + velocities * 0.125, expected)
    assert function.selected_mode == "parallel"


def test_shared_row_storage_and_negative_column_follow_python_order():
    values = np.arange(3072, dtype=np.float64).reshape(1024, 3)
    expected = values.copy()
    _row_mix(expected)
    function = jit.compile(_row_mix, parallel_policy="required")
    function(values)
    np.testing.assert_array_equal(values, expected)
    assert function.selected_mode == "parallel"

    function = jit.compile(_advance, parallel_policy="required")
    expected += expected * 0.125
    function(values, values.view(), 0.125)
    np.testing.assert_array_equal(values, expected)


def test_overlapping_rows_rejected_before_writes():
    storage = np.zeros(1026)
    values = np.lib.stride_tricks.as_strided(storage, shape=(1024, 3), strides=(8, 8))
    function = jit.compile(_advance, parallel_policy="required")
    with pytest.raises(ValueError, match="overlapping|aliased"):
        function(values, np.ones((1024, 3)), 0.125)
    np.testing.assert_array_equal(storage, 0)


def _column_mix(values):
    for i in range(values.shape[1]):
        values[0, i] += values[1, i]


def _volume_mix(values):
    for i in range(len(values)):
        values[i, 0, 1] += values[i, 1, 0]


def _diagonal_mix(values):
    for i in range(len(values)):
        values[i, i] += 2


@pytest.mark.parametrize("kernel,shape", [
    (_column_mix, (2, 1024)), (_volume_mix, (1024, 2, 2)),
    (_diagonal_mix, (128, 128)),
])
def test_native_induction_axis_and_rank_match_python(kernel, shape):
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    expected = values.copy()
    kernel(expected)
    function = jit.compile(kernel, parallel_policy="required")
    function(values)
    np.testing.assert_array_equal(values, expected)
    assert function.selected_mode == "parallel"


@pytest.mark.parametrize("vector", [False, True])
def test_cpu_buffer_rows_share_the_same_native_path(vector):
    from Infernux.compute import buffer
    from Infernux.math import vector3

    values = buffer(shape=1024 if vector else (1024, 3),
                    dtype=vector3 if vector else np.float32,
                    data=np.ones((1024, 3), dtype=np.float32), device="cpu")
    function = jit.compile(_advance, parallel_policy="required")
    function(values, values, 0.125)
    np.testing.assert_array_equal(values.numpy(copy=False), 1.125)
    assert function.selected_mode == "parallel"


def test_cooked_row_kernel_does_not_need_source(monkeypatch):
    source = "from Infernux import jit\n@jit.compile(parallel_policy='required')\n" + inspect.getsource(_advance)
    embedded = kernels.build_auto_parallel_embedded_source(source)
    assert embedded is not None

    def unavailable(*args, **kwargs):
        raise OSError("no author source")

    monkeypatch.setattr(inspect, "getsource", unavailable)
    namespace = {"__name__": "cooked_rows"}
    exec(compile(embedded, "<cooked-rows>", "exec"), namespace)
    values = np.ones((1024, 3))
    namespace["_advance"](values, values.view(), 0.125)
    np.testing.assert_array_equal(values, 1.125)
    assert namespace["_advance"].selected_mode == "parallel"
