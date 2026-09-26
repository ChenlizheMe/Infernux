"""Compiled code retires only after its final native consumer disappears."""

import gc
import subprocess
import sys
import weakref

import numpy as np
import pytest
from llvmlite.binding.executionengine import ExecutionEngine
from numba import prange
from numba.core.registry import cpu_target
from numba.core import serialize

from Infernux import jit
from Infernux._jit_backend import compile_cpu
from Infernux._jit_kernels import _compiled_cache


def _engine(dispatcher):
    result = next(reversed(dispatcher.overloads.values()))
    return result.library._codegen._engine._ee


@pytest.fixture
def disposed_engines(monkeypatch):
    gc.collect()
    disposed = []
    original = ExecutionEngine._dispose

    def dispose(engine):
        disposed.append(id(engine))
        original(engine)

    monkeypatch.setattr(ExecutionEngine, "_dispose", dispose)
    return disposed


def test_public_compilation_retires_engine_after_cache_and_author_release(disposed_engines):
    def function(value):
        return value * 3.25

    wrapper = jit.compile(function, auto_parallel=False)
    assert wrapper(4.0) == 13.0
    native = wrapper._compiled
    reference = weakref.ref(_engine(native))
    identity = id(reference())
    del native, wrapper
    gc.collect()
    assert reference() is not None  # The bounded compilation cache owns it.
    _compiled_cache.clear()
    gc.collect()
    assert reference() is None
    assert disposed_engines.count(identity) == 1


def test_scalar_code_does_not_accumulate_in_shared_engine(disposed_engines):
    compile_cpu(lambda value: value)(0)  # Initialize shared NRT/builtins.
    gc.collect()
    shared = cpu_target.target_context.codegen()._engine._ee
    baseline = len(shared._modules)
    for batch in range(3):
        functions = []
        references = []
        for offset in range(8):
            amount = batch * 8 + offset

            def make_function(amount):
                def kernel(value):
                    return value + amount
                return compile_cpu(kernel)

            functions.append(make_function(amount))
            assert functions[-1](4) == amount + 4
            references.append(weakref.ref(_engine(functions[-1])))
        identities = [id(reference()) for reference in references]
        functions.clear()
        gc.collect()
        assert all(reference() is None for reference in references)
        assert all(identity in disposed_engines for identity in identities)
        assert len(shared._modules) == baseline


def test_parallel_helper_libraries_retire_with_owner(disposed_engines):
    @compile_cpu(parallel=True)
    def fill(values):
        for index in prange(len(values)):
            values[index] = index * 2.0

    values = np.zeros(128)
    fill(values)
    np.testing.assert_array_equal(values, np.arange(128) * 2.0)
    reference = weakref.ref(_engine(fill))
    identity = id(reference())
    del fill
    gc.collect()
    assert reference() is None
    assert disposed_engines.count(identity) == 1


def test_caller_keeps_callee_code_alive_after_dispatcher_disappears(disposed_engines):
    callee = compile_cpu(lambda value: value + 11)
    namespace = {"callee": callee}
    exec("def function(value): return callee(value) * 2", namespace)
    caller = compile_cpu(namespace["function"])
    assert caller(3) == 28
    callee_reference = weakref.ref(_engine(callee))
    caller_reference = weakref.ref(_engine(caller))
    namespace.pop("callee")
    del callee
    gc.collect()
    # No recompilation: the linked machine code must remain callable.
    assert caller(4) == 30
    assert callee_reference() is not None
    del caller
    gc.collect()
    assert caller_reference() is None
    assert callee_reference() is None


def test_native_entry_point_keeps_code_alive_without_dispatcher(disposed_engines):
    dispatcher = compile_cpu(lambda value: value * 7)
    assert dispatcher(3) == 21
    entry = next(iter(dispatcher.overloads.values())).entry_point
    reference = weakref.ref(_engine(dispatcher))
    del dispatcher
    gc.collect()
    assert reference() is not None
    assert entry(4) == 28
    del entry
    gc.collect()
    assert reference() is None


def test_restored_dispatcher_preserves_numba_constructor_contract():
    dispatcher = compile_cpu(lambda value: value + 0.75, fastmath=True)
    assert dispatcher(2.0) == 2.75
    payload = serialize.dumps(dispatcher)
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from numba.core.serialize import loads; "
         "fn = loads(sys.stdin.buffer.read()); "
         "assert fn.targetoptions['fastmath']; print(fn(4.0))"],
        input=payload, capture_output=True, timeout=60, check=True,
    )
    assert result.stdout.strip() == b"4.75"


def test_disk_loaded_code_is_owned_and_retires(tmp_path, disposed_engines):
    from Infernux._jit_kernels import _compile_njit
    from Infernux.engine.project_context import using_project_root

    def function(value):
        return value * 4.5

    with using_project_root(str(tmp_path)):
        for expected_hits in (0, 1):
            dispatcher = _compile_njit(function, {"cache": True})
            assert dispatcher(2.0) == 9.0
            assert sum(dispatcher.stats.cache_hits.values()) == expected_hits
            reference = weakref.ref(_engine(dispatcher))
            identity = id(reference())
            del dispatcher
            gc.collect()
            assert reference() is None
            assert identity in disposed_engines


def test_specializations_own_distinct_code_and_runtime_exception_does_not_retire_it(disposed_engines):
    @compile_cpu
    def function(value):
        if value < 0:
            raise ValueError("authored")
        return value * 2

    assert function(2) == 4
    integer = weakref.ref(_engine(function))
    assert function(2.5) == 5.0
    reference = weakref.ref(_engine(function))
    assert integer() is not reference()
    assert len(function.overloads) == 2
    with pytest.raises(ValueError, match="authored"):
        function(-1)
    assert function(3) == 6
    assert reference() is not None
    assert integer() is not None
    del function
    gc.collect()
    assert reference() is None
    assert integer() is None


def test_authored_helpers_do_not_accumulate_in_global_codegen(disposed_engines):
    compile_cpu(lambda value: value)(0)
    gc.collect()
    shared = cpu_target.target_context.codegen()._engine._ee
    baseline = len(shared._modules)
    for factor in (2, 5, 8):
        namespace = {"__name__": "owned_helpers", "FACTOR": factor}
        exec("def leaf(value): return value * FACTOR\n"
             "def helper(value): return leaf(value)\n"
             "def kernel(value): return helper(value)", namespace)
        author_helper = namespace["helper"]
        function = jit.compile(namespace["kernel"], auto_parallel=False)
        assert function(3) == 3 * factor
        assert namespace["helper"] is author_helper
        assert author_helper(4) == 4 * factor
        native = function._compiled
        helper = native.py_func.__globals__["helper"]
        leaf = helper.py_func.__globals__["leaf"]
        references = [weakref.ref(_engine(item)) for item in (native, helper, leaf)]
        del native, helper, leaf
        del function, namespace, author_helper
        _compiled_cache.clear()
        gc.collect()
        assert all(reference() is None for reference in references)
        assert len(shared._modules) == baseline


def test_compiler_namespace_does_not_retain_unreferenced_author_data():
    unused = np.zeros(1024)
    reference = weakref.ref(unused)
    namespace = {"__name__": "owned_namespace", "UNUSED": unused, "FACTOR": 3}
    exec("def helper(value): return value * FACTOR\n"
         "def kernel(value): return helper(value)", namespace)
    function = compile_cpu(namespace["kernel"])
    assert function(2) == 6
    namespace.pop("UNUSED")
    del unused
    gc.collect()
    assert reference() is None
    assert function(4) == 12


def test_nested_code_preserves_its_referenced_globals():
    namespace = {"__name__": "owned_nested", "FACTOR": 3}
    exec("def helper(value): return value * FACTOR\n"
         "def kernel(value):\n"
         " def nested(item): return helper(item) + FACTOR\n"
         " return nested(value)", namespace)
    function = jit.compile(namespace["kernel"], auto_parallel=False)
    assert function(4) == 15


def test_new_specialization_limit_preserves_existing_code_and_rejects_before_writes(monkeypatch):
    from Infernux import _jit_backend

    monkeypatch.setattr(_jit_backend, "_MAX_CPU_SPECIALIZATIONS", 2)

    @jit.compile(auto_parallel=False)
    def modify(values):
        values[0] += 1

    first = np.zeros(2, dtype=np.float32)
    second = np.zeros(2, dtype=np.float64)
    rejected = np.zeros(2, dtype=np.int32)
    modify(first)
    modify(second)
    engine = _engine(modify._compiled)
    module_count = len(engine._modules)
    with pytest.raises(RuntimeError, match="specialization limit \\(2\\).*not compiled"):
        modify(rejected)
    np.testing.assert_array_equal(rejected, [0, 0])
    assert len(engine._modules) == module_count
    assert len(modify.signatures) == 2
    modify(first)
    modify(second)
    assert first[0] == second[0] == 2


def test_array_length_does_not_consume_specializations(monkeypatch):
    from Infernux import _jit_backend

    monkeypatch.setattr(_jit_backend, "_MAX_CPU_SPECIALIZATIONS", 1)
    function = compile_cpu(lambda values: len(values))
    for length in (0, 1, 7, 64, 1024):
        assert function(np.zeros(length, dtype=np.float32)) == length
    assert len(function.signatures) == 1
    # Explicitly requesting an existing signature also works at capacity.
    function.compile(function.signatures[0])


def test_dtype_rank_and_layout_are_bounded_native_specializations(monkeypatch):
    from Infernux import _jit_backend

    monkeypatch.setattr(_jit_backend, "_MAX_CPU_SPECIALIZATIONS", 4)

    @jit.compile(auto_parallel=False)
    def first_value(values):
        return values.flat[0]

    variants = (
        np.ones((2, 3), dtype=np.float32, order="C"),
        np.ones((2, 3), dtype=np.float32, order="F"),
        np.ones(6, dtype=np.float32),
        np.ones((2, 3), dtype=np.float64, order="C"),
    )
    for values in variants:
        assert first_value(values) == 1
    assert len(first_value.signatures) == 4

    # Extents are runtime values: different ordinary quantities reuse the
    # existing dtype/rank/layout specialization instead of consuming code.
    assert first_value(np.ones((9, 11), dtype=np.float32, order="C")) == 1
    assert len(first_value.signatures) == 4

    rejected = np.ones((2, 3), dtype=np.int32, order="C")
    with pytest.raises(RuntimeError, match="specialization limit \\(4\\).*not compiled"):
        first_value(rejected)
    assert len(first_value.signatures) == 4


def test_structured_field_schema_selects_native_specialization_before_writes(monkeypatch):
    from Infernux import _jit_backend

    monkeypatch.setattr(_jit_backend, "_MAX_CPU_SPECIALIZATIONS", 2)

    @jit.compile(auto_parallel=False)
    def increment_x(values):
        values[0]["x"] += 1.0

    compact = np.zeros(1, dtype=[("x", np.float32)])
    padded = np.zeros(1, dtype=[("padding", np.float32), ("x", np.float32)])
    rejected = np.zeros(
        1,
        dtype=[("padding", np.int32), ("x", np.float32)],
    )

    increment_x(compact)
    increment_x(padded)
    assert compact[0]["x"] == padded[0]["x"] == 1.0
    assert len(increment_x.signatures) == 2

    with pytest.raises(RuntimeError, match="specialization limit \\(2\\).*not compiled"):
        increment_x(rejected)
    assert rejected[0]["x"] == 0.0
    assert len(increment_x.signatures) == 2


def test_concurrent_new_types_share_specialization_capacity(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from Infernux import _jit_backend

    monkeypatch.setattr(_jit_backend, "_MAX_CPU_SPECIALIZATIONS", 1)
    function = compile_cpu(lambda values: len(values))
    ready = Barrier(2)

    def compile_type(dtype):
        ready.wait(timeout=10)
        try:
            return function(np.zeros(3, dtype=dtype))
        except RuntimeError as error:
            assert "specialization limit (1)" in str(error)
            return "capacity"

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(compile_type, dtype) for dtype in (np.int32, np.float64)]
        results = [future.result(timeout=30) for future in pending]
    assert results.count(3) == 1
    assert results.count("capacity") == 1
    assert len(function.signatures) == 1


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_unpublished_native_code_is_closed_without_invalidating_old_specializations(
    monkeypatch, failure,
):
    from Infernux._jit_backend import _OwnedDispatcher

    @compile_cpu
    def increment(values):
        values[0] += 1

    original = np.zeros(2, dtype=np.int64)
    increment(original)
    published = _engine(increment)
    original_context = increment.targetctx
    abandoned = []

    def reject(self, result):
        abandoned.append(result.library._codegen._engine._ee)
        raise failure("rejected before publication")

    monkeypatch.setattr(_OwnedDispatcher, "add_overload", reject)
    candidate = np.zeros(2, dtype=np.float64)
    held_errors = []
    for _ in range(3):
        with pytest.raises(failure, match="before publication") as caught:
            increment(candidate)
        held_errors.append(caught.value)
        np.testing.assert_array_equal(candidate, [0, 0])
        assert len(increment.overloads) == 1
        assert increment.targetctx is original_context
        assert increment.targetdescr.target_context is original_context
        # Keep both exception context and engine references alive: native
        # disposal must not depend on a future garbage-collection cycle.
        assert all(item.closed for item in abandoned)
        assert not published.closed
        increment(original)
    assert original[0] == 4


def test_cache_write_failure_after_publication_keeps_callable_code(monkeypatch):
    @compile_cpu
    def increment(values):
        values[0] += 1

    def fail_save(*args):
        raise OSError("cache write failed")

    monkeypatch.setattr(increment._cache, "save_overload", fail_save)
    values = np.zeros(2)
    with pytest.raises(OSError, match="cache write failed"):
        increment(values)
    assert len(increment.overloads) == 1
    assert not _engine(increment).closed
    assert values[0] == 0  # Compiler failure did not execute the user function.
    increment(values)
    assert values[0] == 1


def test_cached_specialization_rejection_closes_only_unpublished_engine(tmp_path, monkeypatch):
    from Infernux._jit_backend import _OwnedDispatcher
    from Infernux._jit_kernels import _compile_njit
    from Infernux.engine.project_context import using_project_root

    def increment(values):
        values[0] += 1

    with using_project_root(str(tmp_path)):
        seed = _compile_njit(increment, {"cache": True})
        seed(np.zeros(2, dtype=np.float64))
        del seed
        gc.collect()
        native = _compile_njit(increment, {"cache": True})
        existing = np.zeros(2, dtype=np.int64)
        native(existing)
        original = _engine(native)
        abandoned = []

        def reject(self, result):
            abandoned.append(result.library._codegen._engine._ee)
            raise RuntimeError("cached code rejected")

        monkeypatch.setattr(_OwnedDispatcher, "add_overload", reject)
        candidate = np.zeros(2, dtype=np.float64)
        with pytest.raises(RuntimeError, match="cached code rejected"):
            native(candidate)
        assert sum(native.stats.cache_hits.values()) == 1
        assert abandoned[0].closed
        assert not original.closed
        assert candidate[0] == 0
        native(existing)
        assert existing[0] == 2


@pytest.mark.parametrize("first", [int, np.int32])
def test_recursive_typing_restores_enclosing_compilation_context(first):
    namespace = {"compile_cpu": compile_cpu, "__name__": __name__}
    exec("@compile_cpu\ndef factorial(n):\n"
         "    if n < 2: return 1\n"
         "    return n * factorial(n - 1)\n", namespace)
    native = namespace["factorial"]
    assert native(first(6)) == 720
    assert native(7) == 5040
    assert native(np.int32(5)) == 120
    assert native(8) == 40320
