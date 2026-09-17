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
    return dispatcher.targetctx.codegen()._engine._ee


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


def test_specializations_share_owner_and_exception_does_not_retire_it(disposed_engines):
    @compile_cpu
    def function(value):
        if value < 0:
            raise ValueError("authored")
        return value * 2

    assert function(2) == 4
    assert function(2.5) == 5.0
    reference = weakref.ref(_engine(function))
    assert len(function.overloads) == 2
    with pytest.raises(ValueError, match="authored"):
        function(-1)
    assert function(3) == 6
    assert reference() is not None
    del function
    gc.collect()
    assert reference() is None


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
