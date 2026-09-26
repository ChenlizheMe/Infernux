"""Owned LLVM optimization skips no-body work and retires its resources."""

import llvmlite.binding as llvm
from numba.core.registry import cpu_target
import pytest

from Infernux._jit_backend import _OwnedContext


MODULE = """
declare double @external_one(double)
declare i64 @external_two(i64)
define double @first(double %value) {
    %result = fadd double %value, 1.0
    ret double %result
}
define i64 @second(i64 %value) {
    %result = add i64 %value, 2
    ret i64 %result
}
"""


@pytest.mark.parametrize("source", ["", "declare double @external_only(double)"])
def test_empty_or_declaration_only_modules_do_not_build_a_pipeline(monkeypatch, source):
    context = _OwnedContext(cpu_target.typing_context, "cpu")
    generator = context.codegen()
    library = generator.create_library("no-function-bodies")

    def unexpected():
        pytest.fail("no function body requires optimization")

    monkeypatch.setattr(generator, "_function_pass_manager", unexpected)
    with llvm.parse_assembly(source) as module:
        library._optimize_functions(module)
        module.verify()


def test_function_optimizer_only_constructs_pipelines_for_definitions(monkeypatch):
    context = _OwnedContext(cpu_target.typing_context, "cpu")
    generator = context.codegen()
    library = generator.create_library("defined-functions-only")
    factory = generator._function_pass_manager
    created = []

    def track():
        pair = factory()
        created.append(pair)
        return pair

    monkeypatch.setattr(generator, "_function_pass_manager", track)
    with llvm.parse_assembly(MODULE) as module:
        library._optimize_functions(module)
        module.verify()
        assert len(created) == 2
        assert all(manager.closed and builder.closed for manager, builder in created)
        assert module.data_layout == generator._data_layout


def test_optimizer_releases_manager_and_builder_when_native_run_raises(monkeypatch):
    context = _OwnedContext(cpu_target.typing_context, "cpu")
    generator = context.codegen()
    library = generator.create_library("failed-optimization")
    factory = generator._function_pass_manager
    created = []

    def track():
        manager, builder = factory()
        created.append((manager, builder))

        def fail(*args):
            raise RuntimeError("optimization failed")

        monkeypatch.setattr(manager, "run", fail)
        return manager, builder

    monkeypatch.setattr(generator, "_function_pass_manager", track)
    with llvm.parse_assembly(MODULE) as module:
        with pytest.raises(RuntimeError, match="optimization failed"):
            library._optimize_functions(module)
    assert len(created) == 1
    assert all(manager.closed and builder.closed for manager, builder in created)


def test_optional_llvm_pass_timing_uses_fresh_builders(monkeypatch):
    from numba.core import config

    monkeypatch.setattr(config, "LLVM_PASS_TIMINGS", True)
    context = _OwnedContext(cpu_target.typing_context, "cpu")
    library = context.codegen().create_library("timed-functions")
    with llvm.parse_assembly(MODULE) as module:
        library._optimize_functions(module)
    assert len(library.recorded_timings) == 2
