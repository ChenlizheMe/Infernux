"""Public CPU compilation snapshots reflect actual code, not estimates."""

from dataclasses import asdict, FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
import weakref

from llvmlite.binding.executionengine import ExecutionEngine
import numpy as np
import pytest

from Infernux import jit
from Infernux._jit_kernels import _compiled_cache


def _increment(value):
    return value + 1


def _fill(values):
    for index in range(len(values)):
        values[index] += 1


def test_query_does_not_compile_execute_or_create_llvm_text(monkeypatch):
    compiled = jit.compile(_increment, auto_parallel=False)

    def unexpected(*args, **kwargs):
        pytest.fail("statistics must not compile code or serialize LLVM")

    monkeypatch.setattr(compiled._compiled, "compile", unexpected)
    monkeypatch.setattr(compiled._compiled, "inspect_llvm", unexpected)
    report = jit.statistics(compiled)
    assert report.specializations == ()
    assert report.owned_engine_count == report.reachable_engine_count == 0
    assert report.specialization_limit_per_implementation == 64
    assert report.selected_mode == "serial"


def test_steady_calls_do_not_read_or_update_statistics(monkeypatch):
    compiled = jit.compile(lambda value: value + 29, auto_parallel=False)
    assert compiled(2) == 31

    class NoAccess:
        def __getitem__(self, key):
            pytest.fail("steady calls must not inspect compilation statistics")

        def __setitem__(self, key, value):
            pytest.fail("steady calls must not update compilation statistics")

    monkeypatch.setattr(compiled._compiled, "_publication_stats", NoAccess())
    for value in range(64):
        assert compiled(value) == value + 29


def test_report_cold_specializations_and_json_snapshot():
    compiled = jit.compile(lambda value: value * 5, auto_parallel=False)
    assert compiled(3) == 15
    assert compiled(1.5) == 7.5
    report = jit.statistics(compiled)
    assert len(report.specializations) == report.owned_engine_count == 2
    assert report.reachable_engine_count >= 2
    for row in report.specializations:
        assert row.preparation_ms > 0
        assert row.preparation_succeeded and not row.cache_hit and not row.object_mode
        assert row.optimization_level in ("0", "1", "2", "3", "max")
        assert row.pipeline_timings
        assert all(timing.total_ms >= 0 for timing in row.pipeline_timings)
        assert any("lowering" in timing.name for timing in row.pipeline_timings)
    json.dumps(asdict(report), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        report.selected_mode = "unknown"
    before = report.specializations
    compiled(5)
    assert jit.statistics(compiled).specializations == before


def test_serial_only_auto_dispatcher_is_not_counted_twice():
    compiled = jit.compile(lambda value: value + 7)
    compiled(2)
    report = jit.statistics(compiled)
    assert len(report.specializations) == report.owned_engine_count == 1
    assert report.specializations[0].implementation == "serial"


def test_parallel_decision_and_implementations_are_reported():
    compiled = jit.compile(_fill, parallel_policy="required")
    values = np.zeros(64)
    jit.warmup(compiled, values)
    report = jit.statistics(compiled)
    np.testing.assert_array_equal(values, 0)
    assert report.selected_mode == "parallel"
    assert {row.implementation for row in report.specializations} == {"serial", "parallel"}
    assert len(report.decisions) == 1
    assert report.decisions[0][1].mode == "parallel"
    assert "selected" in report.last_diagnostic


def test_snapshot_does_not_retain_native_code_or_author_function():
    compiled = jit.compile(lambda value: value * 7, auto_parallel=False)
    compiled(2)
    result = next(iter(compiled._compiled.overloads.values()))
    reference = weakref.ref(result.library._codegen._engine._ee)
    report = jit.statistics(compiled)
    del compiled, result
    _compiled_cache.clear()
    gc.collect()
    assert reference() is None
    assert report.specializations[0].signature
    json.dumps(asdict(report))


def test_memory_is_measured_when_available_not_guessed_from_ir():
    compiled = jit.compile(lambda value: value * 11, auto_parallel=False)
    compiled(2)
    report = jit.statistics(compiled)
    engines = {id(result.library._codegen._engine._ee): result.library._codegen._engine._ee
               for result in compiled._compiled.overloads.values()}
    if hasattr(ExecutionEngine, "memory_statistics"):
        assert report.memory_statistics_available
        assert report.owned_mapped_bytes == sum(engine.memory_statistics["mapped_bytes"] for engine in engines.values())
        assert report.owned_mapped_bytes > 0
        assert report.reachable_mapped_bytes >= report.owned_mapped_bytes
        row = report.specializations[0]
        assert row.peak_mapped_bytes >= row.mapped_bytes > 0
    else:
        assert not report.memory_statistics_available
        assert report.owned_mapped_bytes is report.reachable_mapped_bytes is None
        assert report.specializations[0].mapped_bytes is None


def test_missing_memory_counter_is_explicitly_unavailable(monkeypatch):
    compiled = jit.compile(lambda value: value * 13, auto_parallel=False)
    compiled(2)
    monkeypatch.delattr(ExecutionEngine, "memory_statistics", raising=False)
    report = jit.statistics(compiled)
    assert not report.memory_statistics_available
    assert report.owned_mapped_bytes is report.reachable_mapped_bytes is None


def test_counter_errors_are_not_swallowed_as_zero_usage(monkeypatch):
    compiled = jit.compile(lambda value: value * 17, auto_parallel=False)
    compiled(2)

    def failed(engine):
        raise RuntimeError("counter failed")

    monkeypatch.setattr(ExecutionEngine, "memory_statistics", property(failed), raising=False)
    with pytest.raises(RuntimeError, match="counter failed"):
        jit.statistics(compiled)


def test_published_cache_write_failure_is_reported_without_retaining_the_error(monkeypatch):
    compiled = jit.compile(_fill, auto_parallel=False)

    def failed(*args):
        raise OSError("cache write failed")

    monkeypatch.setattr(compiled._compiled._cache, "save_overload", failed)
    values = np.zeros(8)
    with pytest.raises(OSError, match="cache write failed"):
        compiled(values)
    row = jit.statistics(compiled).specializations[0]
    assert not row.preparation_succeeded and row.preparation_ms > 0
    np.testing.assert_array_equal(values, 0)
    compiled(values)
    np.testing.assert_array_equal(values, 1)


def test_failed_unpublished_signature_is_not_reported_as_live_code(monkeypatch):
    compiled = jit.compile(lambda value: value * 19, auto_parallel=False)
    compiled(2)
    before = jit.statistics(compiled)

    def reject(*args):
        raise ValueError("publication rejected")

    monkeypatch.setattr(compiled._compiled, "add_overload", reject)
    with pytest.raises(ValueError, match="publication rejected"):
        compiled(2.0)
    assert jit.statistics(compiled) == before
    assert compiled(3) == 57


def test_cache_load_reports_current_preparation_not_old_compiler_timings(tmp_path):
    fixture_dir = Path(__file__).with_name("fixtures")
    script = (
        "import sys,json; from dataclasses import asdict; from Infernux import jit; "
        "sys.path.insert(0,sys.argv[1]); from jit_cache_publication import direct; "
        "value=direct(3); print(json.dumps({'value':value,'report':asdict(jit.statistics(direct))}))"
    )
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path), "INFERNUX_TEST_JIT_AUTO": "0",
           "INFERNUX_TEST_JIT_FACTOR": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    reports = []
    for _ in range(2):
        result = subprocess.run([sys.executable, "-c", script, str(fixture_dir)],
                                env=env, text=True, capture_output=True, check=True, timeout=60)
        data = json.loads(result.stdout.strip().splitlines()[-1])
        assert data["value"] == 6
        reports.append(data["report"]["specializations"][0])
    assert not reports[0]["cache_hit"] and reports[0]["pipeline_timings"]
    assert reports[1]["cache_hit"] and not reports[1]["pipeline_timings"]
    assert reports[1]["preparation_ms"] > 0 and reports[1]["preparation_succeeded"]


def test_statistics_rejects_an_uncompiled_python_function():
    with pytest.raises(TypeError, match="returned by jit.compile"):
        jit.statistics(_increment)


def test_query_waits_for_the_existing_compiler_publication_boundary(monkeypatch):
    compiled = jit.compile(lambda value: value * 23, auto_parallel=False)
    added, release, queried, finished = Event(), Event(), Event(), Event()
    add_overload = compiled._compiled.add_overload

    def hold_publication(result):
        add_overload(result)
        added.set()
        assert release.wait(10)

    def query():
        queried.set()
        try:
            return jit.statistics(compiled)
        finally:
            finished.set()

    monkeypatch.setattr(compiled._compiled, "add_overload", hold_publication)
    with ThreadPoolExecutor(max_workers=2) as pool:
        compilation = pool.submit(compiled, 2)
        try:
            assert added.wait(10)
            snapshot = pool.submit(query)
            assert queried.wait(10)
            assert not finished.wait(0.1)
        finally:
            release.set()
        assert compilation.result(timeout=10) == 46
        assert snapshot.result(timeout=10).specializations[0].preparation_succeeded
