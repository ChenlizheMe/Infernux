"""Real subprocess checks of compiled CPU cache publication boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import infernux

PACKAGE_ROOT = str(Path(infernux.__file__).resolve().parent.parent)


@pytest.mark.parametrize(("kernel", "auto_parallel"), [
    ("direct", False), ("direct", True),
    ("indirect", False), ("indirect", True),
    ("module_constant", False), ("module_constant", True),
    ("foreign_helper", False), ("foreign_helper", True),
    ("fill", True),
    ("source_less", False),
])
def test_disk_cache_uses_published_constants_and_reuses_unchanged_revision(
    tmp_path, auto_parallel, kernel,
):
    environment = dict(os.environ)
    environment.update({
        "NUMBA_CACHE_DIR": str(tmp_path / "compiled"),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    fixture_dir = Path(__file__).with_name("fixtures")
    script = (
        "import json, sys, numpy as np; from dataclasses import asdict; "
        "sys.path.insert(0, sys.argv[5]); from infernux import jit; "
        "sys.path.insert(0, sys.argv[1]); "
        "import jit_cache_dependency as settings; settings.configure(int(sys.argv[4]), sys.argv[3] == '1'); "
        "import jit_cache_publication as source; "
        "fn = getattr(source, sys.argv[2]); "
        "values = np.zeros(8, dtype=np.int64); "
        "value = (fn(values), int(values[3]))[1] if sys.argv[2] == 'fill' else fn(3); "
        "native = fn.parallel if sys.argv[2] == 'fill' else "
        "(fn.serial if sys.argv[3] == '1' else fn._compiled); "
        "print(json.dumps({'value': value, 'hits': sum(native.stats.cache_hits.values()), "
        "'report': asdict(jit.statistics(fn))['specializations'][0]}))"
    )

    def run(factor):
        result = subprocess.run(
            [sys.executable, "-c", script, str(fixture_dir), kernel, str(int(auto_parallel)), str(factor), PACKAGE_ROOT],
            env=environment,
            text=True, capture_output=True, timeout=60, check=True,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    cold, changed, cached = run(2), run(5), run(5)
    assert [(result["value"], result["hits"]) for result in (cold, changed, cached)] == [(6, 0), (15, 0), (15, 1)]
    # Reuse the same process launches for statistics and publication checks.
    # The serial direct case previously repeated two extra interpreter launches.
    if kernel == "direct" and not auto_parallel:
        assert not cold["report"]["cache_hit"] and cold["report"]["pipeline_timings"]
        assert cached["report"]["cache_hit"] and not cached["report"]["pipeline_timings"]
        assert cached["report"]["preparation_ms"] > 0 and cached["report"]["preparation_succeeded"]


def test_recursive_specialization_cache_does_not_require_another_live_engine(tmp_path):
    environment = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "compiled"),
                   "PYTHONDONTWRITEBYTECODE": "1"}
    fixture_dir = Path(__file__).with_name("fixtures")
    script = (
        "import json, sys, numpy as np; sys.path.insert(0, sys.argv[3]); sys.path.insert(0, sys.argv[1]); "
        "from jit_cache_recursive import factorial; "
        "first = factorial(7) if sys.argv[2] == 'warm' else None; "
        "result = factorial(np.int32(6)); "
        "print(json.dumps({'value':result, 'hits':sum(factorial.stats.cache_hits.values()), "
        "'signatures':len(factorial.overloads)}))"
    )

    def run(mode):
        result = subprocess.run(
            [sys.executable, "-c", script, str(fixture_dir), mode, PACKAGE_ROOT],
            env=environment, text=True, capture_output=True, timeout=60, check=True,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    assert run("warm") == {"value": 720, "hits": 0, "signatures": 2}
    # A fresh process only requests int32: the linked int64 implementation
    # must be contained in its object file, not an old process's MCJIT pool.
    assert run("cached") == {"value": 720, "hits": 1, "signatures": 1}
