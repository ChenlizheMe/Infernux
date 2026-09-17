"""Real subprocess checks of compiled CPU cache publication boundaries."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(("kernel", "auto_parallel"), [
    ("direct", False), ("direct", True),
    ("indirect", False), ("indirect", True),
    ("module_constant", False), ("module_constant", True),
    ("foreign_helper", False), ("foreign_helper", True),
    ("fill", True),
])
def test_disk_cache_uses_published_constants_and_reuses_unchanged_revision(
    tmp_path, auto_parallel, kernel,
):
    environment = dict(os.environ)
    environment.update({
        "NUMBA_CACHE_DIR": str(tmp_path / "compiled"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "INFERNUX_TEST_JIT_AUTO": str(int(auto_parallel)),
    })
    fixture_dir = Path(__file__).with_name("fixtures")
    script = (
        "import json, sys, numpy as np; "
        "sys.path.insert(0, sys.argv[1]); "
        "import jit_cache_publication as source; "
        "fn = getattr(source, sys.argv[2]); "
        "values = np.zeros(8, dtype=np.int64); "
        "value = (fn(values), int(values[3]))[1] if sys.argv[2] == 'fill' else fn(3); "
        "native = fn.parallel if sys.argv[2] == 'fill' else "
        "(fn.serial if sys.argv[3] == '1' else fn._compiled); "
        "print(json.dumps({'value': value, 'hits': sum(native.stats.cache_hits.values())}))"
    )

    def run(factor):
        result = subprocess.run(
            [sys.executable, "-c", script, str(fixture_dir), kernel, str(int(auto_parallel))],
            env={**environment, "INFERNUX_TEST_JIT_FACTOR": str(factor)},
            text=True, capture_output=True, timeout=60, check=True,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])

    assert run(2) == {"value": 6, "hits": 0}
    assert run(5) == {"value": 15, "hits": 0}
    assert run(5) == {"value": 15, "hits": 1}
