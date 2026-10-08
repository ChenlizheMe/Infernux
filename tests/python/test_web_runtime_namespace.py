"""First public access must not rely on a preceding direct submodule import."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('name, mode', [
    ('compute', 'lazy'), ('compute', 'direct'),
    ('jit', 'lazy'), ('jit', 'direct'),
    ('buffer', 'lazy'), ('Buffer', 'lazy'), ('unknown', 'lazy'),
])
def test_web_cold_public_namespace(name, mode):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / 'fixtures/web_runtime_namespace.py'), name, mode],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report['passed'] and report['requested'] == name and report['mode'] == mode
