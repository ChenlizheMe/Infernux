"""Run Engine replacement cases outside the suite's resident native Engine."""

from pathlib import Path
import subprocess
import sys


def test_plugin_install_retirement_in_isolated_engines(tmp_path):
    fixture = Path(__file__).parent / "fixtures/plugin_install_retirement.py"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--confcutdir", str(fixture.parent),
         str(fixture), "--basetemp", str(tmp_path / "owners"),
         "--junitxml", str(tmp_path / "retirement.xml")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
