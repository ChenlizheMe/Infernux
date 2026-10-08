"""Keep executable acceptance drivers separate from pytest test modules."""

import pytest
import subprocess

from tests.tool_discovery import find_blender, find_executable

collect_ignore = ["acceptance", "android", "fixtures", "gpu", "native", "release"]


@pytest.fixture(scope="session")
def blender_executable():
    """Run real Blender tests whenever a supported local installation exists."""
    try:
        return find_blender()
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture
def directory_junction(tmp_path):
    """Create a Windows test junction with literal arguments, not process globals."""
    def create(link, target):
        shell = find_executable("pwsh") or find_executable("powershell")
        if not shell:
            pytest.skip("PowerShell is required to create a Windows directory junction")
        script = tmp_path / "create-junction.ps1"
        script.write_text(
            "param([string]$Link, [string]$Target)\n$ErrorActionPreference = 'Stop'\n"
            "New-Item -ItemType Junction -Path $Link -Target $Target | Out-Null\n",
            encoding="utf-8",
        )
        subprocess.run([shell, "-NoProfile", "-NonInteractive", "-File", str(script),
                        "-Link", str(link), "-Target", str(target)],
                       check=True, capture_output=True, timeout=30)
    return create
