"""A failed native build must stop the real Windows workflow run block."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or POWERSHELL is None,
    reason="Windows workflow control flow requires PowerShell 7 on Windows",
)


@pytest.mark.parametrize(
    "workflow,job,step_name,build_count,failing_call",
    [
        ("ci.yml", "windows-desktop", "Build Windows editor and native tests", 2, failing)
        for failing in range(3)
    ]
    + [
        ("platform-player.yml", "windows-player", "Build Windows Player runtime", 4, failing)
        for failing in range(5)
    ],
)
def test_native_build_failure_stops_workflow(
    tmp_path, workflow, job, step_name, build_count, failing_call
):
    document = yaml.safe_load(
        (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    )
    step = next(item for item in document["jobs"][job]["steps"] if item.get("name") == step_name)
    assert step["shell"] == "pwsh"
    python_literal = "'" + sys.executable.replace("'", "''") + "'"
    script = tmp_path / "workflow.ps1"
    # Replace only the external build commands, preserving the workflow's actual
    # shell control flow. Each invocation still obtains a real native exit code.
    prefix = f"""$ErrorActionPreference = 'Stop'
$global:buildCalls = 0
function conda {{}}
function cmake {{
    $global:buildCalls += 1
    Write-Output "BUILD_CALL=$global:buildCalls"
    if ($global:buildCalls -eq {failing_call}) {{
        & {python_literal} -c 'raise SystemExit(37)'
    }} else {{
        & {python_literal} -c 'raise SystemExit(0)'
    }}
}}
"""
    # GitHub's PowerShell runner appends the native exit-code propagation.
    script.write_text(prefix + step["run"] + "\nexit $LASTEXITCODE\n", encoding="utf-8")
    completed = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-File", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == (37 if failing_call else 0), completed.stdout + completed.stderr
    assert completed.stdout.count("BUILD_CALL=") == (failing_call or build_count)
