"""Execute the actual generated script, replacing only its COM writer."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import installer_gui


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows shortcut contract")

RECORDER = r'''
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$literalSegment = 'EXPANDED'
function New-Object {
    $obj = [pscustomobject]@{}
    $obj | Add-Member ScriptMethod CreateShortcut {
        param($path)
        $global:shortcutPath = $path
        $shortcut = [pscustomobject]@{ TargetPath=''; WorkingDirectory=''; Description='' }
        $shortcut | Add-Member ScriptMethod Save {
            $global:recorded = [ordered]@{ ShortcutPath=$global:shortcutPath; TargetPath=$this.TargetPath; WorkingDirectory=$this.WorkingDirectory }
        }
        return $shortcut
    }
    return $obj
}
'''


@pytest.mark.parametrize("segment", [
    "中文 空格 & Hub", "$literalSegment", "$(Get-Date)", "path`name", "O'Brien",
    "[0];#Hub", "mixed ' $literalSegment ` $(Get-Date)",
])
@pytest.mark.parametrize("location", ["install", "programs"])
def test_shortcut_paths_are_literal_in_generated_powershell(tmp_path, monkeypatch, segment, location):
    install = tmp_path / (segment if location == "install" else "Hub")
    programs = tmp_path / (segment if location == "programs" else "Programs")
    install.mkdir()
    calls = []

    def folder(_hwnd, _folder, _token, _flags, buffer):
        buffer.value = str(programs)
        return 0

    with monkeypatch.context() as patch:
        patch.setattr(ctypes, "windll", SimpleNamespace(shell32=SimpleNamespace(SHGetFolderPathW=folder)))
        patch.setattr(subprocess, "run", lambda command, **kwargs:
                      calls.append(command) or SimpleNamespace(returncode=0))
        installer_gui._create_start_menu_shortcut(str(install))
    assert len(calls) == 1
    script = tmp_path / "record-shortcut.ps1"
    script.write_text(RECORDER + calls[0][-1] + '\n$recorded | ConvertTo-Json -Compress\n', encoding="utf-8-sig")
    shell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    result = subprocess.run([str(shell), "-NoProfile", "-NonInteractive", "-File", str(script)],
                            capture_output=True, encoding="utf-8", timeout=15, creationflags=0x08000000)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed == dict(ShortcutPath=str(programs / "Infernux Hub/Infernux Hub.lnk"),
                            TargetPath=str(install / "Infernux Hub.exe"), WorkingDirectory=str(install))
