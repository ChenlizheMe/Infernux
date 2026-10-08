from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from tests import tool_discovery as tools


def test_blender_probes_candidates_once_and_uses_compatible_version(tmp_path, monkeypatch):
    old, broken, current = [tmp_path / name for name in ("old", "broken", "Blender 工具")]
    for path in (old, broken, current):
        path.touch()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[1:] == ["--version"] and not kwargs.get("shell")
        assert kwargs["timeout"] == 10
        if Path(command[0]) == broken:
            raise subprocess.TimeoutExpired(command, 10)
        return SimpleNamespace(returncode=0, stdout="Blender " + ("4.5.0" if Path(command[0]) == old else "5.2.2"), stderr="")

    monkeypatch.setattr(tools.subprocess, "run", run)
    assert tools.find_blender([tmp_path / "missing", old, old, broken, current]) == str(current)
    assert len(calls) == 3


@pytest.mark.parametrize("returncode,output", [(0, "Blender 4.5.0"), (1, "Blender 5.2.2"), (0, "not Blender")])
def test_unusable_blender_reports_actual_discovery_failure(tmp_path, monkeypatch, returncode, output):
    path = tmp_path / "blender"
    path.touch()
    monkeypatch.setattr(tools.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=output, stderr=""))
    with pytest.raises(FileNotFoundError, match="Blender 5.2 is required") as error:
        tools.find_blender([path])
    assert str(path) in str(error.value) and output in str(error.value)


def test_missing_blender_reports_installation_instead_of_a_test_flag(tmp_path):
    with pytest.raises(FileNotFoundError, match="no local Blender executable found"):
        tools.find_blender([tmp_path / "missing"])


def test_windows_blender_discovery_does_not_require_path_or_association(tmp_path, monkeypatch):
    binary = tmp_path / "Blender Foundation/Blender 5.2/blender.exe"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(tools, "_editor_blender", lambda: "")
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    monkeypatch.setattr(tools.sys, "platform", "win32")
    for name in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        monkeypatch.setenv(name, str(tmp_path))
    assert binary in list(tools.blender_candidates())


@pytest.mark.parametrize("name,relative", [("cmake", "CMake/bin/cmake.exe"),
                                         ("git", "Git/cmd/git.exe"),
                                         ("pwsh", "PowerShell/7/pwsh.exe")])
def test_standard_tool_installations_work_without_path(tmp_path, monkeypatch, name, relative):
    binary = tmp_path / relative
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(tools.sys, "platform", "win32")
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setattr(tools.shutil, "which", lambda candidate: str(binary) if candidate == str(binary) else None)
    assert tools.find_executable(name) == str(binary)


def test_current_python_environment_takes_precedence_over_path(tmp_path, monkeypatch):
    binary = tmp_path / "Library/bin/cmake.exe"
    binary.parent.mkdir(parents=True)
    binary.touch()
    monkeypatch.setattr(tools.sys, "platform", "win32")
    monkeypatch.setattr(tools.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(tools.shutil, "which", lambda candidate: str(binary) if candidate == str(binary) else "/wrong/cmake" if candidate == "cmake" else None)
    assert tools.find_executable("cmake") == str(binary)
