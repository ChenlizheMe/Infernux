"""Real Python/DLL execution across the Windows MAX_PATH boundary."""

from pathlib import Path
import json
import shutil
import sys

import pytest

import embed_runtime_manager as manager_module
import private_python_runtime
import python_execution


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 process and DLL loader contract")
@pytest.mark.parametrize("project_length", [197, 275])
def test_cold_project_runtime_loads_stdlib_and_package_extensions(
    tmp_path, monkeypatch, private_python_factory, project_length,
):
    manager = manager_module.PythonRuntimeManager(runtime_dir=str(tmp_path / "Hub"))
    source = Path(manager.private_runtime_root())
    private_python_factory(source)
    archive = private_python_runtime.runtime_archive_for_machine()
    private_python_runtime.write_private_runtime_marker(source, archive.name, archive.sha256)
    # This fixture supplies a real interpreter/stdlib/pip, but no engine wheels
    # or network dependencies. Runtime copying and launcher relocation are real.
    monkeypatch.setattr(manager_module, "_has_build_support", lambda *a: True)
    monkeypatch.setattr(manager, "_has_modules", lambda *a: True)
    project = tmp_path / "迁移 & cold clone"
    while len(str(project)) < project_length:
        project /= "subdirectory-" + "x" * min(35, project_length - len(str(project)))
    target = project / ".runtime" / "python313"
    python = manager.create_project_runtime(str(target))

    package = target / "Lib/site-packages" / ("extension_package_" + "x" * 60)
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    extension = source / "DLLs/unicodedata.pyd"
    shutil.copy2(extension, package / extension.name)
    assert len(str(package / extension.name)) > 260
    code = (
        "import ctypes, ssl, socket, sqlite3, json, sys, subprocess; "
        "subprocess.run([sys.executable, '-I', '-c', 'import ctypes, ssl, socket'], "
        "executable=sys.executable, check=True); "
        f"from {package.name} import unicodedata; "
        "print(json.dumps({'value': unicodedata.normalize('NFC', 'e\\u0301'), "
        "'module': unicodedata.__file__, 'python': sys.executable}))"
    )
    result = manager_module._run_command([python, "-I", "-X", "utf8", "-c", code], timeout=30)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    payload = json.loads(result.stdout)
    assert payload["value"] == "é"
    assert Path(payload["python"]).samefile(python)
    assert Path(payload["module"]).samefile(package / extension.name)
    assert not list(target.parent.glob(".python313.extract-*"))
    # Generated support is relocatable and does not retain either staging or
    # project paths; repeat preparation must not rewrite an unchanged runtime.
    hook = target / "Lib/site-packages/000_infernux_windows_paths.pth"
    before = hook.stat().st_mtime_ns
    assert str(project) not in hook.read_text(encoding="utf-8")
    python_execution.prepare_private_runtime_paths(target)
    assert hook.stat().st_mtime_ns == before


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 path spelling")
@pytest.mark.parametrize("path,expected", [
    (r"C:\project dir\python.exe", r"\\?\C:\project dir\python.exe"),
    (r"\\server\share\python.exe", r"\\?\UNC\server\share\python.exe"),
    (r"\\?\C:\project\python.exe", r"\\?\C:\project\python.exe"),
])
def test_windows_executable_path_is_explicit_and_idempotent(path, expected):
    assert python_execution.python_executable_path(path) == expected
    assert python_execution.python_executable_path(expected) == expected


def test_non_windows_execution_preserves_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(python_execution.sys, "platform", "linux")
    path = "/home/member/项目/bin/python"
    assert python_execution.python_executable_path(path) == path
    python_execution.prepare_private_runtime_paths(tmp_path)
    assert list(tmp_path.iterdir()) == []
