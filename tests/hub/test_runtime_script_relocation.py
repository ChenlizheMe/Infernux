"""Run installed launchers after the staging path is gone, including project copies."""
import csv
import ast
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import zipfile
from types import SimpleNamespace

import pytest

from tests.tool_discovery import find_executable

import embed_runtime_manager as module
import private_python_runtime as runtime
from runtime_script_relocation import RELOCATE_RUNTIME_SCRIPTS


def command(args):
    result = subprocess.run([str(arg) for arg in args], capture_output=True,
                            timeout=45, creationflags=module._NO_WINDOW, env=module.merge_child_env_utf8())
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout.decode("utf-8")


@pytest.mark.parametrize("directory", ["space path", "O'Brien $HOME", "back`tick $(exit 19)"])
def test_posix_launcher_preserves_literal_interpreter_and_arguments(tmp_path, directory):
    shell = shutil.which("sh")
    if shell is None and sys.platform == "win32":
        git = find_executable("git")
        if git:
            candidate = Path(git).parents[1] / "usr/bin/sh.exe"
            if candidate.is_file():
                shell = str(candidate)
    if shell is None:
        pytest.skip("A POSIX shell is required to execute the Unix launcher format")
    # Compile the actual worker's platform-specific writer without running its
    # installation body against this test host's packages.
    definition = next(node for node in ast.parse(RELOCATE_RUNTIME_SCRIPTS).body
                      if isinstance(node, ast.ClassDef) and node.name == "RuntimeScriptMaker")
    from pip._vendor.distlib.scripts import ScriptMaker
    namespace = dict(ScriptMaker=ScriptMaker, os=SimpleNamespace(name="posix"))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "runtime-script-worker", "exec"), namespace)
    interpreter = tmp_path / directory / "python-shim"
    interpreter.parent.mkdir()
    interpreter.write_text("#!/bin/sh\nexec " + shlex.quote(Path(sys.executable).as_posix()) + ' "$@"\n', encoding="utf-8")
    interpreter.chmod(0o755)
    maker = namespace["RuntimeScriptMaker"](None, str(tmp_path))
    header = maker._build_shebang(shlex.quote(interpreter.as_posix()).encode("utf-8"), b"")
    script = tmp_path / "entrypoint"
    script.write_bytes(header + b"import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    result = command([shell, script.as_posix(), "one $HOME", "two ' args"])
    assert json.loads(result) == ["one $HOME", "two ' args"]


@pytest.mark.parametrize("source_kind", ["directory", "zip"])
def test_installed_and_project_launchers_use_their_own_python(tmp_path, monkeypatch, private_python_factory, source_kind):
    source = tmp_path / "bundle/python313"
    source_python = private_python_factory(source)
    archive = runtime.runtime_archive_for_machine()
    runtime.write_private_runtime_marker(source, archive.name, archive.sha256)
    wheel = tmp_path / "infernux_entry_probe-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as stream:
        stream.writestr("infernux_entry_probe.py",
                        "import json, sys\nfrom pathlib import Path\ndef main(): print(json.dumps(sys.executable))\n"
                        "def gui(): Path(sys.argv[1]).write_text(json.dumps(sys.executable), encoding='utf-8')\n")
        info = "infernux_entry_probe-1.0.dist-info/"
        stream.writestr(info + "METADATA", "Metadata-Version: 2.1\nName: infernux-entry-probe\nVersion: 1.0\n")
        stream.writestr(info + "WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        stream.writestr(info + "entry_points.txt",
                        "[console_scripts]\ninfernux-entry-probe = infernux_entry_probe:main\n"
                        "[gui_scripts]\ninfernux-gui-probe = infernux_entry_probe:gui\n")
        stream.writestr(info + "RECORD", "")
    # -I ignores PYTHONUTF8/PYTHONIOENCODING; select the encoding explicitly
    # so pip output containing a Chinese Windows user path remains decodable.
    command([source_python, "-I", "-X", "utf8", "-m", "pip", "install", "--no-index", "--no-deps", wheel])
    if source_kind == "zip":
        with zipfile.ZipFile(source.parent / "runtime_bundle.zip", "w") as stream:
            for path in source.rglob("*"):
                if path.is_file():
                    stream.write(path, path.relative_to(source.parent).as_posix())
        shutil.rmtree(source)
    manager = module.PythonRuntimeManager(runtime_dir=str(tmp_path / "Hub 中文 space & ' $name"))
    monkeypatch.setattr(manager, "bundled_runtime_dirs", lambda: [str(source.parent)])
    monkeypatch.setattr(module, "_has_build_support", lambda *a: True)
    monkeypatch.setattr(module, "_REQUIRED_RUNTIME_MODULES", ("pip", "infernux_entry_probe"))
    # Use actual preparation checks and relocation; no online packages are needed.
    installed = Path(manager.reinstall_runtime())
    shutil.rmtree(source.parent)
    root = Path(manager.private_runtime_root())
    scripts = "Scripts" if sys.platform == "win32" else "bin"
    suffix = ".exe" if sys.platform == "win32" else ""
    assert Path(json.loads(command([root / scripts / ("infernux-entry-probe" + suffix)]))).samefile(installed)
    assert str(root) in command([root / scripts / ("pip" + suffix), "--version"])
    project = tmp_path / "Project 中文 & $name/.runtime/python313"
    project_python = Path(manager.create_project_runtime(str(project)))
    # Make the Hub copy unavailable, so a stale launcher cannot accidentally pass.
    root.rename(root.with_name("retired"))
    assert Path(json.loads(command([project / scripts / ("infernux-entry-probe" + suffix)]))).samefile(project_python)
    for name in ["pip", "pip3", "pip3.13"]:
        assert str(project) in command([project / scripts / (name + suffix), "--version"])
    gui_result = tmp_path / "gui-result.json"
    command([project / scripts / ("infernux-gui-probe" + suffix), gui_result])
    assert Path(json.loads(gui_result.read_text(encoding="utf-8"))).samefile(
        project / "pythonw.exe" if sys.platform == "win32" else project_python)
    site = project / ("Lib/site-packages" if sys.platform == "win32" else "lib/python3.13/site-packages")
    with (site / "infernux_entry_probe-1.0.dist-info/RECORD").open(encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    launcher_rows = [row for row in rows if Path(row[0]).name == "infernux-entry-probe" + suffix]
    assert len(launcher_rows) == 1
    assert (site / launcher_rows[0][0]).is_file()
    assert launcher_rows[0][1] == ""
    assert int(launcher_rows[0][2]) == (site / launcher_rows[0][0]).stat().st_size
    command([project_python, "-I", "-X", "utf8", "-m", "pip", "uninstall", "-y", "infernux-entry-probe"])
    assert not (project / scripts / ("infernux-entry-probe" + suffix)).exists()
    assert not (project / scripts / ("infernux-gui-probe" + suffix)).exists()
