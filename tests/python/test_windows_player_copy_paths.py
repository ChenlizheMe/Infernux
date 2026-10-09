"""Exercise the real Windows copier with paths returned by private Python."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from infernux.engine import nuitka_builder
from infernux.engine.path_utils import process_executable_path


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 process image spelling")
def test_short_python_application_keeps_ordinary_site_directory_paths():
    python = os.path.abspath(sys.executable)
    if python.startswith("\\\\?\\"):
        python = python[4:]
    assert len(python.encode("utf-16-le")) // 2 < 260
    run = subprocess.run(
        [python, "-c", "import json,site;print(json.dumps(site.getsitepackages()))"],
        executable=process_executable_path("\\\\?\\" + python),
        capture_output=True, text=True, check=True, timeout=20,
    )
    assert all(not value.startswith("\\\\?\\") for value in json.loads(run.stdout))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows robocopy and extended paths")
@pytest.mark.parametrize("long_paths", [False, True])
def test_raw_runtime_copy_accepts_private_python_extended_paths(tmp_path, monkeypatch, long_paths):
    root = tmp_path / "中文 runtime"
    if long_paths:
        while len(str(root)) < 275:
            root /= "long private Python directory"
    site = root / "site"
    package = site / "llvmlite"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("value = 42\n", encoding="utf-8")
    companion = site / "llvmlite.libs"
    companion.mkdir()
    (companion / "dependency.bin").write_bytes(b"runtime dependency")
    monkeypatch.setattr(nuitka_builder, "_run_python", lambda *a, **kw:
                        SimpleNamespace(stdout=json.dumps(["\\\\?\\" + str(site)])))
    builder = object.__new__(nuitka_builder.NuitkaBuilder)
    builder._builder_python = sys.executable
    builder.raw_copy_packages = ["llvmlite"]
    dist = root / "dist"
    builder._inject_jit_packages("\\\\?\\" + str(dist))
    assert (dist / "llvmlite/__init__.pyc").is_file()
    assert not (dist / "llvmlite/__init__.py").exists()
    assert (dist / "llvmlite.libs/dependency.bin").read_bytes() == b"runtime dependency"
