"""Repository commands must not depend on the operator's import environment."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _run(code, tmp_path, *arguments):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, *map(str, arguments)],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("entry", [
    "packaging/launcher.py",
    "packaging/installer_gui.py",
    "packaging/installer/install_python_runtime.py",
    "packaging/stage_bundled_python_runtime.py",
    "packaging/build_hub.py",
    "scripts/setup/package_android_support.py",
])
def test_hub_entrypoints_resolve_checkout_lock_without_engine(tmp_path, entry):
    result = _run("""
import importlib.abc
from pathlib import Path
import runpy
import sys
root, entry = Path(sys.argv[1]), Path(sys.argv[2])
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'infernux' or fullname.startswith('infernux.'):
            raise AssertionError('Hub must not import the native engine')
sys.meta_path.insert(0, NoEngine())
# Reproduce normal script-directory lookup while -I excludes ambient paths.
sys.path.insert(0, str(entry.parent))
namespace = runpy.run_path(str(entry), run_name='entrypoint_audit')
if '_support_module' in namespace:
    namespace['_support_module']()
import hub_utils
import infernux_project_lock
assert Path(infernux_project_lock.__file__).resolve() == root / 'python/infernux_project_lock.py'
print('checkout lock, no engine')
""", tmp_path, ROOT, ROOT / entry)
    assert "checkout lock, no engine" in result.stdout


@pytest.mark.parametrize("entry", [
    "editor_project_smoke.py", "asset_scan_profile_gate.py",
    "headless_smoke_test.py", "scene_residency_soak_test.py",
])
def test_source_acceptance_selects_checkout_before_loading_native(tmp_path, entry):
    result = _run("""
import importlib.abc
import importlib.machinery
from pathlib import Path
import runpy
import sys
root, entry = Path(sys.argv[1]), Path(sys.argv[2])
class ObserveEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'infernux':
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            assert spec is not None
            assert Path(spec.origin).resolve() == root / 'python/infernux/__init__.py'
            print('checkout engine selected')
            raise SystemExit(0)
sys.meta_path.insert(0, ObserveEngine())
runpy.run_path(str(entry), run_name='entrypoint_audit')
raise AssertionError('engine import not reached')
""", tmp_path, ROOT, ROOT / "tests/acceptance" / entry)
    assert "checkout engine selected" in result.stdout


def test_gpu_runner_uses_source_and_sibling_helpers_and_forwards_arguments(tmp_path):
    source = tmp_path / "checkout"
    gpu = source / "tests/gpu"
    gpu.mkdir(parents=True)
    package = source / "python/infernux"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("origin = 'checkout'\n", encoding="utf-8")
    (gpu / "helper.py").write_text("value = 42\n", encoding="utf-8")
    (gpu / "probe_test.py").write_text(
        "import infernux, helper, sys\n"
        "assert infernux.origin == 'checkout' and helper.value == 42\n"
        "assert sys.argv[1:] == ['--frames', '3']\n"
        "print('source and sibling resolved')\n", encoding="utf-8",
    )
    runner = gpu / "run.py"
    shutil.copy2(ROOT / "tests/gpu/run.py", runner)
    result = _run(
        "import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
        tmp_path, runner, "probe_test", "--frames", "3",
    )
    assert "source and sibling resolved" in result.stdout


def test_gpu_list_does_not_load_native_engine(tmp_path):
    result = _run(
        "import runpy,sys; sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')",
        tmp_path, ROOT / "tests/gpu/run.py", "--list",
    )
    assert "view_history_gpu_test" in result.stdout.splitlines()


def test_frozen_hub_does_not_add_source_directories(tmp_path):
    _run("""
import __main__
from pathlib import Path
import sys
import types
sys.path.insert(0, str(Path(sys.argv[1]) / 'packaging'))
__main__.__compiled__ = object()
sys.modules['infernux_project_lock'] = types.ModuleType('infernux_project_lock')
before = list(sys.path)
import hub_utils
assert hub_utils.is_frozen()
assert sys.path == before
""", tmp_path, ROOT)
