"""Exercise the managed runtime's public install path without network packages."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest

import embed_runtime_manager as module
import private_python_runtime as runtime


def write_runtime(root: Path, label: str) -> None:
    root.mkdir(parents=True)
    executable = root / ("python.exe" if sys.platform == "win32" else "bin/python")
    executable.parent.mkdir(exist_ok=True)
    shutil.copy2(sys.executable, executable)
    if sys.platform == "win32":
        shutil.copy2(Path(sys.base_prefix) / "python313.dll", root / "python313.dll")
    (root / "pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).parent}\ninclude-system-site-packages = false\n", encoding="utf-8")
    archive = runtime.runtime_archive_for_machine()
    runtime.write_private_runtime_marker(root, archive.name, archive.sha256)
    (root / "identity.txt").write_text(label)


@pytest.fixture
def installation(tmp_path, monkeypatch):
    manager = module.PythonRuntimeManager(runtime_dir=str(tmp_path / "Hub 中文 & data"))
    target = Path(manager.private_runtime_root())
    bundle = tmp_path / "bundle"
    source = bundle / "python313"
    write_runtime(source, "new")
    monkeypatch.setattr(manager, "bundled_runtime_dirs", lambda: [str(bundle)])
    monkeypatch.setattr(manager, "_relocate_runtime_scripts", lambda *a: None, raising=False)
    return manager, target, source


def select_source(source_kind, manager, source, monkeypatch):
    if source_kind == "zip":
        with zipfile.ZipFile(source.parent / "runtime_bundle.zip", "w") as stream:
            for path in source.rglob("*"):
                if path.is_file():
                    stream.write(path, path.relative_to(source.parent).as_posix())
        shutil.rmtree(source)
    elif source_kind == "archive":
        archive = source.parent / "runtime.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(source, arcname="python")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive_info = runtime.RuntimeArchive(archive.name, "https://unused.invalid/runtime", digest)
        monkeypatch.setattr(module, "runtime_archive_for_machine", lambda **kw: archive_info)
        monkeypatch.setattr(manager, "_ensure_runtime_archive", lambda *a, **kw: str(archive))
        monkeypatch.setattr(manager, "bundled_runtime_dirs", lambda: [])


@pytest.mark.parametrize("source_kind", ["directory", "zip", "archive"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_complete_preparation_precedes_publication(installation, monkeypatch, source_kind, existing, success):
    manager, target, source = installation
    if existing:
        write_runtime(target, "old")
    select_source(source_kind, manager, source, monkeypatch)
    observed = []

    def prepare(python, *a, **kw):
        candidate = Path(runtime.runtime_prefix(python))
        observed.append(candidate)
        assert candidate != target
        assert manager.get_runtime_path() == (manager.private_runtime_python() if existing else None)
        if existing:
            assert (target / "identity.txt").read_text() == "old"
        (candidate / "support-package.txt").write_text("partially prepared")
        if not success:
            raise module.PythonRuntimeError("injected dependency installation failure")
        (candidate / "support-package.txt").write_text("complete")

    monkeypatch.setattr(manager, "_prepare_managed_runtime", prepare)
    if success:
        assert manager.reinstall_runtime() == manager.private_runtime_python()
        assert (target / "identity.txt").read_text() == "new"
        assert (target / "support-package.txt").read_text() == "complete"
    else:
        with pytest.raises(module.PythonRuntimeError, match="injected dependency"):
            manager.reinstall_runtime()
        assert target.exists() == existing
        if existing:
            assert (target / "identity.txt").read_text() == "old"
            assert not (target / "support-package.txt").exists()
    assert len(observed) == 1
    assert not list(target.parent.glob(".python313.extract-*"))


def test_ready_runtime_is_not_mutated_by_ensure(installation, monkeypatch):
    manager, target, _ = installation
    write_runtime(target, "old")
    monkeypatch.setattr(module, "_has_build_support", lambda *a: True)
    monkeypatch.setattr(manager, "_has_modules", lambda *a: True)
    monkeypatch.setattr(manager, "_prepare_managed_runtime", lambda *a, **kw: pytest.fail("ready runtime must remain untouched"))
    assert manager.ensure_runtime() == manager.private_runtime_python()


def test_external_pythonpath_cannot_satisfy_runtime_requirements(installation, monkeypatch, tmp_path):
    manager, _, source = installation
    external = tmp_path / "external-packages"
    external.mkdir()
    (external / "external_requirement_probe.py").write_text("VALUE = True\n")
    monkeypatch.setenv("PYTHONPATH", str(external))
    python = module._find_python_in_root(str(source))
    control = module._run_command([python, "-c", "import external_requirement_probe; print(1)"], timeout=15)
    assert control.returncode == 0 and control.stdout.strip() == "1"
    assert not manager._has_modules(python, "external_requirement_probe")


@pytest.mark.parametrize("frozen,repair", [(False, False), (True, False), (True, True)])
def test_incomplete_runtime_repair_is_transactional(installation, monkeypatch, frozen, repair):
    manager, target, _ = installation
    write_runtime(target, "old")
    monkeypatch.setattr(module, "is_frozen", lambda: frozen)
    monkeypatch.setattr(module, "_has_build_support", lambda *a: True)
    monkeypatch.setattr(manager, "_has_modules", lambda *a: False)
    observed = []
    def prepare(python, *a, **kw):
        assert Path(runtime.runtime_prefix(python)) != target
        observed.append(python)
        raise module.PythonRuntimeError("dependency installation failed")
    monkeypatch.setattr(manager, "_prepare_managed_runtime", prepare)
    with pytest.raises(module.PythonRuntimeError):
        manager.ensure_runtime(allow_frozen_repair=repair)
    assert len(observed) == (0 if frozen and not repair else 1)
    assert manager.get_runtime_path() == manager.private_runtime_python()
    assert (target / "identity.txt").read_text() == "old"


@pytest.mark.parametrize("phase", ["copy", "relocation"])
def test_bundle_failure_preserves_previous_runtime(installation, monkeypatch, phase):
    manager, target, _ = installation
    write_runtime(target, "old")
    monkeypatch.setattr(manager, "_prepare_managed_runtime", lambda *a, **kw: None)
    def fail(*a, **kw):
        raise OSError("injected " + phase)
    if phase == "copy":
        monkeypatch.setattr(module, "_copy_tree", fail)
        monkeypatch.setattr(module, "_copy_runtime_payload", fail, raising=False)
    else:
        monkeypatch.setattr(manager, "_relocate_runtime_scripts", fail, raising=False)
    with pytest.raises((OSError, module.PythonRuntimeError), match="injected " + phase):
        manager.reinstall_runtime()
    assert (target / "identity.txt").read_text() == "old"
    assert manager.get_runtime_path() == manager.private_runtime_python()


@pytest.mark.parametrize("member", ["python313/../outside", "python313/C:/outside", "python313/sub/../../outside"])
def test_bundle_rejects_noncanonical_paths_before_copy(installation, monkeypatch, member):
    manager, target, source = installation
    write_runtime(target, "old")
    shutil.rmtree(source)
    with zipfile.ZipFile(source.parent / "runtime_bundle.zip", "w") as stream:
        stream.writestr(member, "unexpected")
    with pytest.raises(module.PythonRuntimeError, match="path"):
        manager.reinstall_runtime()
    assert (target / "identity.txt").read_text() == "old"
    assert not (target.parent / "outside").exists()


@pytest.mark.parametrize("operation", ["install", "ensure", "copy"])
def test_runtime_operations_exclude_other_writers(installation, monkeypatch, operation):
    manager, target, _ = installation
    write_runtime(target, "old")
    competing = module.PythonRuntimeManager(runtime_dir=manager.installed_runtime_dir())
    monkeypatch.setattr(competing, "_provision_managed_runtime", lambda *a, **kw: pytest.fail("busy writer ran"))
    with manager._runtime_lock(manager._runtime_id()):
        with pytest.raises(module.PythonRuntimeError, match="busy"):
            if operation == "install":
                competing.reinstall_runtime()
            elif operation == "ensure":
                competing.ensure_runtime()
            else:
                competing.create_project_runtime(str(target.parent / "project-copy"))
    with competing._runtime_lock(competing._runtime_id()):
        assert (target / "identity.txt").read_text() == "old"


def test_os_lock_excludes_another_process_and_is_released_on_exit(installation):
    manager, target, _ = installation
    script = '''
import os, sys
sys.path.insert(0, sys.argv[1])
from embed_runtime_manager import PythonRuntimeManager, PythonRuntimeError
m = PythonRuntimeManager(runtime_dir=sys.argv[2])
try:
    with m._runtime_lock(m._runtime_id()):
        os._exit(0)
except PythonRuntimeError:
    sys.exit(17)
'''
    command = [sys.executable, "-I", "-c", script, str(Path(module.__file__).parent), manager.installed_runtime_dir()]
    def run():
        return subprocess.run(command, capture_output=True, timeout=20, creationflags=module._NO_WINDOW)
    with manager._runtime_lock(manager._runtime_id()):
        observed = run()
        assert observed.returncode == 17, observed.stderr
    observed = run()
    assert observed.returncode == 0, observed.stderr
    with manager._runtime_lock(manager._runtime_id()):
        pass


def test_project_copy_holds_source_lock_and_publishes_only_complete_copy(installation, monkeypatch):
    manager, target, _ = installation
    write_runtime(target, "old")
    monkeypatch.setattr(module, "_has_build_support", lambda *a: True)
    monkeypatch.setattr(manager, "_has_modules", lambda *a: True)
    competing = module.PythonRuntimeManager(runtime_dir=manager.installed_runtime_dir())
    destination = target.parent / "project/.runtime/python313"
    observed = []
    def copy(source, dest):
        assert not destination.exists()
        with pytest.raises(module.PythonRuntimeError, match="busy"):
            competing.reinstall_runtime()
        shutil.copytree(source, dest)
        observed.append(dest)
    monkeypatch.setattr(module, "_copy_project_runtime_tree", copy)
    installed = manager.create_project_runtime(str(destination))
    assert Path(installed).is_file()
    assert len(observed) == 1
    assert not Path(observed[0]).exists()
    assert (target / "identity.txt").read_text() == "old"


@pytest.mark.parametrize("phase", ["copy", "relocation"])
def test_failed_project_copy_never_publishes_partial_tree(installation, monkeypatch, phase):
    manager, target, _ = installation
    write_runtime(target, "old")
    monkeypatch.setattr(module, "_has_build_support", lambda *a: True)
    monkeypatch.setattr(manager, "_has_modules", lambda *a: True)
    destination = target.parent / "project/.runtime/python313"
    def fail(*a, **kw):
        raise OSError("copy unavailable")
    if phase == "copy":
        monkeypatch.setattr(module, "_copy_project_runtime_tree", fail)
    else:
        monkeypatch.setattr(manager, "_relocate_runtime_scripts", fail)
    with pytest.raises(module.PythonRuntimeError, match="copy unavailable"):
        manager.create_project_runtime(str(destination))
    assert not destination.exists()
    assert not list(destination.parent.glob(".python313.extract-*"))
    assert (target / "identity.txt").read_text() == "old"


def test_failed_robocopy_does_not_retry_using_another_copier(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.shutil, "which", lambda name: "robocopy")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda args, **kw:
                        calls.append(args) or subprocess.CompletedProcess(args, 8, stderr="locked"))
    monkeypatch.setattr(module.shutil, "copytree", lambda *a, **kw: pytest.fail("unexpected second copy path"))
    with pytest.raises(module.PythonRuntimeError, match="locked"):
        module._copy_tree(str(source), str(tmp_path / "dest"))
    assert len(calls) == 1
