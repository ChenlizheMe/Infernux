"""Runtime acquisition must preserve the installed tree on failed replacement."""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import urllib.error

import pytest

import embed_runtime_manager as manager_module
import private_python_runtime as runtime


@pytest.fixture
def runtime_archive(tmp_path):
    archive = tmp_path / "incoming.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo("python/python.exe")
        payload = b"new runtime fixture, not an executable"
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    return archive


@pytest.fixture
def installed_tree(tmp_path):
    target = tmp_path / "中文 空格 & runtimes" / "python313"
    target.mkdir(parents=True)
    (target / "python.exe").write_bytes(b"previous interpreter")
    (target / "keep.txt").write_bytes(b"previous support package")
    return target


def assert_previous(target):
    assert (target / "python.exe").read_bytes() == b"previous interpreter"
    assert (target / "keep.txt").read_bytes() == b"previous support package"
    assert not (target / runtime.PRIVATE_RUNTIME_MARKER).exists()


@pytest.mark.parametrize("failure", ["marker", "old_rename", "new_rename"])
def test_archive_publication_failure_preserves_previous_tree(
    installed_tree, runtime_archive, monkeypatch, failure,
):
    archive = runtime_archive
    target = installed_tree
    replace = os.replace
    reached = []

    def fail_marker(*args, **kwargs):
        reached.append(True)
        raise OSError("injected marker failure")

    def fail_rename(source, destination):
        source, destination = Path(source), Path(destination)
        if ((failure == "old_rename" and source == target)
                or (failure == "new_rename" and source.name == "python" and destination == target)):
            reached.append(True)
            raise PermissionError("injected rename failure")
        return replace(source, destination)

    if failure == "marker":
        monkeypatch.setattr(runtime, "write_private_runtime_marker", fail_marker)
    else:
        monkeypatch.setattr(runtime.os, "replace", fail_rename)
    with pytest.raises(OSError, match="injected"):
        runtime.extract_runtime_archive(archive, target)

    assert reached == [True]
    assert_previous(target)
    assert list(target.parent.iterdir()) == [target]


def test_archive_failure_to_restore_preserves_recoverable_backup(
    installed_tree, runtime_archive, monkeypatch,
):
    archive = runtime_archive
    target = installed_tree
    replace = os.replace

    def deny_publication_and_restore(source, destination):
        if Path(destination) == target:
            raise PermissionError("target unavailable")
        return replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", deny_publication_and_restore)
    with pytest.raises(RuntimeError, match="previous runtime.*preserved"):
        runtime.extract_runtime_archive(archive, target)

    assert not target.exists()
    backups = list(target.parent.glob(".python313.extract-*/previous-runtime"))
    assert len(backups) == 1
    assert_previous(backups[0])


def test_archive_success_publishes_complete_marker_and_replaces_old_files(
    installed_tree, runtime_archive, monkeypatch,
):
    archive = runtime_archive
    target = installed_tree
    replace = os.replace
    published = []

    def observe_publication(source, destination):
        if Path(destination) == target:
            marker = json.loads((Path(source) / runtime.PRIVATE_RUNTIME_MARKER).read_text())
            assert marker["source_archive"] == archive.name
            published.append(True)
        return replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", observe_publication)
    runtime.extract_runtime_archive(archive, target)
    assert published == [True]
    assert (target / "python.exe").read_bytes() == b"new runtime fixture, not an executable"
    assert runtime.is_private_runtime_root(target)
    assert not (target / "keep.txt").exists()
    assert list(target.parent.iterdir()) == [target]


def test_archive_rejects_candidate_before_touching_installed_tree(
    installed_tree, runtime_archive,
):
    archive = runtime_archive
    checked = []

    def reject(candidate):
        checked.append(candidate)
        assert (candidate / "python.exe").is_file()
        assert runtime.is_private_runtime_root(candidate)
        assert_previous(installed_tree)
        raise RuntimeError("candidate rejected")

    with pytest.raises(RuntimeError, match="candidate rejected"):
        runtime.extract_runtime_archive(
            archive, installed_tree, validate=reject,
        )
    assert len(checked) == 1
    assert_previous(installed_tree)


@pytest.mark.parametrize("existing", [False, True])
def test_failed_first_or_repeat_extract_leaves_no_incomplete_runtime(
    tmp_path, runtime_archive, monkeypatch, existing,
):
    archive = runtime_archive
    target = tmp_path / "python313"
    if existing:
        target.mkdir()
        (target / "keep.txt").write_text("old")
    replace = os.replace

    def reject_candidate(source, destination):
        if Path(destination) == target and Path(source).name == "python":
            raise PermissionError("publication denied")
        return replace(source, destination)

    monkeypatch.setattr(runtime.os, "replace", reject_candidate)
    with pytest.raises(PermissionError, match="publication denied"):
        runtime.extract_runtime_archive(archive, target)
    assert target.exists() == existing
    if existing:
        assert (target / "keep.txt").read_text() == "old"
    assert not list(tmp_path.glob(".python313.extract-*"))


def test_reinstall_download_failure_keeps_real_interpreter_discoverable(tmp_path, monkeypatch):
    manager = manager_module.PythonRuntimeManager(runtime_dir=str(tmp_path / "runtime"))
    target = Path(manager.private_runtime_root())
    target.mkdir()
    executable = Path(manager.private_runtime_python())
    executable.parent.mkdir(exist_ok=True)
    # Uses the current Python executable and host stdlib, not a standalone release fixture.
    shutil.copy2(sys.executable, executable)
    if sys.platform == "win32":
        shutil.copy2(Path(sys.base_prefix) / "python313.dll", target / "python313.dll")
    (target / "pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).parent}\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    archive = runtime.runtime_archive_for_machine()
    runtime.write_private_runtime_marker(target, archive.name)
    (target / "keep.txt").write_text("old")
    assert manager.get_runtime_path() == str(executable)
    monkeypatch.setattr(manager, "bundled_runtime_dirs", lambda: [])
    requests = []

    def unavailable(*args, **kwargs):
        requests.append(args)
        raise urllib.error.HTTPError(archive.url, 404, "fixture unavailable", {}, None)

    monkeypatch.setattr(manager_module, "_download_file", unavailable)
    with pytest.raises(manager_module.PythonRuntimeError, match="Failed to download"):
        manager.reinstall_runtime()
    assert len(requests) == 1
    assert (target / "keep.txt").read_text() == "old"
    assert manager.get_runtime_path() == str(executable)


def test_invalid_executable_rejected_before_manager_replaces_old_tree(
    installed_tree, runtime_archive, monkeypatch,
):
    archive = runtime_archive
    manager = manager_module.PythonRuntimeManager(runtime_dir=str(installed_tree.parent))
    monkeypatch.setattr(manager, "_ensure_runtime_archive", lambda *a, **kw: str(archive))
    with pytest.raises(manager_module.PythonRuntimeError, match="valid full runtime"):
        manager._extract_runtime_to_root(str(installed_tree))
    assert_previous(installed_tree)


@pytest.mark.parametrize("operation", ["ensure_runtime", "reinstall_runtime"])
def test_custom_ca_reaches_download_through_transactional_install(
    tmp_path, runtime_archive, monkeypatch, operation,
):
    manager = manager_module.PythonRuntimeManager(runtime_dir=str(tmp_path / "managed"))
    monkeypatch.setattr(manager, "bundled_runtime_dirs", lambda: [])
    monkeypatch.setattr(manager, "_prepare_candidate", lambda *args, **kwargs: None)
    requests = []

    def download(url, destination, *, user_agent, ca_bundle):
        requests.append(ca_bundle)
        shutil.copyfile(runtime_archive, destination)

    monkeypatch.setattr(manager_module, "_download_file", download)
    installed = getattr(manager, operation)(download_ca_bundle="company-root.pem")
    assert installed == manager.private_runtime_python()
    assert requests == ["company-root.pem"]
    assert runtime.is_private_runtime_root(manager.private_runtime_root())
    assert not list(Path(manager.installed_runtime_dir()).glob(".*.extract-*"))
