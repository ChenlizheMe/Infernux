"""Permanent Windows installations must not retain tempfile's protected ACL."""

import ctypes
from ctypes import wintypes
import json
import stat
import subprocess
from pathlib import Path
import sys
import tempfile
import zipfile

import pytest

from installer.install_application import HubInstallTransaction
from installer.payload import HUB_EXECUTABLE
from installer import install_python_runtime
from installer import install_application
from installer_safety import WINDOWS_ACCESS_LAYOUT, install_marker_path, write_install_marker
from private_python_runtime import runtime_publication
from publication_directory import clear_windows_readonly_tree, restore_windows_inherited_access_tree


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows inherited access control")


def test_install_directory_waits_for_transient_windows_handle(tmp_path, monkeypatch):
    source, target = tmp_path / "stage", tmp_path / "live"
    source.mkdir()
    (source / "payload").write_bytes(b"complete")
    native_replace = install_application.os.replace
    calls = []

    def replace(before, after):
        calls.append((before, after))
        if len(calls) == 1:
            error = PermissionError("directory briefly occupied")
            error.winerror = 32
            raise error
        return native_replace(before, after)

    monkeypatch.setattr(install_application.os, "replace", replace)
    monkeypatch.setattr(install_application.time, "sleep", lambda delay: None)
    install_application._replace_install_directory(source, target)
    assert calls == [(source, target), (source, target)]
    assert (target / "payload").read_bytes() == b"complete"


def test_install_directory_access_failure_is_bounded(tmp_path, monkeypatch):
    source, target = tmp_path / "stage", tmp_path / "live"
    source.mkdir()
    (source / "payload").write_bytes(b"retained")
    clock = iter([0.0, 0.1, 10.0])
    calls = []

    def denied(before, after):
        calls.append((before, after))
        error = PermissionError("access permanently denied")
        error.winerror = 5
        raise error

    monkeypatch.setattr(install_application.os, "replace", denied)
    monkeypatch.setattr(install_application.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(install_application.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError, match="permanently denied"):
        install_application._replace_install_directory(source, target)
    assert len(calls) == 2
    assert not target.exists()
    assert (source / "payload").read_bytes() == b"retained"


def protected_dacl(path):
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_security = advapi.GetNamedSecurityInfoW
    get_security.argtypes = [wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    get_security.restype = wintypes.DWORD
    control = wintypes.WORD()
    revision = wintypes.DWORD()
    descriptor = ctypes.c_void_p()
    advapi.GetSecurityDescriptorControl.argtypes = [ctypes.c_void_p,
        ctypes.POINTER(wintypes.WORD), ctypes.POINTER(wintypes.DWORD)]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    result = get_security(str(path), 1, 4, None, None, None, None, ctypes.byref(descriptor))
    if result:
        raise ctypes.WinError(result)
    try:
        if not advapi.GetSecurityDescriptorControl(descriptor, ctypes.byref(control), ctypes.byref(revision)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(control.value & 0x1000)
    finally:
        kernel.LocalFree(descriptor)


def restrict_fixture_to_owner_acl_control(path):
    """Keep repair rights but deny data access to the ordinary test token."""
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    get_dacl = advapi.GetSecurityDescriptorDacl
    get_dacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.BOOL)]
    get_dacl.restype = wintypes.BOOL
    set_security = advapi.SetNamedSecurityInfoW
    set_security.argtypes = [wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    set_security.restype = wintypes.DWORD
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor, dacl = ctypes.c_void_p(), ctypes.c_void_p()
    # READ_CONTROL | WRITE_DAC | FILE_READ_ATTRIBUTES, without file-data access.
    sddl = "D:P(A;OICI;0x60080;;;OW)"
    if not convert(sddl, 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        present, defaulted = wintypes.BOOL(), wintypes.BOOL()
        if not get_dacl(descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)):
            raise ctypes.WinError(ctypes.get_last_error())
        result = set_security(str(path), 1, 4 | 0x80000000, None, None, dacl, None)
        if result:
            raise ctypes.WinError(result)
    finally:
        kernel.LocalFree(descriptor)


def dacl_snapshot(path):
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    get_security = advapi.GetNamedSecurityInfoW
    get_security.argtypes = [wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    get_security.restype = wintypes.DWORD
    convert = advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW
    convert.argtypes = [ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                        ctypes.POINTER(wintypes.LPWSTR), ctypes.c_void_p]
    convert.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    descriptor, sddl = ctypes.c_void_p(), wintypes.LPWSTR()
    result = get_security(str(path), 1, 4, None, None, None, None, ctypes.byref(descriptor))
    if result:
        raise ctypes.WinError(result)
    try:
        if not convert(descriptor, 1, 4, ctypes.byref(sddl), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return sddl.value
    finally:
        kernel.LocalFree(sddl)
        kernel.LocalFree(descriptor)


def test_published_hub_inherits_install_parent_access(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / HUB_EXECUTABLE).write_bytes(b"test payload")
    target = tmp_path / "Installed Hub"
    with HubInstallTransaction(payload, target) as transaction:
        transaction.prepare()
        assert not protected_dacl(transaction.staged_dir)
        transaction.activate()
        transaction.commit()
    assert not protected_dacl(target)
    assert (target / HUB_EXECUTABLE).read_bytes() == b"test payload"


def test_runtime_candidate_inherits_shared_storage_access(tmp_path):
    target = tmp_path / "Shared/Runtimes/python313"
    with runtime_publication(target) as candidate:
        assert not protected_dacl(candidate.parent)
        candidate.mkdir()
        (candidate / "python.exe").write_bytes(b"runtime")
        assert not protected_dacl(candidate)
    assert not protected_dacl(target)
    (target / "write-probe").write_text("ordinary user writable")


@pytest.mark.parametrize("previous_layout", [None, 1, 2])
def test_legacy_update_repairs_native_acl_once(tmp_path, monkeypatch, previous_layout):
    old_install = Path(tempfile.mkdtemp(prefix="legacy-hub-", dir=tmp_path))
    assert protected_dacl(old_install)
    write_install_marker(str(old_install))
    marker = Path(install_marker_path(str(old_install)))
    document = json.loads(marker.read_text(encoding="utf-8"))
    del document["access_layout"]
    if previous_layout is not None:
        document["access_layout"] = previous_layout
    marker.write_text(json.dumps(document), encoding="utf-8")
    shared = old_install / "InfernuxHubData/Shared"
    runtimes = shared / "Runtimes"
    runtimes.mkdir(parents=True)
    # Old runtime publication also renamed a private tempfile directory into
    # place. Its independent protected DACL survives repairing the app root.
    runtime = Path(tempfile.mkdtemp(prefix="old-runtime-", dir=runtimes))
    runtime.rename(runtimes / "python313")
    retained = runtimes / "python313/keep.txt"
    assert protected_dacl(retained.parent)
    retained.write_text("existing runtime", encoding="utf-8")
    retained.chmod(stat.S_IREAD)
    old_install.chmod(stat.S_IREAD)
    # The fixture is owned by this test user, allowing real ACL changes without
    # a UAC prompt. The production migration runs only in the elevated updater.
    monkeypatch.setattr(install_python_runtime.ctypes.windll.shell32, "IsUserAnAdmin", lambda: True)
    assert install_python_runtime.repair_legacy_windows_permissions(str(old_install))
    assert not protected_dacl(old_install)
    assert not protected_dacl(retained.parent)
    assert retained.read_text(encoding="utf-8") == "existing runtime"
    retained.write_text("ordinary writes work after repair", encoding="utf-8")
    assert not old_install.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
    assert json.loads(marker.read_text(encoding="utf-8"))["access_layout"] == WINDOWS_ACCESS_LAYOUT
    stamp = marker.stat().st_mtime_ns
    assert not install_python_runtime.repair_legacy_windows_permissions(str(old_install))
    assert marker.stat().st_mtime_ns == stamp


def test_ordinary_launch_does_not_attempt_legacy_elevation(tmp_path, monkeypatch):
    write_install_marker(str(tmp_path))
    marker = Path(install_marker_path(str(tmp_path)))
    document = json.loads(marker.read_text(encoding="utf-8"))
    document.pop("access_layout")
    marker.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(install_python_runtime.ctypes.windll.shell32, "IsUserAnAdmin", lambda: False)
    assert not install_python_runtime.repair_legacy_windows_permissions(str(tmp_path))
    assert json.loads(marker.read_text(encoding="utf-8")) == document


def test_replacing_readonly_live_runtime_with_ordinary_token(tmp_path):
    target = tmp_path / "Shared/Runtimes/python313"
    target.mkdir(parents=True)
    previous = target / "python.exe"
    previous.write_bytes(b"old runtime")
    previous.chmod(stat.S_IREAD)
    target.chmod(stat.S_IREAD)
    try:
        with runtime_publication(target) as candidate:
            candidate.mkdir()
            (candidate / "python.exe").write_bytes(b"new runtime")
            (candidate / "python.exe").chmod(stat.S_IREAD)
            candidate.chmod(stat.S_IREAD)
        assert (target / "python.exe").read_bytes() == b"new runtime"
        assert not target.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
        (target / "python.exe").write_bytes(b"ordinary user writable")
        assert not list(target.parent.glob(".python313.extract-*"))
    finally:
        clear_windows_readonly_tree(tmp_path)


@pytest.mark.parametrize("path_length", [140, 280])
def test_shared_repair_restores_actual_denied_runtime_writes(tmp_path, path_length):
    # Exercise an actual access denial with the ordinary test process, rather
    # than mistaking Explorer's directory checkbox for an access check.
    app = tmp_path / "Hub 中文"
    while len(str(app)) < path_length:
        app /= "nested-" + "x" * 30
    runtime = app / "InfernuxHubData/Shared/Runtimes/python313"
    runtime.mkdir(parents=True)
    retained = runtime / "python.exe"
    retained.write_bytes(b"existing runtime")
    # Keep owner READ_CONTROL/WRITE_DAC/READ_ATTRIBUTES so the test user can
    # repair its own fixture, but remove all file-data read/write permissions.
    try:
        for path in (runtime, retained):
            restrict_fixture_to_owner_acl_control(path)
        assert protected_dacl(runtime)
        assert not runtime.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
        with pytest.raises(PermissionError):
            retained.write_bytes(b"must be denied")
        install_python_runtime.prepare_shared_storage(str(app))
        assert not protected_dacl(runtime)
        assert retained.read_bytes() == b"existing runtime"
        retained.write_bytes(b"ordinary user write")
        probe = runtime / "probe"
        probe.write_bytes(b"new file")
        probe.rename(runtime / "renamed-probe")
        (runtime / "renamed-probe").unlink()
        with runtime_publication(runtime) as candidate:
            candidate.mkdir()
            (candidate / "python.exe").write_bytes(b"replacement runtime")
        assert retained.read_bytes() == b"replacement runtime"
        assert not list(runtime.parent.glob(".python313.extract-*"))
    finally:
        restore_windows_inherited_access_tree(runtime)


@pytest.mark.parametrize("archive_payload", [False, True])
def test_installer_publishes_writable_files_from_readonly_payload(tmp_path, archive_payload):
    from installer.payload import HUB_PAYLOAD_ARCHIVE

    payload = tmp_path / "payload"
    payload.mkdir()
    if archive_payload:
        with zipfile.ZipFile(payload / HUB_PAYLOAD_ARCHIVE, "w") as archive:
            for name in [HUB_EXECUTABLE, "resources/settings.json"]:
                entry = zipfile.ZipInfo(name)
                entry.create_system = 3
                entry.external_attr = (stat.S_IFREG | 0o444) << 16 | 1
                archive.writestr(entry, b"payload")
    else:
        executable = payload / HUB_EXECUTABLE
        executable.write_bytes(b"payload")
        executable.chmod(stat.S_IREAD)
        payload.chmod(stat.S_IREAD)
    target = tmp_path / "Installed Hub"
    target.mkdir()
    write_install_marker(str(target))
    previous_executable = target / HUB_EXECUTABLE
    previous_executable.write_bytes(b"old payload")
    previous_executable.chmod(stat.S_IREAD)
    target.chmod(stat.S_IREAD)
    try:
        with HubInstallTransaction(payload, target) as transaction:
            transaction.prepare()
            transaction.activate()
            transaction.commit()
        assert not target.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
        executable = target / HUB_EXECUTABLE
        executable.write_bytes(b"replacement")
        executable.unlink()
        assert not list(tmp_path.glob(".Installed Hub.backup-*"))
    finally:
        clear_windows_readonly_tree(payload)


@pytest.mark.parametrize("external_protected", [False, True])
def test_shared_repair_clears_readonly_and_preserves_external_junction(tmp_path, external_protected):
    shared = tmp_path / "Hub/InfernuxHubData/Shared"
    retained = shared / "Runtimes/python313/keep.txt"
    retained.parent.mkdir(parents=True)
    retained.write_text("retained")
    retained.chmod(stat.S_IREAD)
    external = (Path(tempfile.mkdtemp(prefix="external-", dir=tmp_path))
                if external_protected else tmp_path / "external")
    external.mkdir(exist_ok=True)
    untouched = external / "outside.txt"
    untouched.write_text("external")
    untouched.chmod(stat.S_IREAD)
    external_access = dacl_snapshot(external), dacl_snapshot(untouched)
    link = shared / "external-link"
    subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(external)],
                   check=True, capture_output=True, creationflags=0x08000000)
    try:
        install_python_runtime.prepare_shared_storage(str(tmp_path / "Hub"))
        retained.write_text("ordinary user replacement")
        assert (dacl_snapshot(external), dacl_snapshot(untouched)) == external_access
        assert untouched.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY
    finally:
        link.rmdir()
        untouched.chmod(stat.S_IREAD | stat.S_IWRITE)
