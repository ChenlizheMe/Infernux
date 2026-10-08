"""Atomic Hub publication under actual Windows readers and bounded OS errors."""

from contextlib import contextmanager, ExitStack
import io
import json
import os
from types import SimpleNamespace
import zipfile

import pytest

import blender_support
import hub_utils
import version_manager


def windows_error(code):
    error = PermissionError("publication denied")
    error.winerror = code
    return error


@pytest.mark.parametrize("platform,code,release_after,attempts", [
    ("win32", None, 0, 1), ("win32", 5, 2, 3), ("win32", 32, 2, 3),
    ("win32", 33, 2, 3), ("win32", 5, None, 8),
    ("win32", 87, None, 1), ("win32", None, None, 1), ("linux", 5, None, 1),
])
def test_publication_waits_only_for_bounded_windows_occupancy(monkeypatch, platform, code, release_after, attempts):
    calls, waits = [], []
    error = windows_error(code)

    def replace(source, destination):
        calls.append((source, destination))
        if release_after is None or len(calls) <= release_after:
            raise error

    monkeypatch.setattr(hub_utils, "os", SimpleNamespace(replace=replace))
    monkeypatch.setattr(hub_utils, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(hub_utils, "time", SimpleNamespace(sleep=waits.append), raising=False)
    if release_after is None:
        with pytest.raises(PermissionError) as raised:
            hub_utils.replace_path("stage", "live")
        assert raised.value is error
    else:
        hub_utils.replace_path("stage", "live")
    assert calls == [("stage", "live")] * attempts
    assert len(waits) == attempts - 1 and sum(waits) <= 0.254


@pytest.mark.parametrize("directory", [False, True])
def test_prepared_file_or_directory_is_published_atomically(tmp_path, directory):
    stage, live = tmp_path / "stage", tmp_path / "live"
    if directory:
        stage.mkdir()
    candidate = stage / "content" if directory else stage
    candidate.write_bytes(b"complete generation")
    hub_utils.replace_path(stage, live)
    assert not stage.exists()
    assert (live / "content" if directory else live).read_bytes() == b"complete generation"


@contextmanager
def deny_delete(path, *, directory=False):
    """Hold a real Windows handle without FILE_SHARE_DELETE; caller owns it."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handle = create(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0x02000000 if directory else 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        if not close(handle):
            raise ctypes.WinError(ctypes.get_last_error())


@pytest.mark.skipif(os.name != "nt", reason="Windows FILE_SHARE_DELETE semantics")
@pytest.mark.parametrize("release", [False, True])
def test_actual_reader_keeps_atomic_source_and_target_intact_until_publication(tmp_path, monkeypatch, release):
    stage, live = tmp_path / "stage", tmp_path / "live"
    stage.write_bytes(b"complete candidate")
    live.write_bytes(b"previous complete generation")
    waits = []
    with ExitStack() as handles:
        handles.enter_context(deny_delete(live))

        def wait(seconds):
            waits.append(seconds)
            assert stage.read_bytes() == b"complete candidate"
            assert live.read_bytes() == b"previous complete generation"
            if release:
                handles.close()

        monkeypatch.setattr(hub_utils, "time", SimpleNamespace(sleep=wait), raising=False)
        if release:
            hub_utils.replace_path(stage, live)
            assert live.read_bytes() == b"complete candidate" and not stage.exists()
            assert waits
        else:
            with pytest.raises(PermissionError):
                hub_utils.replace_path(stage, live)
            assert len(waits) == 7
            assert stage.read_bytes() == b"complete candidate"
            assert live.read_bytes() == b"previous complete generation"


@pytest.mark.skipif(os.name != "nt", reason="Windows directory publication sharing semantics")
def test_blender_install_publishes_after_candidate_reader_releases(tmp_path, monkeypatch):
    archive = tmp_path / "blender.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("blender/blender.exe", b"candidate executable")
    manager = blender_support.BlenderSupportManager(tmp_path / "installed")
    validate = blender_support.validate_blender_support
    waits = []
    monkeypatch.delenv("INFERNUX_BLENDER_EXECUTABLE", raising=False)
    with ExitStack() as handles:
        def inspect_candidate(path):
            result = validate(path)
            if ".staging-" in str(path):
                handles.enter_context(deny_delete(path, directory=True))
            return result

        def release(seconds):
            waits.append(seconds)
            handles.close()

        monkeypatch.setattr(blender_support, "validate_blender_support", inspect_candidate)
        monkeypatch.setattr(hub_utils, "time", SimpleNamespace(sleep=release), raising=False)
        manager.install_archive(archive)
    assert waits and manager.status().installed
    assert manager.status().executable.read_bytes() == b"candidate executable"


@pytest.mark.skipif(os.name != "nt", reason="Windows cache replacement sharing semantics")
def test_engine_catalog_replaces_invalid_cache_after_reader_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(version_manager, "_VERSIONS_DIR", tmp_path / "Engines")
    manager = version_manager.VersionManager(SimpleNamespace())
    manager._cache_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(version_manager.urllib.request, "urlopen", lambda request, **_: io.BytesIO(
        json.dumps({"releases": {}} if "pypi.org" in request.full_url else []).encode()))
    waits = []
    with ExitStack() as handles:
        handles.enter_context(deny_delete(manager._cache_file))

        def release(seconds):
            waits.append(seconds)
            handles.close()

        monkeypatch.setattr(hub_utils, "time", SimpleNamespace(sleep=release), raising=False)
        assert manager._fetch_releases() == []
    assert waits and json.loads(manager._cache_file.read_text())["releases"] == []
    assert not list(manager._cache_file.parent.glob("*.tmp-*"))
