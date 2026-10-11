"""Checkpoint commits preserve atomicity under Windows readers without delete sharing."""
from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from infernux.engine import filesystem
from infernux_mcp import checkpoints


@contextmanager
def reader(path, *, directory=False):
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                               wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.CreateFileW(str(path), 0x80000000, 3, None, 3, 0x02000000 if directory else 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        assert api.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows FILE_SHARE_DELETE publication contract")
def test_checkpoint_creation_waits_for_candidate_reader_and_publishes_one_complete_tree(tmp_path, monkeypatch):
    project = tmp_path / "Project"
    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    (project / "Assets/proof.txt").write_text("unchanged author data", encoding="utf-8")
    artifacts = project / ".infernux/checkpoint-test"
    write = checkpoints._write_json
    waits = []
    with ExitStack() as handles:
        def write_manifest(path, value):
            write(path, value)
            handles.enter_context(reader(Path(path).parent, directory=True))
        def release(seconds):
            waits.append(seconds)
            handles.close()
        monkeypatch.setattr(checkpoints, "_write_json", write_manifest)
        monkeypatch.setattr(filesystem, "time", SimpleNamespace(sleep=release))
        created = checkpoints.create_checkpoint(str(project), str(artifacts), "before", session_id="fixture")
    assert waits
    assert checkpoints.load_checkpoint(str(project), str(artifacts), "before", session_id="fixture")["ledger"] == created["ledger"]
    assert (project / "Assets/proof.txt").read_text() == "unchanged author data"
    assert not list((artifacts / "checkpoints").glob(".*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="Windows FILE_SHARE_DELETE publication contract")
@pytest.mark.parametrize("release_reader", [False, True])
def test_checkpoint_manifest_publication_keeps_previous_bytes_until_atomic_replace(tmp_path, monkeypatch, release_reader):
    path = tmp_path / "manifest.json"
    path.write_text('{"generation": 1}', encoding="utf-8")
    waits = []
    with ExitStack() as handles:
        handles.enter_context(reader(path))
        def wait(seconds):
            waits.append(seconds)
            assert json.loads(path.read_text()) == {"generation": 1}
            if release_reader:
                handles.close()
        monkeypatch.setattr(filesystem, "time", SimpleNamespace(sleep=wait))
        if release_reader:
            checkpoints._write_json(str(path), {"generation": 2})
            assert waits and json.loads(path.read_text()) == {"generation": 2}
        else:
            with pytest.raises(PermissionError):
                checkpoints._write_json(str(path), {"generation": 2})
            assert len(waits) == 7 and json.loads(path.read_text()) == {"generation": 1}
    assert list(tmp_path.iterdir()) == [path]
