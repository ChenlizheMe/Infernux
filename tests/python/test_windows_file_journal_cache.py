"""Read models distinguish writes and tree edits despite timestamp collisions."""
import ctypes
import os

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows file journal contract")


def freeze_times(monkeypatch):
    from infernux.core import _windows_file_observation as windows

    query = windows._query

    def frozen(handle, destination):
        result = query(handle, destination)
        if result:
            info = ctypes.cast(destination, ctypes.POINTER(windows._FileInformation)).contents
            info.CreationTime.dwLowDateTime = info.CreationTime.dwHighDateTime = 0
            info.LastWriteTime.dwLowDateTime = info.LastWriteTime.dwHighDateTime = 0
        return result

    monkeypatch.setattr(windows, "_query", frozen)


def test_repeated_same_length_writes_are_visible_with_identical_timestamps(tmp_path, monkeypatch):
    from infernux.core.file_read_cache import FileReadCache

    freeze_times(monkeypatch)
    path = tmp_path / "快速写入.txt"
    path.write_text("0000", encoding="utf-8")
    cache = FileReadCache()

    def prepare(observed):
        observed.watch(path)
        return path.read_text(encoding="utf-8")

    assert cache.get("text", prepare) == "0000"
    for index in range(1, 25):
        value = f"{index:04}"
        path.write_text(value, encoding="utf-8")
        assert cache.get("text", prepare) == value


def test_directory_names_observe_creation_rename_and_removal(tmp_path, monkeypatch):
    from infernux.core import _windows_file_observation as windows
    from infernux.core.file_read_cache import FileReadCache

    freeze_times(monkeypatch)
    # Directory USNs are not sufficient to observe changes to child entries.
    monkeypatch.setattr(windows, "_journal_revision", lambda handle: None)
    cache = FileReadCache()

    def prepare(observed):
        observed.watch(tmp_path)
        return sorted(path.name for path in tmp_path.iterdir())

    assert cache.get("children", prepare) == []
    first = tmp_path / "first.txt"
    first.write_text("value", encoding="utf-8")
    assert cache.get("children", prepare) == ["first.txt"]
    second = first.with_name("second.txt")
    first.rename(second)
    assert cache.get("children", prepare) == ["second.txt"]
    second.unlink()
    assert cache.get("children", prepare) == []


@pytest.mark.parametrize("error_code", [1, 50, 1178, 1179])
def test_files_without_a_journal_do_not_reuse_a_previous_read(tmp_path, monkeypatch, error_code):
    from infernux.core import _windows_file_observation as windows
    from infernux.core.file_read_cache import FileReadCache

    def unsupported(*args):
        ctypes.set_last_error(error_code)
        return False

    monkeypatch.setattr(windows, "_ioctl", unsupported)
    path = tmp_path / "remote.txt"
    path.write_text("first", encoding="utf-8")
    cache = FileReadCache()
    calls = []

    def prepare(observed):
        observed.watch(path)
        calls.append(1)
        return path.read_text(encoding="utf-8")

    assert cache.get("text", prepare) == "first"
    assert cache.get("text", prepare) == "first"
    assert len(calls) == 2
    path.write_text("other", encoding="utf-8")
    assert cache.get("text", prepare) == "other"


def test_journal_permission_errors_are_not_hidden(tmp_path, monkeypatch):
    from infernux.core import _windows_file_observation as windows
    from infernux.core.file_read_cache import FileReadCache

    def denied(*args):
        ctypes.set_last_error(5)
        return False

    monkeypatch.setattr(windows, "_ioctl", denied)
    path = tmp_path / "denied.txt"
    path.write_text("value", encoding="utf-8")
    cache = FileReadCache()

    def prepare(observed):
        observed.watch(path)
        return path.read_text(encoding="utf-8")

    with pytest.raises(OSError) as error:
        cache.get("text", prepare)
    assert error.value.winerror == 5
