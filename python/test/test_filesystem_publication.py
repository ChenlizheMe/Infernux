from types import SimpleNamespace

import pytest

from Infernux.engine import filesystem


def setup_publication(monkeypatch, *, platform="win32", errors=()):
    calls, waits = [], []
    pending = iter(errors)

    def rename(source, destination):
        calls.append((source, destination))
        error = next(pending, None)
        if error is not None:
            raise error

    monkeypatch.setattr(filesystem, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(filesystem, "os", SimpleNamespace(replace=rename))
    monkeypatch.setattr(filesystem, "time", SimpleNamespace(sleep=waits.append))
    return calls, waits


def windows_error(code):
    error = PermissionError("injected Windows publication failure")
    error.winerror = code
    return error


def test_normal_publication_is_one_rename_without_wait(monkeypatch):
    calls, waits = setup_publication(monkeypatch)
    filesystem.replace_path("stage", "product")
    assert calls == [("stage", "product")]
    assert waits == []


@pytest.mark.parametrize("code", [5, 32, 33])
def test_windows_temporary_occupancy_uses_same_rename(monkeypatch, code):
    calls, waits = setup_publication(monkeypatch, errors=(windows_error(code), windows_error(code)))
    filesystem.replace_path("stage", "product")
    assert calls == [("stage", "product")] * 3
    assert waits == [0.002, 0.004]


def test_permanent_windows_failure_is_bounded_and_preserves_exception(monkeypatch):
    error = windows_error(5)
    calls, waits = setup_publication(monkeypatch, errors=[error] * 8)
    with pytest.raises(PermissionError) as raised:
        filesystem.replace_path("stage", "product")
    assert raised.value is error
    assert len(calls) == 8 and sum(waits) == pytest.approx(0.254)


@pytest.mark.parametrize("platform,error", [("linux", windows_error(5)),
                                              ("win32", FileNotFoundError("missing")),
                                              ("win32", windows_error(17))])
def test_other_platforms_and_errors_do_not_retry(monkeypatch, platform, error):
    calls, waits = setup_publication(monkeypatch, platform=platform, errors=(error,))
    with pytest.raises(type(error)) as raised:
        filesystem.replace_path("stage", "product")
    assert raised.value is error
    assert len(calls) == 1 and not waits


@pytest.mark.parametrize('directory', [False, True])
def test_real_file_and_directory_publication_share_one_policy(tmp_path, monkeypatch, directory):
    source, destination = tmp_path / 'stage', tmp_path / 'product'
    if directory:
        source.mkdir()
        payload = source / 'content.bin'
    else:
        payload = source
    payload.write_bytes(b'complete generation')
    native_replace = filesystem.os.replace
    native_sleep = filesystem.time.sleep
    calls, waits = [], []

    def temporarily_locked(old, new):
        calls.append((old, new))
        if len(calls) == 1:
            raise windows_error(5)
        native_replace(old, new)

    def wait(seconds):
        waits.append(seconds)
        native_sleep(seconds)  # The real directory can also be held by a Windows scanner.

    monkeypatch.setattr(filesystem, 'os', SimpleNamespace(replace=temporarily_locked))
    monkeypatch.setattr(filesystem, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(filesystem, 'time', SimpleNamespace(sleep=wait))
    filesystem.replace_path(source, destination)
    published = destination / 'content.bin' if directory else destination
    assert published.read_bytes() == b'complete generation'
    assert not source.exists()
    assert 2 <= len(calls) <= 8
    assert calls == [(source, destination)] * len(calls)
    assert waits == [.002 * 2 ** attempt for attempt in range(len(calls) - 1)]
