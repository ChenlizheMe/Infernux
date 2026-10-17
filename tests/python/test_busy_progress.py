"""The progress helper is a presentation boundary, not a Play prerequisite."""

import subprocess

import pytest

from infernux.engine.ui.busy_progress import BusyProgress


class _DeadStdin:
    def write(self, _value):
        raise BrokenPipeError("helper exited")

    def flush(self):
        raise BrokenPipeError("helper exited")

    def close(self):
        raise BrokenPipeError("helper exited")


class _DeadProcess:
    stdin = _DeadStdin()

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


def test_update_degrades_after_the_helper_exits():
    progress = BusyProgress.__new__(BusyProgress)
    progress._process = _DeadProcess()

    progress.update(0.5, "Preparing")
    assert progress._process is None

    # Later updates and close are no-ops instead of raising into Play Mode.
    progress.update(0.6, "Still preparing")
    progress.close()


def test_missing_executable_surfaces_as_oserror(monkeypatch):
    def _missing(*_args, **_kwargs):
        raise OSError("helper executable is missing")

    monkeypatch.setattr(subprocess, "Popen", _missing)
    with pytest.raises(OSError):
        BusyProgress()
