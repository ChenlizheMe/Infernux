"""Actual Linux change events remain authoritative when metadata collides."""
import gc
import importlib.util
import multiprocessing
import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux inotify contract")


@pytest.fixture(scope="module")
def observation():
    path = Path(__file__).resolve().parents[2] / "python/infernux/core/_linux_file_observation.py"
    spec = importlib.util.spec_from_file_location("linux_observation_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("operation", ["overwrite", "replace", "hardlink"])
def test_change_events_do_not_depend_on_timestamp_resolution(tmp_path, monkeypatch, observation, operation):
    path = tmp_path / "observed.txt"
    path.write_text("first")
    probe = observation.FileProbe(str(path))
    first = probe()
    assert probe() == first
    metadata = path.stat()
    actual_stat = os.stat

    def colliding_stat(name, *args, **kwargs):
        if os.fspath(name) == str(path):
            return metadata
        return actual_stat(name, *args, **kwargs)

    # File events come from the real kernel. Only the observed stat is fixed,
    # proving that equal timestamps/size cannot hide a change.
    monkeypatch.setattr(os, "stat", colliding_stat)
    if operation == "replace":
        replacement = tmp_path / "replacement"
        replacement.write_text("other")
        replacement.replace(path)
    elif operation == "hardlink":
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        linked = elsewhere / "same-inode"
        os.link(path, linked)
        linked.write_text("other")
    else:
        path.write_text("other")
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    changed = probe()
    assert changed != first
    assert changed.metadata == first.metadata
    assert probe() == changed


def test_unrelated_parent_writes_do_not_reparse_a_file(tmp_path, observation):
    path = tmp_path / "observed.txt"
    path.write_text("first")
    probe = observation.FileProbe(str(path))
    first = probe()
    (tmp_path / "unrelated.txt").write_text("other")
    assert probe() == first


def test_deleted_and_recreated_path_is_observed(tmp_path, observation):
    path = tmp_path / "observed.txt"
    path.write_text("first")
    probe = observation.FileProbe(str(path))
    first = probe()
    path.unlink()
    assert probe() is None
    path.write_text("other")
    assert probe() != first


def test_evicted_observers_release_kernel_watches(tmp_path, observation):
    def watches():
        return sum(line.startswith("inotify wd:") for line in
                   Path(f"/proc/self/fdinfo/{observation._events.fd}").read_text().splitlines())

    baseline = watches()
    probes = []
    for index in range(128):
        path = tmp_path / str(index)
        path.write_text("data")
        probes.append(observation.FileProbe(str(path)))
        probes[-1]()
    assert watches() > baseline
    probes.clear()
    gc.collect()
    assert watches() == baseline


def test_forked_reader_cannot_consume_parent_change_events(tmp_path, observation):
    path = tmp_path / "shared.txt"
    path.write_text("first")
    probe = observation.FileProbe(str(path))
    first = probe()
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()

    def read_in_child():
        initial = probe()
        child.send("ready")
        child.recv()
        child.send(probe() != initial)
        child.close()

    worker = context.Process(target=read_in_child)
    worker.start()
    assert parent.poll(5) and parent.recv() == "ready"
    path.write_text("other")
    parent.send("written")
    assert parent.poll(5) and parent.recv() is True
    worker.join(5)
    assert worker.exitcode == 0
    assert probe() != first
    parent.close()
