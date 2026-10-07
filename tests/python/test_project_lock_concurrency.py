"""The Hub and both engine hosts share one OS-serialized project lease."""
import builtins
import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest


def _contender(project, kind, token, ready, start, release, legacy_writers):
    import infernux.engine as engine_entry
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
    import hub_utils

    lock = str(Path(project) / "ProjectSettings/.infernux-engine-lock.json")
    original_open = builtins.open

    def scheduled_open(path, mode="r", *args, **kwargs):
        # Reproduce the old legal schedule: both processes passed their
        # existence checks before either overwrites the canonical lock file.
        # Atomic implementations write a private temp file and never enter it.
        if os.fspath(path) == lock and mode == "w":
            legacy_writers.wait(timeout=15)
        return original_open(path, mode, *args, **kwargs)

    os.environ.pop("_INFERNUX_PROJECT_LOCK_PATH", None)
    os.environ["_INFERNUX_PROJECT_LOCK_TOKEN"] = token
    acquired = False
    builtins.open = scheduled_open
    ready.put(("ready", token, None))
    try:
        assert start.wait(20)
        try:
            if kind == "hub":
                hub_utils.write_project_lock(project, os.getpid(), token, "editor", "preparing")
            else:
                engine_entry._acquire_project_lock(project, kind)
            acquired = True
            ready.put(("acquired", token, os.getpid()))
        except Exception as error:
            ready.put(("rejected", token, type(error).__name__ + ": " + str(error)))
        assert release.wait(20)
    finally:
        builtins.open = original_open
        if acquired:
            if kind == "hub":
                hub_utils.remove_project_lock(project, token)
            else:
                engine_entry._remove_project_lock(lock, token)


@pytest.mark.parametrize("kinds", [("editor", "headless"), ("hub", "hub"), ("hub", "headless")])
def test_only_one_process_can_claim_a_project(tmp_path, kinds):
    project = tmp_path / "中文协作 & project"
    (project / "ProjectSettings").mkdir(parents=True)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    start, release = context.Event(), context.Event()
    writers = context.Barrier(2)
    children = [context.Process(target=_contender, args=(str(project), kind, str(index), results,
                                                       start, release, writers))
                for index, kind in enumerate(kinds)]
    try:
        for child in children:
            child.start()
        assert all(results.get(timeout=30)[0] == "ready" for _ in children)
        start.set()
        outcomes = [results.get(timeout=30) for _ in children]
        assert [item[0] for item in outcomes].count("acquired") == 1, outcomes
        rejected, = [item for item in outcomes if item[0] == "rejected"]
        assert rejected[2].startswith("RuntimeError:"), outcomes
        assert "already open" in rejected[2], outcomes
    finally:
        release.set()
        for child in children:
            child.join(30)
            if child.is_alive():
                child.terminate()
                child.join(10)
        results.close()
        results.join_thread()
    assert all(child.exitcode == 0 for child in children)


def _handoff_child(project, token, ready, start, release, abrupt):
    from infernux import engine
    os.environ["_INFERNUX_PROJECT_LOCK_PATH"] = engine._default_lock_path(project)
    os.environ["_INFERNUX_PROJECT_LOCK_TOKEN"] = token
    ready.put("ready")
    assert start.wait(20)
    path, owned = engine._acquire_project_lock(project, "editor")
    ready.put("acquired")
    assert release.wait(20)
    if abrupt:
        ready.close()
        ready.join_thread()
        os._exit(0)
    engine._remove_project_lock(path, owned)


@pytest.mark.parametrize("transfer_first", [False, True])
@pytest.mark.parametrize("abrupt", [False, True])
def test_hub_handoff_live_owner_protection_and_exit_recovery(tmp_path, transfer_first, abrupt):
    import infernux_project_lock as locks
    project = str(tmp_path / "交接 & project")
    token = "one-use-reservation"
    path = locks.reserve(project, os.getpid(), token, "editor", "preparing")
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    start, release = context.Event(), context.Event()
    child = context.Process(target=_handoff_child, args=(project, token, results, start, release, abrupt))
    try:
        child.start()
        assert results.get(timeout=30) == "ready"
        if transfer_first:
            locks.reserve(project, child.pid, token, "editor", "launching")
        start.set()
        assert results.get(timeout=30) == "acquired"
        running = Path(path).read_bytes()
        assert json.loads(running)["pid"] == child.pid
        assert json.loads(running)["state"] == "running"
        locks.reserve(project, child.pid, token, "editor", "launching")
        assert Path(path).read_bytes() == running
        assert not locks.remove_lock(path, token)
        assert not locks.remove_lock(path, None)
        with pytest.raises(RuntimeError, match="already open"):
            locks.claim(project, token, "headless")
        with pytest.raises(RuntimeError, match="already open"):
            locks.reserve(project, os.getpid(), "other-hub", "editor", "preparing")
        assert Path(path).read_bytes() == running
    finally:
        release.set()
        child.join(30)
        if child.is_alive():
            child.terminate()
            child.join(10)
        results.close()
        results.join_thread()
    assert child.exitcode == 0
    assert Path(path).exists() == abrupt
    guard = Path(project) / locks.GUARD_NAME
    identity = guard.stat().st_ino
    assert locks.claim(project, "next-owner", "headless") == path
    assert guard.stat().st_ino == identity
    assert locks.remove_lock(path, "next-owner")


@pytest.mark.parametrize("prior", [False, True])
def test_failed_publication_preserves_previous_lease_and_releases_guard(tmp_path, monkeypatch, prior):
    import infernux_project_lock as locks
    project = str(tmp_path)
    path = Path(locks.lock_path(project))
    before = None
    if prior:
        locks.reserve(project, os.getpid(), "owned", "editor", "preparing")
        before = path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(locks.os, "replace", lambda *_: (_ for _ in ()).throw(PermissionError("publication denied")))
        with pytest.raises(PermissionError, match="publication denied"):
            locks.reserve(project, os.getpid(), "owned", "editor", "preparing")
    assert (path.read_bytes() if path.exists() else None) == before
    assert not list(path.parent.glob(".infernux-engine-lock.*.tmp"))
    locks.reserve(project, os.getpid(), "owned", "editor", "preparing")
    assert locks.remove_lock(str(path), "owned")


def test_cancelled_hub_reservation_cannot_be_claimed_as_a_fresh_launch(tmp_path, monkeypatch):
    from infernux import engine
    monkeypatch.setenv("_INFERNUX_PROJECT_LOCK_PATH", engine._default_lock_path(str(tmp_path)))
    monkeypatch.setenv("_INFERNUX_PROJECT_LOCK_TOKEN", "cancelled")
    with pytest.raises(RuntimeError, match="reservation is no longer available"):
        engine._acquire_project_lock(str(tmp_path), "editor")
    assert not Path(engine._default_lock_path(str(tmp_path))).exists()


def test_lock_path_override_cannot_bypass_project_exclusion(tmp_path, monkeypatch):
    from infernux import engine
    monkeypatch.setenv("_INFERNUX_PROJECT_LOCK_PATH", str(tmp_path / "different.lock"))
    with pytest.raises(ValueError, match="canonical lock"):
        engine._acquire_project_lock(str(tmp_path), "editor")
    assert not (tmp_path / "different.lock").exists()


def _crash_in_transaction(project, entered, crash):
    import infernux_project_lock as locks
    with locks._guard(locks.lock_path(project)):
        entered.set()
        assert crash.wait(20)
        os._exit(0)


def test_crashed_transaction_releases_kernel_guard_without_replacing_it(tmp_path):
    import infernux_project_lock as locks
    context = multiprocessing.get_context("spawn")
    entered, crash = context.Event(), context.Event()
    child = context.Process(target=_crash_in_transaction, args=(str(tmp_path), entered, crash))
    try:
        child.start()
        assert entered.wait(30)
        guard = tmp_path / locks.GUARD_NAME
        identity = guard.stat().st_ino
        crash.set()
        child.join(30)
        assert child.exitcode == 0
        path = locks.claim(str(tmp_path), "after-crash", "headless")
        assert guard.stat().st_ino == identity
        assert locks.remove_lock(path, "after-crash")
    finally:
        crash.set()
        if child.is_alive():
            child.terminate()
        child.join(10)
