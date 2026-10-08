"""Isolated real Engine lifetimes; invoked by test_plugin_install_retirement."""

import threading

import pytest

from infernux.engine.engine import Engine
from infernux.engine.interaction import EditorInteractionCore
from infernux.engine.ui.plugin_install_progress import PluginInstallProgressService
from infernux.lib import LogLevel, RuntimeMode


@pytest.fixture
def owners(tmp_path, monkeypatch):
    from infernux.engine.preferences_store import PreferencesStore

    monkeypatch.setattr(PreferencesStore(), "_path", str(tmp_path / "preferences.json"))
    monkeypatch.setattr(PluginInstallProgressService, "_instance", None)
    created = []

    def create(name):
        root = tmp_path / name
        (root / "Assets").mkdir(parents=True)
        (root / "ProjectSettings").mkdir()
        engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
        created.append(engine)
        engine.init_headless(str(root))
        return engine, root

    yield create
    for engine in reversed(created):
        engine.exit()


def start(service):
    transaction = service._transaction
    transaction.presented_phase = "opening"
    service.post_present_tick()
    return transaction


def finish(service, transaction):
    assert transaction.worker_done.wait(3)
    transaction.presented_phase = "running"
    service.post_present_tick()
    transaction.presented_phase = "complete"
    transaction.completed_at -= 1
    service.post_present_tick()


@pytest.mark.parametrize("phase", ["opening", "running", "complete"])
def test_engine_exit_drains_old_worker_without_publishing_to_new_owner(owners, phase):
    engine, root = owners("A")
    service = PluginInstallProgressService.instance()
    release = threading.Event()
    events, callbacks = [], []

    def work(report):
        assert release.wait(3), "test worker was not released"
        assert engine.get_native_engine() is not None
        (root / "installation.txt").write_text("committed", encoding="utf-8")
        events.append("worker-finished")
        return "installed"

    assert service.begin(label="package", work=work, complete=lambda *args: callbacks.append(args))
    transaction = service._transaction
    timer = None
    if phase != "opening":
        start(service)
        if phase == "complete":
            release.set()
            assert transaction.worker_done.wait(3)
            transaction.presented_phase = "running"
            service.post_present_tick()
        else:
            timer = threading.Timer(.1, release.set)
            timer.start()
    engine.set_before_exit_callback(lambda: events.append("owner-exit"))
    try:
        engine.exit()
    finally:
        release.set()
        if timer is not None:
            timer.join()
        if transaction.worker is not None:
            transaction.worker.join(3)
    assert not service.is_active
    assert events == (["owner-exit"] if phase == "opening" else ["worker-finished", "owner-exit"])
    assert callbacks == []
    owners("B")
    assert service.begin(label="new-owner", work=lambda report: "B", complete=lambda *args: callbacks.append(args))
    finish(service, start(service))
    assert callbacks == [(True, "B", "")]
    assert not service.is_active


@pytest.mark.parametrize("phase", ["running", "complete"])
def test_escape_keeps_install_and_exit_barrier_until_owner_completion(owners, phase):
    from infernux.engine.ui.dirty_panel_confirmation import DirtyPanelConfirmationCoordinator

    owners("A")
    core = EditorInteractionCore.instance()
    service = PluginInstallProgressService.instance()
    completed = []
    release = threading.Event()
    assert service.begin(label="install", work=lambda report: release.wait(3),
                         complete=lambda *args: completed.append(args))
    transaction = start(service)
    try:
        if phase == "complete":
            release.set()
            assert transaction.worker_done.wait(3)
            transaction.presented_phase = "running"
            service.post_present_tick()
        assert not core.cancel_active_interaction()
        assert core.modals.active_modal_id == service.MODAL_ID
        assert not core.modals.cancel_owner("plugins")
        core.modals.unregister(service.MODAL_ID)
        assert core.modals.active_modal_id == service.MODAL_ID
        with pytest.raises(RuntimeError, match="must finish"):
            core.modals.clear()
        coordinator = DirtyPanelConfirmationCoordinator()
        assert not coordinator.request_exit(lambda: pytest.fail("owner exited while installing"), lambda: None)
        release.set()
        if phase == "running":
            finish(service, transaction)
        else:
            transaction.presented_phase = "complete"
            transaction.completed_at -= 1
            service.post_present_tick()
        assert completed == [(True, True, "")]
        assert not core.modals.active_modal_id
    finally:
        release.set()
        transaction.worker.join(3)


def test_unstarted_cancel_does_not_launch_work_or_keep_busy(owners):
    owners("A")
    core = EditorInteractionCore.instance()
    service = PluginInstallProgressService.instance()
    assert service.begin(label="pending", work=lambda report: pytest.fail("cancelled work ran"),
                         complete=lambda *args: pytest.fail("cancelled callback ran"))
    assert core.cancel_active_interaction()
    service.post_present_tick()
    assert not service.is_active
    assert not core.modals.active_stack


def test_completion_can_start_another_owned_modal(owners):
    owners("A")
    service = PluginInstallProgressService.instance()
    callbacks = []

    def complete(*args):
        callbacks.append(args)
        assert service.begin(label="next", work=lambda report: "next", complete=lambda *args: callbacks.append(args))

    assert service.begin(label="first", work=lambda report: "first", complete=complete)
    finish(service, start(service))
    assert EditorInteractionCore.instance().modals.active_modal_id == service.MODAL_ID
    finish(service, start(service))
    assert callbacks == [(True, "first", ""), (True, "next", "")]


def test_core_shutdown_drains_worker_and_unrelated_owner_cannot_retire_it(owners):
    engine, root = owners("A")
    core = EditorInteractionCore.instance()
    service = PluginInstallProgressService.instance()
    release = threading.Event()
    callbacks = []

    def work(report):
        assert release.wait(3)
        assert engine.get_native_engine() is not None
        (root / "worker-finished").write_text("finished", encoding="utf-8")

    assert service.begin(label="package", work=work, complete=lambda *args: callbacks.append(args))
    transaction = start(service)
    service.shutdown(engine=object())
    assert service.is_active and not transaction.worker_done.is_set()
    timer = threading.Timer(.1, release.set)
    timer.start()
    try:
        core.shutdown()
    finally:
        release.set()
        timer.join()
        transaction.worker.join(3)
    assert not service.is_active
    assert not transaction.worker.is_alive()
    assert (root / "worker-finished").is_file()
    assert callbacks == []
    service.post_present_tick()
    assert callbacks == []


def test_failure_in_previous_completion_cannot_cancel_its_successor(owners):
    owners("A")
    service = PluginInstallProgressService.instance()
    callbacks = []

    def complete(*args):
        assert service.begin(label="next", work=lambda report: "next", complete=lambda *args: callbacks.append(args))
        raise RuntimeError("intentional failure after starting successor")

    assert service.begin(label="first", work=lambda report: "first", complete=complete)
    finish(service, start(service))
    assert service.is_active
    assert EditorInteractionCore.instance().modals.active_modal_id == service.MODAL_ID
    finish(service, start(service))
    assert callbacks == [(True, "next", "")]
