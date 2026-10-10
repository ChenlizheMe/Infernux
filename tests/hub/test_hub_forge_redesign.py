"""Behavior added with the FORGE SIGNAL Hub redesign."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

import version_manager as vm
from install_queue import InstallCancelled, InstallQueue
from version_manager import EngineWheel, VersionManager


def _app():
    return QApplication.instance() or QApplication([])


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
    assert predicate()


# ── Install queue ────────────────────────────────────────────────────

def test_running_cancellable_download_stops_at_its_next_progress_checkpoint():
    app = _app()
    queue = InstallQueue(app)
    started = threading.Event()
    chunks = []

    def download(report):
        for index in range(400):
            report("Downloading", index * 1024, 400 * 1024)
            chunks.append(index)
            started.set()
            time.sleep(0.005)
        return "done"

    job = queue.submit("engine:1.0", "Infernux 1.0", download, cancellable=True)
    _wait(started.is_set)
    assert queue.cancel(job) is True
    assert job.state == "cancelling"
    _wait(lambda: not queue.busy)
    assert job.state == "cancelled"
    assert job.error == "" and job.result is None
    assert len(chunks) < 400
    queue.deleteLater()


def test_runtime_publication_is_not_cancellable_once_running():
    app = _app()
    queue = InstallQueue(app)
    release = threading.Event()

    def publish(report):
        report("Copying runtime", 0, 0)
        release.wait(5)
        return "ok"

    job = queue.submit("python:3.13", "Python 3.13", publish)
    _wait(lambda: job.state == "running")
    assert queue.cancel(job) is False
    release.set()
    _wait(lambda: not queue.busy)
    assert job.state == "succeeded"
    queue.deleteLater()


def test_manager_specific_cancellation_is_reported_as_cancelled():
    app = _app()
    queue = InstallQueue(app)
    started = threading.Event()

    def download(report):
        while True:
            started.set()
            if report.should_cancel():
                raise vm.DownloadCancelled("1.0")
            time.sleep(0.005)

    job = queue.submit("engine:1.0", "Infernux 1.0", download, cancellable=True)
    _wait(started.is_set)
    queue.cancel(job)
    _wait(lambda: not queue.busy)
    assert job.state == "cancelled"
    queue.deleteLater()


def test_failed_job_can_be_retried_and_speed_is_measured():
    app = _app()
    queue = InstallQueue(app)
    attempts = []

    def flaky(report):
        attempts.append(1)
        for index in range(6):
            report("Downloading", index * 512 * 1024, 6 * 512 * 1024)
            time.sleep(0.12)
        if len(attempts) == 1:
            raise OSError("connection reset")
        return "ok"

    job = queue.submit("engine:2.0", "Infernux 2.0", flaky, cancellable=True)
    _wait(lambda: job.bytes_per_second > 0, timeout=5)
    assert job.eta_seconds is not None
    _wait(lambda: not queue.busy)
    assert job.state == "failed" and "connection reset" in job.error
    retried = queue.retry(job)
    assert retried is not None and retried is not job and job not in queue.jobs
    _wait(lambda: not queue.busy)
    assert retried.state == "succeeded" and len(attempts) == 2
    queue.deleteLater()


def test_reporter_raises_after_cancel_even_without_manager_support():
    from install_queue import InstallJob, ProgressReporter
    job = InstallJob("k", "t", lambda report: None, cancellable=True)
    emitted = []
    reporter = ProgressReporter(job, lambda *args: emitted.append(args))
    reporter("Downloading", 1, 2)
    job.cancel_event.set()
    with pytest.raises(InstallCancelled):
        reporter("Downloading", 2, 2)
    assert emitted == [("Downloading", 1, 2)]


# ── Version catalog ──────────────────────────────────────────────────

@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "_VERSIONS_DIR", tmp_path / "Engines")
    return VersionManager(runtime_manager=SimpleNamespace(has_runtime=lambda _v: True,
                                                          installed_versions=lambda: ["3.13"]))


def test_cached_versions_never_touch_the_network(manager, monkeypatch):
    def offline(*_args, **_kwargs):
        raise AssertionError("cached_versions must not open a connection")

    monkeypatch.setattr(vm.urllib.request, "urlopen", offline)
    assert manager.cached_versions(include_prerelease=True) == []
    assert manager.catalog_timestamp() is None
    cache = {"_ts": 123.0, "releases": [{"tag_name": "v0.4.1", "assets": []}]}
    import json
    manager._cache_file.write_text(json.dumps(cache), encoding="utf-8")
    versions = manager.cached_versions(include_prerelease=True)
    assert [item.version for item in versions] == ["0.4.1"]
    assert manager.catalog_timestamp() == 123.0


def test_wheel_validation_is_memoized_until_the_file_changes(tmp_path, monkeypatch):
    import zipfile
    wheel = tmp_path / "infernux-0.4.1-cp313-cp313-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("x", "1")
    hashes = []
    monkeypatch.setattr(vm, "has_matching_wheel_identity", lambda path: True)
    monkeypatch.setattr(vm, "_verify_download", lambda path, expected: hashes.append(path))
    expected = (EngineWheel(wheel.name, "", 0, "3.13", sha256="0" * 64),)
    vm._WHEEL_VERDICTS.clear()
    assert vm._cached_wheel_verdict(str(wheel), expected) == "valid"
    assert vm._cached_wheel_verdict(str(wheel), expected) == "valid"
    assert len(hashes) == 1
    time.sleep(0.01)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("y", "2")
    assert vm._cached_wheel_verdict(str(wheel), expected) == "valid"
    assert len(hashes) == 2
    wheel.write_bytes(b"truncated")
    assert vm._cached_wheel_verdict(str(wheel), expected) == "corrupt"


# ── Project list ─────────────────────────────────────────────────────

def test_double_click_and_enter_request_a_launch(tmp_path):
    _app()
    from ui_project_list import ProjectListPane
    records = [SimpleNamespace(project_id=str(index), name=f"Project {index}", created_at="",
                               path=str(tmp_path)) for index in range(2)]
    pane = ProjectListPane(SimpleNamespace(all_projects=lambda: records,
                                           get_project=lambda pid: records[int(pid)]))
    launched = []
    pane.launch_requested.connect(launched.append)
    pane.project_cards["1"].mouseDoubleClickEvent(None)
    assert launched == ["1"] and pane.get_selected_project_id() == "1"
    QTest.keyClick(pane, Qt.Key.Key_Up)
    assert pane.get_selected_project_id() == "0"
    QTest.keyClick(pane, Qt.Key.Key_Return)
    assert launched == ["1", "0"]
    pane.project_cards["1"]._launch_button.click()
    assert launched == ["1", "0", "1"]
    pane.close()


def test_empty_project_list_offers_create_and_open(tmp_path):
    _app()
    from PySide6.QtWidgets import QPushButton
    from ui_project_list import ProjectListPane
    pane = ProjectListPane(SimpleNamespace(all_projects=lambda: []))
    requests = []
    pane.create_requested.connect(lambda: requests.append("create"))
    pane.open_requested.connect(lambda: requests.append("open"))
    empty = pane.card_layout.itemAt(0).widget()
    for button in empty.findChildren(QPushButton):
        button.click()
    assert sorted(requests) == ["create", "open"]
    pane.close()


# ── Creation sequence ────────────────────────────────────────────────

def test_creation_dialog_tracks_real_status_messages():
    _app()
    from viewmodel.control_pane_viewmodel import CustomProgressDialog
    dialog = CustomProgressDialog()
    try:
        dialog.set_status("Creating project folders...")
        assert dialog._current == 0
        dialog.set_status("Copying Python runtime into the project...")
        assert dialog._current == 1
        dialog.set_status("Installing default project libraries...")
        assert dialog._current == 3
        # Later stages never move the sequence backwards.
        dialog.set_status("Finalizing project...")
        assert dialog._current == 3
        assert [led.kind() for led, _text, _state in dialog._steps] == ["ok", "ok", "ok", "busy", "idle"]
    finally:
        dialog.finish()
        dialog.deleteLater()


# ── Community ────────────────────────────────────────────────────────

def test_community_feed_loads_when_first_shown(monkeypatch):
    _app()
    from view import discussion_view
    monkeypatch.setitem(discussion_view._FEED_CACHE, "topics", None)
    monkeypatch.setitem(discussion_view._FEED_CACHE, "at", 0.0)
    view = discussion_view.DiscussionView()
    calls = []
    monkeypatch.setattr(view, "refresh", lambda: calls.append("refresh"))
    try:
        assert calls == []
        view.show()
        _wait(lambda: calls == ["refresh"], timeout=2)
        assert view._auto_refresh.isActive()
        view.hide()
        assert not view._auto_refresh.isActive()
    finally:
        view.shutdown()
        view.close()
        view.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


# ── Hub overhaul: dialogs, dropdown, project rows, window ────────────

def _run_dialog(action):
    from PySide6.QtCore import QTimer
    QTimer.singleShot(60, action)


def test_hub_dialog_answers_follow_keyboard_and_buttons():
    _app()
    from view import dialogs
    from PySide6.QtWidgets import QApplication

    def press(key):
        def act():
            dialog = QApplication.activeModalWidget()
            assert isinstance(dialog, dialogs.HubDialog)
            QTest.keyClick(dialog, key)
        return act

    _run_dialog(press(Qt.Key.Key_Return))
    assert dialogs.question(None, "Title", "Body", dialogs.Yes | dialogs.No, dialogs.Yes) == dialogs.Yes
    _run_dialog(press(Qt.Key.Key_Escape))
    assert dialogs.question(None, "Title", "Body") == dialogs.No
    # A destructive confirmation defaults to the safe answer.
    _run_dialog(press(Qt.Key.Key_Return))
    assert dialogs.confirm(None, "Remove?", "Body", confirm_label="Remove", destructive=True) is False


def test_hub_dialog_exposes_text_and_details():
    _app()
    from view import dialogs
    dialog = dialogs.HubDialog(None, level="critical", title="Failed", text="Summary", detail="Line 1\nLine 2")
    cancel = dialog.addButton("Stop", "reject")
    retry = dialog.addButton("Retry", "accept")
    dialog.setDefaultButton(retry)
    assert dialog.text() == "Summary" and dialog.detailedText() == "Line 1\nLine 2"
    assert dialog.defaultButton() is retry and cancel.objectName() == "normalBtn"
    assert retry.objectName() == "primaryBtn"
    assert "ESC" in dialog._hint.text() and "ENTER" in dialog._hint.text()
    dialog.deleteLater()


def test_forge_combo_carries_meta_and_tag():
    _app()
    from view.forge import META_ROLE, TAG_ROLE, ForgeComboBox
    combo = ForgeComboBox()
    combo.addItem("0.4.1", "0.4.1", meta="PY 3.13", tag="LATEST")
    combo.addItem("0.4.0", "0.4.0")
    assert combo.itemData(0, META_ROLE) == "PY 3.13" and combo.itemData(0, TAG_ROLE) == "LATEST"
    assert combo.itemData(1, META_ROLE) is None
    assert combo.itemDelegate().sizeHint(__import__("PySide6.QtWidgets").QtWidgets.QStyleOptionViewItem(),
                                         combo.model().index(0, 0)).height() >= 30
    combo.deleteLater()


def test_unavailable_rows_keep_their_menu_and_offer_a_fix(tmp_path, monkeypatch):
    _app()
    import ui_project_list
    project = tmp_path / "Needs engine"
    (project / "ProjectSettings").mkdir(parents=True)
    (project / "ProjectSettings" / "PythonRuntime.json").write_text('{"pythonVersion": "3.13"}', encoding="utf-8")
    (project / ".infernux-version").write_text("0.9.0\n", encoding="utf-8")
    monkeypatch.setattr(ui_project_list, "is_frozen", lambda: True)
    records = [
        SimpleNamespace(project_id="engine", name="Needs engine", created_at="", path=str(project)),
        SimpleNamespace(project_id="gone", name="Gone", created_at="", path=str(tmp_path / "missing")),
    ]
    manager = SimpleNamespace(is_installed=lambda *_args: False)
    pane = ui_project_list.ProjectListPane(SimpleNamespace(all_projects=lambda: records), manager)
    installs, relocations, removals = [], [], []
    pane.install_requested.connect(installs.append)
    pane.relocate_requested.connect(relocations.append)
    pane.remove_requested.connect(removals.append)
    try:
        engine, gone = pane.project_cards["engine"], pane.project_cards["gone"]
        for card in (engine, gone):
            assert card.isEnabled() and not card.launchable
            assert card._actions_button.isEnabled()
            assert not card._launch_button.isVisibleTo(card)
        engine._fix_button.click()
        gone._fix_button.click()
        assert installs == ["0.9.0"] and relocations == ["gone"]
        remove = next(action for action in gone._actions_menu.actions() if action.text() == ui_project_list.tr("Remove from Hub"))
        remove.trigger()
        assert removals == ["gone"]
    finally:
        pane.close()


def test_pins_and_sorting_order_rows(tmp_path):
    _app()
    import json
    import ui_project_list
    settings = {}
    records = []
    for index, name in enumerate(("Beta", "alpha", "Gamma")):
        folder = tmp_path / name
        folder.mkdir()
        import os
        os.utime(folder, (1000 + index, 1000 + index))
        records.append(SimpleNamespace(project_id=name, name=name, created_at="", path=str(folder)))
    db = SimpleNamespace(all_projects=lambda: records, get_project=lambda pid: None,
                         get_setting=lambda key, default="": settings.get(key, default),
                         set_setting=settings.__setitem__)
    pane = ui_project_list.ProjectListPane(db)
    order = lambda: [pane.card_layout.itemAt(i).widget().project_id for i in range(len(records))]
    try:
        assert order() == ["Gamma", "alpha", "Beta"]  # most recently modified first
        pane.set_sort("name")
        assert order() == ["alpha", "Beta", "Gamma"] and settings["project_sort"] == "name"
        pane.toggle_pin("Gamma")
        assert order()[0] == "Gamma" and json.loads(settings["pinned_projects"]) == ["Gamma"]
        assert pane.project_cards["Gamma"].pinned
    finally:
        pane.close()


def test_toast_host_stacks_and_dismisses():
    app = _app()
    from PySide6.QtWidgets import QWidget
    from view.forge import ToastHost, toast
    window = QWidget()
    window.resize(900, 600)
    window.toast_host = ToastHost(window, left_inset=200)
    window.show()
    clicked = []
    first = toast(window, "First", "ok")
    second = toast(window, "Second", "info", action=("Undo", lambda: clicked.append(True)), timeout_ms=200)
    assert first is not None and second is not None
    assert second.y() >= first.y() + first.height()
    assert first.x() == 220
    import shiboken6
    QTest.qWait(700)
    # The timed toast dismissed itself and was deleted; the other one stays.
    assert not shiboken6.isValid(second) or not second.isVisible()
    assert first.isVisible()
    window.close()


def test_hub_window_fits_the_available_screen(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERNUX_DATA_ROOT", str(tmp_path / "User"))
    monkeypatch.setenv("INFERNUX_SHARED_DATA_ROOT", str(tmp_path / "Shared"))
    app = _app()
    from hub_utils import HubLaunchContext
    from launcher import GameEngineLauncher
    hub = GameEngineLauncher(HubLaunchContext.SOURCE)
    try:
        available = (hub.screen() or app.primaryScreen()).availableGeometry()
        assert hub.width() <= available.width() and hub.height() <= available.height()
        assert hub.minimumWidth() <= available.width()
        hub.resize(hub.minimumWidth() + 33, hub.minimumHeight() + 21)
        hub._save_geometry()
        assert hub.db.get_setting(hub.GEOMETRY_SETTING, "")
    finally:
        hub.db.close()
        hub.deleteLater()
