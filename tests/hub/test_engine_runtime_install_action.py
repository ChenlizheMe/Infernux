from types import SimpleNamespace

import pytest
import version_manager
from PySide6.QtTest import QTest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication

from i18n import configure_language, language_mode
from version_manager import EngineVersion, VersionManager
from view.installs_view import InstallEditorDialog, PythonRuntimesView
from install_queue import InstallQueue
from view import installs_view


@pytest.fixture
def engine_dialog(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    previous_language = language_mode()
    configure_language("en")
    installed = set()
    monkeypatch.setattr(version_manager, "_VERSIONS_DIR", tmp_path / "engines")
    manager = VersionManager(
        runtime_manager=SimpleNamespace(has_runtime=lambda version: version in installed),
    )
    engine = EngineVersion(
        tag="v0.4.0",
        version="0.4.0",
        python_version="3.13",
        wheel_url="https://example.invalid/infernux.whl",
    )
    monkeypatch.setattr(manager, "list_versions", lambda **kwargs: [engine])
    queue = InstallQueue(app)
    dialog = InstallEditorDialog(manager, queue)
    try:
        for _ in range(500):
            app.processEvents()
            if not dialog._fetch_thread.isRunning():
                break
            QTest.qWait(10)
        assert not dialog._fetch_thread.isRunning()
        yield dialog, engine, installed
    finally:
        dialog._fetch_thread.quit()
        dialog._fetch_thread.wait()
        dialog.close()
        dialog.deleteLater()
        queue.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        configure_language(previous_language)


@pytest.mark.parametrize("installed_successfully", [False, True])
def test_missing_runtime_requires_explicit_action_and_preserves_engine_selection(
    engine_dialog,
    installed_successfully,
):
    dialog, engine, installed = engine_dialog
    requests = []

    def install(version):
        requests.append(version)
        if installed_successfully:
            installed.add(version)

    dialog.runtime_install_requested.connect(install)
    dialog._select(engine)
    assert requests == []
    assert not dialog._btn_install.isEnabled()
    assert not dialog._btn_runtime.isHidden()
    assert "3.13" in dialog._btn_runtime.text()

    dialog._btn_runtime.click()
    assert requests == ["3.13"]
    assert dialog._selected is engine
    assert dialog._btn_install.isEnabled() is installed_successfully
    assert dialog._btn_runtime.isHidden() is installed_successfully
    assert dialog._queue.jobs == []  # Runtime installation does not install the engine.


def test_install_page_connects_dependency_action_to_shared_queue(engine_dialog, monkeypatch):
    dialog, engine, installed = engine_dialog
    calls = []

    def prepare(*, version, on_status, download_ca_bundle):
        calls.append((version, download_ca_bundle))
        installed.add(version)
        return "/managed/python"

    manager = SimpleNamespace(
        default_version="3.13",
        supported_versions=lambda: ["3.13"],
        get_runtime_path=lambda version: "/managed/python" if version in installed else "",
        has_runtime=lambda version: version in installed,
        ensure_runtime=prepare,
    )
    monkeypatch.setattr(installs_view, "InstallEditorDialog", lambda *args, **kwargs: dialog)
    page = installs_view.InstallsView(dialog._vm, dialog._queue)
    settings = SimpleNamespace(get_setting=lambda key, default="": "proxy-ca.pem")
    python_page = PythonRuntimesView(manager, dialog._queue, settings=settings)
    page.runtime_install_requested.connect(python_page.install)
    try:
        page._on_install_editor()
        dialog._select(engine)
        dialog._btn_runtime.click()
        assert calls == []  # The UI callback only enqueues work.
        for _ in range(500):
            QTest.qWait(10)
            if not dialog._queue.busy:
                break
        assert not dialog._queue.busy
        assert calls == [("3.13", "proxy-ca.pem")]
        assert dialog._btn_install.isEnabled()
        assert [job.key for job in dialog._queue.jobs] == ["python:3.13"]
    finally:
        page.close()
        python_page.close()
        page.deleteLater()
        python_page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("automatic", [False, True])
def test_install_page_respects_automatic_update_preference_and_allows_manual_check(engine_dialog, monkeypatch, automatic):
    dialog, _, _ = engine_dialog
    calls = []
    monkeypatch.setattr(installs_view.InstallsView, "check_updates", lambda self: calls.append("check"))
    settings = SimpleNamespace(get_setting=lambda *_: "enabled" if automatic else "disabled")
    page = installs_view.InstallsView(dialog._vm, dialog._queue, settings=settings)
    try:
        page.show()
        QCoreApplication.processEvents()
        assert calls == (["check"] if automatic else [])
        calls.clear()
        page.btn_check.click()
        assert calls == ["check"]
    finally:
        page.close()
        page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize("failure", [None, "installation failed", ""])
def test_runtime_install_completion_follows_worker_cleanup(failure):
    app = QApplication.instance() or QApplication([])
    queue = InstallQueue(app)

    def prepare(**kwargs):
        if failure is not None:
            raise RuntimeError(failure)
        return "/managed/python"

    manager = SimpleNamespace(
        default_version="3.13",
        supported_versions=lambda: [],
        has_runtime=lambda version: False,
        ensure_runtime=prepare,
    )
    page = PythonRuntimesView(manager, queue)
    cleaned_up = []
    queue.job_finished.connect(lambda job: cleaned_up.append(queue._thread is None))
    job = page.install("3.13")
    for _ in range(500):
        QTest.qWait(10)
        if not queue.busy:
            break
    assert not queue.busy
    assert cleaned_up == [True]
    assert job.state == ("succeeded" if failure is None else "failed")
    page.close()
    page.deleteLater()
    queue.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_installed_engine_card_updates_its_hotfix_without_replacing_old_installs(engine_dialog, monkeypatch):
    dialog, engine, installed = engine_dialog
    installed.add('3.13')
    cached = ['0.4.0']
    hotfix = EngineVersion(tag='v0.4.0-v2', version='0.4.0-v2', python_version='3.13', wheel_url=engine.wheel_url)
    other = EngineVersion(tag='v0.5.0', version='0.5.0', python_version='3.13', wheel_url=engine.wheel_url)
    manager = dialog._vm
    monkeypatch.setattr(manager, 'list_versions', lambda **kwargs: [engine, hotfix, other])
    monkeypatch.setattr(manager, 'installed_versions', lambda: cached)
    calls = []

    def download(version, **kwargs):
        calls.append(version)
        cached.append(version)
        return '/cached/new.whl'

    monkeypatch.setattr(manager, 'download_version', download)
    page = installs_view.InstallsView(manager, dialog._queue)
    try:
        card = page.findChild(installs_view._VersionCard)
        button = next(button for button in card.findChildren(installs_view.QPushButton)
                      if button.text() == 'Update to hotfix 2')
        button.click()
        for _ in range(500):
            QTest.qWait(10)
            if not dialog._queue.busy:
                break
        assert calls == ['0.4.0-v2']
        assert dialog._queue.jobs[-1].state == 'succeeded'
        assert cached == ['0.4.0', '0.4.0-v2']
    finally:
        page.close()
        page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
