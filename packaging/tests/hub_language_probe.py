"""Language refresh must preserve Hub services and in-flight work."""
from pathlib import Path
import sys
import threading
import time
import traceback

from PySide6.QtCore import QCoreApplication, QEvent, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import shiboken6

from hub_utils import HubLaunchContext
from launcher import GameEngineLauncher
from i18n import tr


def spin(predicate):
    until = time.monotonic() + 5
    while not predicate() and time.monotonic() < until:
        QTest.qWait(5)
    assert predicate(), 'Qt operation did not complete'


def main():
    repeats, outcome, destination = sys.argv[1:]
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    errors = []
    def error(*exc):
        errors.append(str(exc[1]))
        traceback.print_exception(*exc)
    sys.excepthook = error
    hub = GameEngineLauncher(HubLaunchContext.SOURCE)
    database, queue, controller, vm = hub.db, hub.install_queue, hub.update_controller, hub.viewmodel
    project = Path(destination) / 'Example'
    project.mkdir()
    record = database.add_project('Example', str(project))
    hub.project_list.refresh()
    hub.project_list.select_project(record.project_id)
    hub.project_list.search_edit.setText('Exam')
    hub.install_tabs.setCurrentIndex(2)
    hub.sidebar.select_page(2)
    gate = threading.Event()
    entered = threading.Event()
    calls = []
    old_pages = []
    old_refreshes = []

    def operation(report):
        entered.set()
        assert gate.wait(5)
        calls.append('operation')
        if outcome == 'failure':
            raise RuntimeError('controlled queue failure')
        return 'result'

    update_done = []
    if outcome == 'update':
        import view.update_dialog as update_module
        from hub_updater import HubUpdateCheck, HubUpdateStatus
        def check():
            entered.set()
            assert gate.wait(5)
            return HubUpdateCheck(HubUpdateStatus.UP_TO_DATE, '0.4.1', '0.4.1')
        update_module.check_for_update = check
        controller.check_finished.connect(lambda: update_done.append(True))
        controller.check()
        spin(entered.is_set)
    elif outcome == 'versions':
        def versions(*, include_prerelease):
            assert include_prerelease
            entered.set()
            assert gate.wait(5)
            return []
        hub.version_manager.list_versions = versions
        hub.installs_view._on_install_editor()
        spin(entered.is_set)
    job = queue.submit('language-test', 'Local task', operation)
    if outcome == 'cancelled':
        queue.cancel_queued(job)
    elif outcome not in ('update', 'versions'):
        spin(entered.is_set)

    try:
        for index in range(int(repeats)):
            old_pages.append(hub.centralWidget())
            old_list = hub.project_list
            original_refresh = old_list.refresh
            def refresh_current(owner=old_list, refresh=original_refresh):
                if hub.project_list is not owner:
                    old_refreshes.append(True)
                refresh()
            old_list.refresh = refresh_current
            combo = hub.settings_view.language_combo
            mode = 'en' if combo.currentData() != 'en' else 'zh'
            combo.setCurrentIndex(combo.findData(mode))
            if outcome == 'versions' and index == 0:
                # A page-owned catalog thread must finish before its page dies.
                QTest.qWait(40)
                assert shiboken6.isValid(old_pages[-1])
                gate.set()
            spin(lambda: hub.centralWidget() is not old_pages[-1] or hasattr(hub, '_language_replacement'))
            assert hub.db is database and hub.install_queue is queue and hub.update_controller is controller
            assert hub.viewmodel is vm
            assert not hasattr(hub, '_language_replacement'), 'language change replaced the service owner'
            assert hub.project_list.get_selected_project_id() == record.project_id
            assert hub.project_list.search_edit.text() == 'Exam'
            assert hub.pages.currentIndex() == 2 and hub.install_tabs.currentIndex() == 2
            assert hub.sidebar._nav_buttons[2].text() == tr('Settings')
    finally:
        gate.set()
        spin(lambda: not queue.busy)
        if outcome == 'update':
            spin(lambda: bool(update_done))
        QTest.qWait(40)
    assert not old_refreshes, 'retired project pages still receive queue callbacks'
    assert not errors, errors
    assert job.state == ('failed' if outcome == 'failure' else 'cancelled' if outcome == 'cancelled' else 'succeeded')
    assert calls == ([] if outcome == 'cancelled' else ['operation'])
    assert database.get_project(record.project_id).name == 'Example'
    assert hub.project_list.get_selected_project_id() == record.project_id
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert all(not shiboken6.isValid(page) for page in old_pages)
    QTimer.singleShot(0, hub.request_quit)
    app.exec()
    assert not errors, errors
    print('LANGUAGE_LIFECYCLE_OK', flush=True)


if __name__ == '__main__':
    main()
