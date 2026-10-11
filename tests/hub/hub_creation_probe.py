"""Real Qt project-creation lifecycle, isolated from pytest's process."""
from pathlib import Path
import sys
import threading
import time
import traceback
from types import SimpleNamespace

from PySide6.QtCore import QCoreApplication, QEvent, QSettings, QThread, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
import shiboken6

from hub_utils import HubLaunchContext
from view.new_project_view import NewProjectView
from viewmodel.control_pane_viewmodel import ControlPaneViewModel, CustomProgressDialog


def spin(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(5)
    assert predicate(), "Qt operation did not finish"


def main():
    action, outcome, destination = sys.argv[1:]
    root = Path(destination)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root / 'settings'))
    errors, messages, started, registered, selected, refreshed = [], [], [], [], [], []
    def record_error(*exc):
        errors.append(str(exc[1]))
        traceback.print_exception(*exc)
    sys.excepthook = record_error
    from view import dialogs
    dialogs.critical = lambda *args, **kwargs: messages.append(args[2])
    gate = threading.Event()
    finished = threading.Event()

    class Model:
        def init_project_folder(self, name, path, version, on_status):
            assert QThread.currentThread() != app.thread()
            started.append(name)
            on_status('Preparing project...')
            assert gate.wait(5), 'test did not release worker'
            finished.set()
            if outcome == 'failure' and len(started) == 1:
                raise RuntimeError('controlled initialization failure')
            return str(Path(path) / name)

        def add_project(self, name, path):
            assert QThread.currentThread() == app.thread()
            registered.append((name, path))
            return SimpleNamespace(project_id=name)

    if action in ('hub-quit', 'hub-language'):
        from launcher import GameEngineLauncher
        hub = GameEngineLauncher(HubLaunchContext.SOURCE)
        vm = hub.viewmodel
        vm.model.init_project_folder = Model().init_project_folder
        add_project = vm.model.add_project
        def add(name, path):
            result = add_project(name, path)
            registered.append((name, path))
            return result
        vm.model.add_project = add
        select_project = vm.project_list.select_project
        def select(project_id):
            selected.append(vm.model.db.get_project(project_id).name)
            select_project(project_id)
        vm.project_list.select_project = select
    else:
        vm = ControlPaneViewModel(Model(), SimpleNamespace(
            refresh=lambda: refreshed.append(True), select_project=selected.append),
            launch_context=HubLaunchContext.SOURCE)
    form_exec = NewProjectView.exec
    forms = []

    def fill_form(form):
        forms.append(form)
        # A second entry while busy must never open another form/thread.
        if len(forms) > 1 and not finished.is_set():
            form.reject()
            return QDialog.Rejected
        def fill():
            form.name_edit.setText('First' if len(forms) == 1 else 'Second')
            form.path_edit.setText(str(root))
            form.viewmodel.set_path(str(root))
            form.accept()
        QTimer.singleShot(0, fill)
        return form_exec(form)

    NewProjectView.exec = fill_form
    vm.create_project(None)
    spin(lambda: started == ['First'])
    thread, worker = vm._creation_thread, vm._creation_worker
    progress = next(w for w in app.topLevelWidgets() if isinstance(w, CustomProgressDialog) and w.isVisible())
    try:
        if action == 'hub-language':
            old_central = hub.centralWidget()
            combo = hub.settings_view.language_combo
            combo.setCurrentIndex(combo.findData('en' if combo.currentData() != 'en' else 'zh'))
            QTest.qWait(30)
            assert hub.centralWidget() is old_central
            gate.set()
            spin(lambda: vm._creation_thread is None)
            spin(lambda: hub.centralWidget() is not old_central)
            assert vm.project_list is hub.project_list
        elif action in ('quit', 'hub-quit'):
            if action == 'hub-quit':
                # Also cover application shutdown with a hidden task window.
                progress.hide()
            threading.Timer(0.25, gate.set).start()
            QTimer.singleShot(0, hub.request_quit if action == 'hub-quit' else app.quit)
            # Qt may reject the first quit while a non-cancellable dialog is
            # visible. Retry the user action after initialization completes.
            QTimer.singleShot(500, app.quit)
            app.exec()
            assert finished.is_set()
            assert not shiboken6.isValid(thread) or not thread.isRunning()
        else:
            if action == 'escape':
                QTest.keyClick(progress, Qt.Key.Key_Escape)
            elif action == 'close':
                progress.close()
            elif action == 'reject':
                progress.reject()
            elif action == 'accept':
                progress.accept()
            elif action == 'done':
                progress.done(QDialog.Accepted)
            assert progress.isVisible(), 'creation progress disappeared while task still runs'
            vm.create_project(None)
            assert len(forms) == 1, 'creation reentry opened another form'
            gate.set()
            spin(lambda: bool(refreshed))
            assert not shiboken6.isValid(progress) or not progress.isVisible()
    finally:
        gate.set()
        spin(lambda: not thread.isRunning() if shiboken6.isValid(thread) else True)
        QTest.qWait(30)
    assert not errors, errors
    assert registered == ([] if outcome == 'failure' else [('First', str(root / 'First'))])
    assert selected == ([] if outcome == 'failure' else ['First'])
    if action not in ('quit', 'hub-quit'):
        assert len(messages) == (1 if outcome == 'failure' else 0)
        # Both success and failure must allow the next independent task.
        vm.create_project(None)
        spin(lambda: len(registered) == (1 if outcome == 'failure' else 2))
        assert registered[-1] == ('Second', str(root / 'Second'))
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not shiboken6.isValid(worker), 'worker was leaked after its thread stopped'
    assert not errors, errors
    if action == 'hub-quit':
        from database import ProjectDatabase
        db = ProjectDatabase()
        assert [p.name for p in db.all_projects()] == ([] if outcome == 'failure' else ['First'])
        db.close()
    print('CREATION_LIFECYCLE_OK', flush=True)


if __name__ == '__main__':
    main()
