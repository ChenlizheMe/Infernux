import os
import sys
import uuid
from PySide6.QtWidgets import (
    QApplication, QMessageBox, QDialog, QVBoxLayout, QLabel, QProgressBar, QFileDialog
)
from PySide6.QtCore import QThread, Signal, Slot, QObject, QTimer, Qt
from model.project_model import ProjectModel, source_engine_version
from hub_utils import HubLaunchContext, is_project_open, write_project_lock, remove_project_lock
from project_paths import ProjectPathError
from i18n import tr
import random


class CustomProgressDialog(QDialog):
    """Indeterminate progress dialog shown during project initialization."""

    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        self.setWindowTitle(title or tr("Initializing"))
        self.setWindowModality(Qt.WindowModal)
        self.setFixedSize(340, 110)

        self.label = QLabel(tr("Preparing project..."), self)
        self.label.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.messages = [
            tr("Setting up project structure..."),
            tr("Copying engine libraries..."),
            tr("Setting up Python runtime..."),
            tr("Preparing asset folders..."),
            tr("Almost there..."),
        ]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._rotate_message)
        self.timer.start(2000)

    def set_status(self, message: str):
        if self.timer.isActive():
            self.timer.stop()
        self.label.setText(tr(message))

    def _rotate_message(self):
        self.label.setText(random.choice(self.messages))

    def reject(self):
        # Initialization cannot be cancelled halfway through runtime publication.
        pass

    def done(self, _result):
        pass

    def closeEvent(self, event):
        event.ignore()

    def finish(self):
        self.timer.stop()
        super().done(QDialog.Accepted)


class InitProjectWorker(QObject):
    """Worker that runs project initialization on a background thread."""
    progress = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, model, name, path, engine_version=""):
        super().__init__()
        self.model = model
        self.name = name
        self.path = path
        self.engine_version = engine_version
        self.project_dir = ""
        self.error_message = ""

    def run(self):
        try:
            self.project_dir = self.model.init_project_folder(
                self.name,
                self.path,
                self.engine_version,
                on_status=self.progress.emit,
            )
        except Exception as exc:
            self.error_message = str(exc)
            self.error.emit(self.error_message)
            return
        self.finished.emit()




class LaunchPreparationWorker(QObject):
    """Prepare a project runtime without blocking the Hub event loop."""

    progress = Signal(str, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, model, version_manager, project_path: str, launch_context=None):
        super().__init__()
        self.model = model
        self.version_manager = version_manager
        self.project_path = project_path
        self.launch_context = launch_context or HubLaunchContext.current()
        self.lock_token = uuid.uuid4().hex
        self.editor_runtime = None

    def run(self):
        reserved = False
        try:
            project_path = self.project_path
            if not os.path.isdir(project_path):
                raise RuntimeError(
                    f"The project folder could not be found:\n{project_path}\n\n"
                    "Remove this entry from Hub, then use Open Existing to register its new location."
                )
            if is_project_open(project_path):
                raise RuntimeError(
                    "The project is already open in Infernux and cannot be opened again:\n"
                    f"{project_path}"
                )

            # The pre-check is presentation only. This atomic reservation owns
            # runtime preparation as well as the subsequent editor launch.
            write_project_lock(project_path, os.getpid(), self.lock_token, "editor", "preparing")
            reserved = True

            self.progress.emit("Checking engine version...", 4)
            python_version = ProjectModel.get_project_python_version(project_path)
            pinned_version = (
                self.version_manager.read_project_version(project_path)
                if self.version_manager
                else ""
            )
            if (
                self.launch_context.uses_installed_versions
                and pinned_version
                and self.version_manager
                and not self.version_manager.is_installed(
                    pinned_version, python_version
                )
            ):
                raise RuntimeError(
                    f"Infernux {pinned_version} for Python {python_version} is "
                    f"required by this project. Open {tr('Installs')} and install "
                    "that exact engine/Python combination before launching."
                )

            if self.launch_context.uses_installed_versions:
                if not pinned_version:
                    raise RuntimeError(
                        "This project does not pin an Infernux engine version.\n\n"
                        "Commit .infernux-version with the project before opening it "
                        "from an installed Hub."
                    )
                self.progress.emit("Checking project runtime...", 7)
                python_exe = ProjectModel._get_project_python(project_path)
                if not os.path.isfile(python_exe):
                    runtime_manager = getattr(self.model, "runtime_manager", None)
                    if runtime_manager is None or not runtime_manager.has_runtime(
                        python_version
                    ):
                        raise RuntimeError(
                            f"Infernux {pinned_version} requires Python {python_version}.\n\n"
                            f"Install Python {python_version} in Hub before launching "
                            "this project."
                        )

                    self.progress.emit("Preparing project runtime...", 8)
                    status = lambda message: self.progress.emit(message, 9)
                    self.model._create_project_runtime(
                        project_path,
                        on_status=status,
                        replace_existing=True,
                    )
                    python_exe = ProjectModel._get_project_python(project_path)
                    if not os.path.isfile(python_exe):
                        raise RuntimeError(
                            "Project Python runtime could not be rebuilt at:\n"
                            f"{os.path.dirname(python_exe)}"
                        )
                # The private runtime must contain exactly the pinned engine.
                # Publish it from the local installation; its receipt skips an
                # unchanged runtime. Project version migration is not supported.
                self.model._install_infernux_in_runtime(
                    project_path,
                    pinned_version,
                    on_status=lambda message: self.progress.emit(message, 9),
                    validate_current=False,
                )
                self.model._create_vscode_workspace(project_path)
                if sys.platform == "win32":
                    from python_execution import editor_python_runtime
                    self.editor_runtime = editor_python_runtime(
                        python_exe, self.model.runtime_manager.get_runtime_path(python_version)
                    )
                # Starting the editor is the authoritative native import check.
                # A separate smoke-test process here used to double cold-start
                # Python before the splash screen could even become responsive.
            else:
                current_version = source_engine_version()
                if pinned_version != current_version:
                    raise RuntimeError(
                        f"Project requires Infernux {pinned_version or 'an explicit version pin'}; "
                        f"the source engine is {current_version}. "
                        "Use the exact required engine version. Engine upgrades are not supported."
                    )

                # A source-launched Hub is a development/test frontend for the
                # Python environment that started it.  Infernux is therefore
                # already supplied by that environment; consulting the Hub
                # install catalog or trying to locate/install a wheel would
                # incorrectly turn this path back into the packaged-Hub flow.
                self.progress.emit("Using current development environment...", 7)
                self.progress.emit("Updating editor integration...", 9)
                self.model._create_vscode_workspace(project_path)
                python_exe = sys.executable
        except Exception as exc:
            detail = str(exc)
            if reserved:
                try:
                    remove_project_lock(self.project_path, self.lock_token)
                except Exception as cleanup_error:
                    detail += f"\nProject reservation release failed: {cleanup_error}"
            self.error.emit(detail)
            return
        self.finished.emit(python_exe)


class ControlPaneViewModel(QObject):
    def __init__(
        self,
        model,
        project_list,
        version_manager=None,
        runtime_manager=None,
        launch_context=None,
    ):
        super().__init__()
        self.model = model
        self.project_list = project_list
        self.version_manager = version_manager
        self.runtime_manager = runtime_manager
        self.launch_context = launch_context or HubLaunchContext.current()
        self._launch_thread = None
        self._launch_worker = None
        self._launch_splash = None
        self._creation_thread = None
        self._creation_worker = None
        self._creation_dialog = None

    def launch_project(self, parent):
        record = self.project_list.get_selected_record()
        if record is None:
            QMessageBox.warning(parent, tr("No Selection"), tr("Please select a project to launch."))
            return

        if self._launch_thread is not None and self._launch_thread.isRunning():
            if self._launch_splash is not None:
                self._launch_splash.show()
                self._launch_splash.raise_()
            return

        project_name = record.name
        project_path = record.path
        
        from engine_wheel import editor_launch_script
        script = editor_launch_script(installed=self.launch_context.uses_installed_versions)

        from splash_screen import EngineSplashScreen
        from hub_resources import ICON_PATH

        splash = EngineSplashScreen(ICON_PATH, project_name, parent=None)
        splash.show()
        self._splash = splash
        self._launch_splash = splash

        self._launch_python_exe = ""
        self._launch_error = ""
        self._launch_thread = QThread()
        self._launch_worker = LaunchPreparationWorker(
            self.model,
            self.version_manager,
            project_path,
            self.launch_context,
        )
        self._launch_worker.moveToThread(self._launch_thread)

        def store_python_exe(python_exe: str):
            self._launch_python_exe = python_exe

        def store_error(message: str):
            self._launch_error = message

        def finish_preparation():
            error = self._launch_error
            python_exe = self._launch_python_exe
            worker = self._launch_worker
            thread = self._launch_thread
            self._launch_error = ""
            self._launch_python_exe = ""
            self._launch_worker = None
            self._launch_thread = None

            if error:
                splash.show_preparation_failure(error)
            elif python_exe:
                splash.launch(
                    python_exe,
                    script,
                    project_path,
                    detached=self.launch_context.uses_installed_versions,
                    lock_token=worker.lock_token,
                    runtime=worker.editor_runtime,
                )

            if worker is not None:
                worker.deleteLater()
            if thread is not None:
                thread.deleteLater()

        self._launch_cleanup_timer = QTimer()
        self._launch_cleanup_timer.setSingleShot(True)
        self._launch_cleanup_timer.setInterval(0)
        self._launch_cleanup_timer.timeout.connect(finish_preparation)

        self._launch_thread.started.connect(self._launch_worker.run)
        self._launch_worker.progress.connect(splash.set_preparation_status)
        self._launch_worker.finished.connect(store_python_exe)
        self._launch_worker.finished.connect(self._launch_thread.quit)
        self._launch_worker.error.connect(store_error)
        self._launch_worker.error.connect(self._launch_thread.quit)
        self._launch_thread.finished.connect(self._launch_cleanup_timer.start)

        # Let Qt paint the splash before any runtime preparation begins.
        QTimer.singleShot(0, self._launch_thread.start)

    def open_existing_project(self, parent):
        initial_dir = self.model.db.get_setting("last_open_project_dir", "") if self.model.db else ""
        selected_dir = QFileDialog.getExistingDirectory(
            parent, tr("Open Existing Infernux Project"), initial_dir,
        )
        if not selected_dir:
            return

        try:
            record, info = self.model.register_existing_project(selected_dir)
        except (ProjectPathError, RuntimeError) as exc:
            QMessageBox.critical(parent, tr("Cannot Open Project"), str(exc))
            return

        if self.model.db:
            self.model.db.set_setting("last_open_project_dir", os.path.dirname(info.path))
        self.project_list.refresh()
        self.project_list.select_project(record.project_id)

        if (
            self.launch_context.uses_installed_versions
            and info.engine_version
            and self.version_manager is not None
            and not self.version_manager.is_installed(info.engine_version)
        ):
            QMessageBox.information(
                parent,
                tr("Engine Version Not Installed"),
                f"Project '{info.name}' was added to Hub, but engine version "
                f"{info.engine_version} is not installed yet.\n\nOpen Installs to install it before launching.",
            )

    def remove_project(self, parent):
        record = self.project_list.get_selected_record()
        if record is None:
            QMessageBox.warning(parent, tr("No Selection"), tr("Please select a project to remove from Hub."))
            return

        confirm = QMessageBox.question(
            parent,
            tr("Remove Project from Hub"),
            f"Remove '{record.name}' from Infernux Hub?\n\n"
            f"{tr('Project files will not be deleted.')}\n{record.path}",
        )
        if confirm != QMessageBox.Yes:
            return

        self.model.remove_project(record.project_id)
        self.project_list.refresh()

    def relocate_project(self, parent):
        record = self.project_list.get_selected_record()
        if record is None:
            QMessageBox.warning(parent, tr("No Selection"), tr("Please select a project to relocate."))
            return

        initial_dir = record.path if os.path.isdir(record.path) else os.path.dirname(record.path)
        selected_dir = QFileDialog.getExistingDirectory(
            parent, tr("Relocate Infernux Project"), initial_dir,
        )
        if not selected_dir:
            return

        try:
            relocated, info = self.model.relocate_project(record.project_id, selected_dir)
        except (ProjectPathError, RuntimeError) as exc:
            QMessageBox.critical(parent, tr("Cannot Relocate Project"), str(exc))
            return

        if self.model.db:
            self.model.db.set_setting("last_open_project_dir", os.path.dirname(info.path))
        self.project_list.refresh()
        self.project_list.select_project(relocated.project_id)


    def create_project(self, parent):
        from view.new_project_view import NewProjectView

        if self._creation_dialog is not None:
            self._creation_dialog.show()
            self._creation_dialog.raise_()
            return

        dialog = NewProjectView(self.version_manager, self.runtime_manager, parent)
        self._creation_dialog = dialog
        try:
            accepted = dialog.exec() == QDialog.Accepted
        finally:
            self._creation_dialog = None
        if not accepted:
            return

        new_name, project_path, engine_version = dialog.get_data()
        if not new_name:
            QMessageBox.warning(parent, tr("Missing Name"), tr("Please enter a project name."))
            return
        if not project_path:
            QMessageBox.warning(parent, tr("Missing Location"), tr("Please choose a project location."))
            return
        if self.launch_context.uses_installed_versions and not engine_version:
            QMessageBox.warning(parent, tr("Missing Version"), tr("Please select an installed engine version."))
            return
        progress_dialog = CustomProgressDialog(parent)
        self._creation_dialog = progress_dialog
        progress_dialog.show()

        self._creation_thread = QThread(self)
        self._creation_worker = InitProjectWorker(
            self.model, new_name, project_path, engine_version
        )
        self._creation_worker.moveToThread(self._creation_thread)

        self._creation_thread.started.connect(self._creation_worker.run)
        self._creation_worker.progress.connect(progress_dialog.set_status)
        # quit() is thread-safe. It must not depend on the GUI event loop,
        # which is already stopping when aboutToQuit waits for initialization.
        self._creation_worker.finished.connect(self._creation_thread.quit, Qt.DirectConnection)
        self._creation_worker.error.connect(self._creation_thread.quit, Qt.DirectConnection)
        self._creation_thread.finished.connect(self._creation_worker.deleteLater)
        self._creation_thread.finished.connect(self._finish_creation)
        QApplication.instance().aboutToQuit.connect(self._wait_for_creation)
        self._creation_thread.start()

    @Slot()
    def _wait_for_creation(self):
        if self._creation_thread is not None:
            self._creation_thread.wait()
            self._finish_creation(show_dialogs=False)

    @Slot()
    def _finish_creation(self, *, show_dialogs=True):
        if self._creation_thread is None:
            return
        thread, worker, progress = self._creation_thread, self._creation_worker, self._creation_dialog
        thread.wait()
        try:
            record = None
            if worker.error_message:
                if show_dialogs:
                    QMessageBox.critical(progress, tr("Project Creation Failed"), worker.error_message)
            elif worker.project_dir:
                record = self.model.add_project(worker.name, worker.project_dir)
                if record is None and show_dialogs:
                    QMessageBox.warning(
                        progress, tr("Project Created"),
                        "The project was created successfully, but it is already registered in Hub.\n\n"
                        f"{worker.project_dir}",
                    )
            self.project_list.refresh()
            if record is not None:
                self.project_list.select_project(record.project_id)
        finally:
            QApplication.instance().aboutToQuit.disconnect(self._wait_for_creation)
            progress.finish()
            progress.deleteLater()
            thread.deleteLater()
            self._creation_thread = self._creation_worker = self._creation_dialog = None
