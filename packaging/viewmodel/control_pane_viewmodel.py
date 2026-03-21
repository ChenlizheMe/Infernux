import os
from PySide6.QtWidgets import (
    QMessageBox, QDialog, QVBoxLayout, QLabel, QProgressBar
)
from PySide6.QtCore import QThread, Signal, QObject, QTimer, Qt
from model.project_model import ProjectModel
from hub_utils import is_frozen
import random


class CustomProgressDialog(QDialog):
    """Indeterminate progress dialog shown during project initialization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Initializing")
        self.setWindowModality(Qt.WindowModal)
        self.setFixedSize(340, 110)

        self.label = QLabel("Preparing project...", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.messages = [
            "Setting up project structure...",
            "Copying engine libraries...",
            "Configuring virtual environment...",
            "Preparing asset folders...",
            "Almost there...",
        ]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._rotate_message)
        self.timer.start(2000)

    def _rotate_message(self):
        self.label.setText(random.choice(self.messages))


class InitProjectWorker(QObject):
    """Worker that runs project initialization on a background thread."""
    finished = Signal()
    error = Signal(str)

    def __init__(self, model, name, path, engine_version=""):
        super().__init__()
        self.model = model
        self.name = name
        self.path = path
        self.engine_version = engine_version

    def run(self):
        try:
            self.model.init_project_folder(self.name, self.path, self.engine_version)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit()


class ControlPaneViewModel:
    def __init__(self, model, project_list, version_manager=None, runtime_manager=None):
        self.model = model
        self.project_list = project_list
        self.version_manager = version_manager
        self.runtime_manager = runtime_manager

    def launch_project(self, parent):
        project_name = self.project_list.get_selected_project()
        if not project_name:
            QMessageBox.warning(parent, "No Selection", "Please select a project to launch.")
            return
        
        import sys
        
        project_path = os.path.join(self.project_list.get_selected_project_path(), project_name)

        # Determine Python interpreter based on mode
        if is_frozen():
            # Packaged Hub → always use the project's .venv
            python_exe = ProjectModel._get_venv_python(project_path)
            if not os.path.isfile(python_exe):
                QMessageBox.critical(
                    parent,
                    "Missing .venv",
                    f"Project virtual environment not found at:\n"
                    f"{os.path.join(project_path, '.venv')}\n\n"
                    "Please recreate the project or reinstall the engine version.",
                )
                return
        else:
            # Dev mode → use current Python (conda / system)
            python_exe = sys.executable
        
        script = (
            'import sys;'
            'from InfEngine.engine import release_engine;'
            'from InfEngine.lib import LogLevel;'
            'release_engine(engine_log_level=LogLevel.Info, project_path=sys.argv[1])'
        )

        from splash_screen import EngineSplashScreen
        from hub_resources import ICON_PATH

        splash = EngineSplashScreen(ICON_PATH, project_name, parent=None)
        splash.show()
        splash.launch(python_exe, script, project_path, detached=is_frozen())
        self._splash = splash

    def delete_project(self, parent):
        project_name = self.project_list.get_selected_project()
        if not project_name:
            QMessageBox.warning(parent, "No Selection", "Please select a project to delete.")
            return

        confirm = QMessageBox.question(
            parent,
            "Confirm Deletion",
            f"Are you sure you want to delete the project '{project_name}'?",
        )
        if confirm != QMessageBox.Yes:
            return

        self.model.delete_project(project_name)
        self.project_list.refresh()

    def create_project(self, parent):
        from view.new_project_view import NewProjectView

        if is_frozen() and self.runtime_manager is not None and not self.runtime_manager.has_runtime():
            QMessageBox.warning(
                parent,
                "Python 3.12 Required",
                "Python 3.12 is not installed yet.\n"
                "Open the Installs page or restart the Hub and let it finish runtime setup first.",
            )
            return

        dialog = NewProjectView(self.version_manager, self.runtime_manager, parent)
        if dialog.exec() != QDialog.Accepted:
            return

        new_name, project_path, engine_version = dialog.get_data()
        if not new_name:
            QMessageBox.warning(parent, "Missing Name", "Please enter a project name.")
            return
        if not project_path:
            QMessageBox.warning(parent, "Missing Location", "Please choose a project location.")
            return

        if not self.model.add_project(new_name, project_path):
            QMessageBox.critical(parent, "Duplicate Name", f"Project '{new_name}' already exists.")
            return

        progress_dialog = CustomProgressDialog(parent)
        progress_dialog.show()

        self._init_error: str = ""

        self.thread = QThread()
        self.worker = InitProjectWorker(self.model, new_name, project_path, engine_version)
        self.worker.moveToThread(self.thread)

        def _store_error(msg: str):
            # Called from the worker thread — only store the message.
            self._init_error = msg

        def _cleanup():
            # Guaranteed to run on the main thread (QTimer fires in main loop).
            progress_dialog.accept()
            if self._init_error:
                self.model.delete_project(new_name)
                QMessageBox.critical(
                    parent, "Project Creation Failed", self._init_error,
                )
                self._init_error = ""
            self.project_list.refresh()
            self.worker.deleteLater()
            self.thread.deleteLater()

        # QTimer in the main thread — its start() slot is auto-QueuedConnection
        # when invoked from worker thread, so _cleanup always runs on main thread.
        self._cleanup_timer = QTimer()
        self._cleanup_timer.setSingleShot(True)
        self._cleanup_timer.setInterval(0)
        self._cleanup_timer.timeout.connect(_cleanup)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(_store_error)
        self.worker.error.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_timer.start)

        self.thread.start()
