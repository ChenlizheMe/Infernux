import os
import sys
sys.dont_write_bytecode = True

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QMessageBox, QDialog,
    QHBoxLayout, QVBoxLayout, QSizePolicy, QStackedWidget,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QFontDatabase

from ui_project_list import ProjectListPane
from database import ProjectDatabase
from style import StyleManager
from hub_resources import ICON_PATH, FONT_PATH
from hub_utils import is_frozen
from python_runtime import PythonRuntimeManager
from version_manager import VersionManager

from model.project_model import ProjectModel
from viewmodel.control_pane_viewmodel import ControlPaneViewModel
from view.control_pane_view import ControlPane
from view.sidebar_view import SidebarView
from view.installs_view import InstallsView, PythonRuntimeInstallDialog


class GameEngineLauncher(QMainWindow):
    def __init__(self) -> None:
        self._own_app = False
        if QApplication.instance() is None:
            self._own_app = True
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        super().__init__()

        # Load custom engine font
        font_id = QFontDatabase.addApplicationFont(FONT_PATH)
        if font_id >= 0:
            QFontDatabase.applicationFontFamilies(font_id)

        # Apply global dark theme
        self.app.is_dark_theme = True
        self.app.setStyleSheet(StyleManager.get_stylesheet(self.app.is_dark_theme))

        self.setWindowTitle("Infernux Hub")
        self.setWindowIcon(QIcon(ICON_PATH))
        self.resize(1080, 720)

        # Database & version manager
        self.db = ProjectDatabase()
        self.version_manager = VersionManager()
        self.runtime_manager = PythonRuntimeManager()

        # ── Root layout: sidebar | content ───────────────────────────
        central = QWidget(self)
        central.setObjectName("central")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar
        self.sidebar = SidebarView(parent=central)
        root_layout.addWidget(self.sidebar)

        # Stacked pages
        self.pages = QStackedWidget()
        root_layout.addWidget(self.pages, 1)

        # ── Page 0: Projects ─────────────────────────────────────────
        projects_page = QWidget()
        projects_layout = QVBoxLayout(projects_page)
        projects_layout.setContentsMargins(28, 24, 28, 24)
        projects_layout.setSpacing(16)

        self.project_list = ProjectListPane(self.db, parent=projects_page)
        model = ProjectModel(self.db, self.version_manager, self.runtime_manager)
        viewmodel = ControlPaneViewModel(
            model,
            self.project_list,
            self.version_manager,
            self.runtime_manager,
        )
        self.controls = ControlPane(viewmodel, parent=projects_page)

        self.controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.project_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        projects_layout.addWidget(self.controls, 0)
        projects_layout.addWidget(self.project_list, 1)

        self.pages.addWidget(projects_page)

        # ── Page 1: Installs ─────────────────────────────────────────
        installs_page = QWidget()
        installs_layout = QVBoxLayout(installs_page)
        installs_layout.setContentsMargins(28, 24, 28, 24)
        installs_layout.setSpacing(0)

        self.installs_view = InstallsView(self.version_manager, self.runtime_manager, parent=installs_page)
        installs_layout.addWidget(self.installs_view)

        self.pages.addWidget(installs_page)

        # ── Sidebar → page switching ─────────────────────────────────
        self.sidebar.page_changed.connect(self._on_page_changed)

        # Cleanup on close
        self.app.aboutToQuit.connect(self._on_close)

    def _on_page_changed(self, index: int):
        self.pages.setCurrentIndex(index)
        # Refresh installs when switching to that page
        if index == 1:
            self.installs_view.refresh()

    def run(self):
        self.show()
        if is_frozen():
            QTimer.singleShot(0, self._bootstrap_python_runtime)
        if self._own_app:
            sys.exit(self.app.exec())

    def _bootstrap_python_runtime(self):
        if self.runtime_manager.has_runtime():
            self.installs_view.refresh()
            return

        QMessageBox.information(
            self,
            "Python 3.12 Setup",
            "Infernux Hub needs Python 3.12 to create and launch projects.\n\n"
            "The recommended path is to install Infernux Hub through the installer. The installer or standalone Hub will\n"
            "download the matching full Python 3.12 installer for this machine when needed and install it under\n"
            "C:\\Users\\Public\\InfernuxHub.  Each project then receives its own full copy of the runtime.",
        )

        dlg = PythonRuntimeInstallDialog(self.runtime_manager, self)
        if dlg.exec() != QDialog.Accepted and dlg.error_text:
            QMessageBox.warning(
                self,
                "Python 3.12 Not Ready",
                dlg.error_text,
            )
        self.installs_view.refresh()

    def _on_close(self):
        self.db.close()


def _handle_uninstall() -> int:
    """Remove registry entries, Start Menu shortcut, and optionally the install directory."""
    if sys.platform == "darwin":
        return _handle_uninstall_macos()
    if sys.platform != "win32":
        return 1
    import winreg

    # Read install location from registry before removing the key.
    install_dir = ""
    reg_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\InfernuxHub"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_key) as key:
            install_dir, _ = winreg.QueryValueEx(key, "InstallLocation")
    except OSError:
        pass

    # Remove registry entry
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, reg_key)
    except OSError:
        pass

    # Remove Start Menu shortcut
    try:
        import ctypes.wintypes
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(None, 0x0002, None, 0, buf)
        if buf.value:
            import shutil as _shutil
            _shutil.rmtree(os.path.join(buf.value, "Infernux Hub"), ignore_errors=True)
    except Exception:
        pass

    # Ask user if they want to remove install files
    app = QApplication.instance() or QApplication(sys.argv)
    answer = QMessageBox.question(
        None,
        "Uninstall Infernux Hub",
        "Registry entries and shortcuts have been removed.\n\n"
        f"Do you also want to delete the installation folder?\n{install_dir}",
    )
    if answer == QMessageBox.Yes and install_dir and os.path.isdir(install_dir):
        import shutil as _shutil
        _shutil.rmtree(install_dir, ignore_errors=True)

    QMessageBox.information(None, "Uninstall Complete", "Infernux Hub has been uninstalled.")
    return 0


def _handle_uninstall_macos() -> int:
    """Remove Infernux Hub from macOS."""
    import shutil as _shutil

    app = QApplication.instance() or QApplication(sys.argv)

    # Typical macOS install / config locations
    config_dir = os.path.expanduser("~/.config/Infernux")
    app_link = os.path.expanduser("~/Applications/Infernux Hub")
    runtime_dir = os.path.expanduser("~/.infernux")

    dirs_to_remove = [d for d in (config_dir, app_link) if os.path.exists(d)]
    if runtime_dir and os.path.isdir(runtime_dir):
        dirs_to_remove.append(runtime_dir)

    if dirs_to_remove:
        answer = QMessageBox.question(
            None,
            "Uninstall Infernux Hub",
            "Do you want to remove Infernux Hub configuration and cached data?\n\n"
            + "\n".join(dirs_to_remove),
        )
        if answer == QMessageBox.Yes:
            for d in dirs_to_remove:
                _shutil.rmtree(d, ignore_errors=True)

    QMessageBox.information(None, "Uninstall Complete", "Infernux Hub has been uninstalled.")
    return 0


if __name__ == "__main__":
    if "--uninstall" in sys.argv:
        raise SystemExit(_handle_uninstall())
    launcher = GameEngineLauncher()
    launcher.run()
