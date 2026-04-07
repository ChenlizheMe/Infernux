from __future__ import annotations

import os
import shutil
import sys

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from installer.install_python_runtime import install_runtime_for_app
import logging

if sys.platform == "win32":
    import winreg


def _resource_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "resources")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def _payload_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "payload")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dist", "Infernux Hub")


_UNINSTALL_REG_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\InfernuxHub"


def _write_registry(install_dir: str) -> None:
    """Register Infernux Hub in Windows Add/Remove Programs."""
    if sys.platform != "win32":
        return
    exe_path = os.path.join(install_dir, "Infernux Hub.exe")
    uninstall_cmd = f'"{os.path.join(install_dir, "Infernux Hub.exe")}" --uninstall'
    icon_path = os.path.join(install_dir, "InfernuxHubData", "runtime", "icon.png")
    if not os.path.isfile(icon_path):
        icon_path = exe_path

    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _UNINSTALL_REG_KEY, 0, winreg.KEY_WRITE) as key:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, "Infernux Hub")
            winreg.SetValueEx(key, "UninstallString", 0, winreg.REG_SZ, uninstall_cmd)
            winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, icon_path)
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, install_dir)
            winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, "Infernux")
            winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)
    except OSError as _exc:
        logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
        pass


def _remove_registry() -> None:
    """Remove Infernux Hub from Windows Add/Remove Programs."""
    if sys.platform != "win32":
        return
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _UNINSTALL_REG_KEY)
    except OSError as _exc:
        logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
        pass


def _create_start_menu_shortcut(install_dir: str) -> None:
    """Create a Start Menu shortcut (Windows) or Applications symlink (macOS)."""
    if sys.platform == "darwin":
        # macOS: create a symlink in ~/Applications
        try:
            apps_dir = os.path.expanduser("~/Applications")
            os.makedirs(apps_dir, exist_ok=True)
            link_path = os.path.join(apps_dir, "Infernux Hub")
            if os.path.lexists(link_path):
                os.remove(link_path)
            os.symlink(install_dir, link_path)
        except Exception as _exc:
            logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
            pass
        return
    if sys.platform != "win32":
        return
    try:
        import ctypes.wintypes
        CSIDL_PROGRAMS = 0x0002
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PROGRAMS, None, 0, buf)
        programs_dir = buf.value
        if not programs_dir:
            return

        shortcut_dir = os.path.join(programs_dir, "Infernux Hub")
        os.makedirs(shortcut_dir, exist_ok=True)
        shortcut_path = os.path.join(shortcut_dir, "Infernux Hub.lnk")
        exe_path = os.path.join(install_dir, "Infernux Hub.exe")

        # Use PowerShell to create .lnk — avoids pywin32 dependency
        ps_script = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut("{shortcut_path}"); '
            f'$s.TargetPath = "{exe_path}"; '
            f'$s.WorkingDirectory = "{install_dir}"; '
            f'$s.Description = "Infernux Hub"; '
            f'$s.Save()'
        )
        import subprocess
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            creationflags=0x08000000,
        )
    except Exception as _exc:
        logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
        pass


def _remove_start_menu_shortcut() -> None:
    """Remove Start Menu shortcut (Windows) or Applications symlink (macOS)."""
    if sys.platform == "darwin":
        try:
            link_path = os.path.expanduser("~/Applications/Infernux Hub")
            if os.path.lexists(link_path):
                os.remove(link_path)
        except Exception as _exc:
            logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
            pass
        return
    if sys.platform != "win32":
        return
    try:
        import ctypes.wintypes
        CSIDL_PROGRAMS = 0x0002
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_PROGRAMS, None, 0, buf)
        programs_dir = buf.value
        if not programs_dir:
            return
        shortcut_dir = os.path.join(programs_dir, "Infernux Hub")
        shutil.rmtree(shortcut_dir, ignore_errors=True)
    except Exception as _exc:
        logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
        pass


class InstallWorker(QObject):
    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, install_dir: str):
        super().__init__()
        self.install_dir = install_dir

    def run(self) -> None:
        try:
            payload_dir = os.path.abspath(_payload_dir())
            if not os.path.isdir(payload_dir):
                raise RuntimeError(f"Hub payload directory not found: {payload_dir}")

            os.makedirs(self.install_dir, exist_ok=True)
            self.progress.emit("Copying Infernux Hub files...")
            shutil.copytree(payload_dir, self.install_dir, dirs_exist_ok=True)

            self.progress.emit("Installing private Python 3.12 runtime...")
            install_runtime_for_app(self.install_dir, progress_callback=self.progress.emit)

            self.progress.emit("Registering Infernux Hub...")
            _write_registry(self.install_dir)
            _create_start_menu_shortcut(self.install_dir)

            self.finished.emit(self.install_dir)
        except Exception as exc:
            self.error.emit(str(exc))


class InstallerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._installed_dir = ""
        self._thread: QThread | None = None
        self._worker: InstallWorker | None = None

        self.setWindowTitle("Infernux Hub Installer")
        self.setFixedSize(600, 320)

        icon_path = os.path.join(_resource_dir(), "icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        default_dir = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "Infernux Hub")

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        title = QLabel("Install Infernux Hub")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        root.addWidget(title)

        intro = QLabel(
            "This installer copies Infernux Hub onto your machine. During setup, it will download and prepare "
            "a managed full Python 3.12 runtime under C:\\Users\\Public\\InfernuxHub for use by all projects."
        )
        intro.setWordWrap(True)
        intro.setMinimumHeight(56)
        intro.setContentsMargins(0, 0, 0, 6)
        root.addWidget(intro)

        root.addWidget(QLabel("Install location"))

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(default_dir)
        path_row.addWidget(self.path_edit, 1)
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse)
        path_row.addWidget(browse_button)
        root.addLayout(path_row)

        self.status_label = QLabel("Ready to install.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        root.addWidget(self.progress_bar)

        button_row = QHBoxLayout()
        button_row.addStretch()
        self.install_button = QPushButton("Install")
        self.install_button.clicked.connect(self._start_install)
        button_row.addWidget(self.install_button)
        self.launch_button = QPushButton("Launch Hub")
        self.launch_button.setEnabled(False)
        self.launch_button.clicked.connect(self._launch_hub)
        button_row.addWidget(self.launch_button)
        root.addLayout(button_row)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select installation directory", self.path_edit.text())
        if folder:
            self.path_edit.setText(folder)

    def _start_install(self) -> None:
        install_dir = os.path.abspath(self.path_edit.text().strip())
        if not install_dir:
            QMessageBox.warning(self, "Missing Directory", "Please select an installation directory.")
            return

        if os.path.exists(install_dir) and os.listdir(install_dir):
            answer = QMessageBox.question(
                self,
                "Directory Not Empty",
                "The selected directory already contains files. Continue and overwrite matching files?",
            )
            if answer != QMessageBox.Yes:
                return

        self.install_button.setEnabled(False)
        self.launch_button.setEnabled(False)
        self.status_label.setText("Starting installation...")
        self.progress_bar.setRange(0, 0)

        self._thread = QThread(self)
        self._worker = InstallWorker(install_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.finished.connect(self._on_install_finished)
        self._worker.error.connect(self._on_install_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_install_finished(self, install_dir: str) -> None:
        self._installed_dir = install_dir
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setText(f"Installation completed successfully. Installed to: {install_dir}")
        self.install_button.setEnabled(True)
        self.launch_button.setEnabled(True)

    def _on_install_failed(self, message: str) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.install_button.setEnabled(True)
        self.status_label.setText("Installation failed.")
        QMessageBox.critical(self, "Installation Failed", message)

    def _launch_hub(self) -> None:
        if not self._installed_dir:
            return
        exe_path = os.path.join(self._installed_dir, "Infernux Hub.exe")
        if not os.path.isfile(exe_path):
            QMessageBox.warning(self, "Launch Failed", f"Hub executable not found: {exe_path}")
            return
        os.startfile(exe_path)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = InstallerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())