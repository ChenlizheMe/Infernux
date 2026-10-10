from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QFrame,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from installer.install_application import HubInstallTransaction
from installer.install_python_runtime import install_runtime_for_app
from installer_safety import install_target_error, is_recognized_install_dir
from i18n import tr
from python_runtime_catalog import DEFAULT_PYTHON_RUNTIME

if sys.platform == "win32":
    import winreg


_BUNDLED_PYTHON_VERSION = DEFAULT_PYTHON_RUNTIME.series


def _hub_executable_name() -> str:
    return "Infernux Hub.exe" if sys.platform == "win32" else "Infernux Hub"


def _default_install_dir() -> str:
    if sys.platform == "win32":
        return os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"), "Infernux Hub"
        )
    if sys.platform == "linux":
        return os.path.expanduser("~/.local/opt/InfernuxHub")
    raise RuntimeError(f"Infernux Hub has no installer contract for {sys.platform}")


def _resource_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")


def _payload_dir() -> str:
    if "__compiled__" in globals():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "payload")
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "out", "package", "hub"
    )


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


def _create_start_menu_shortcut(install_dir: str) -> None:
    """Register the installed Hub in the host desktop application menu."""
    if sys.platform == "linux":
        applications_dir = os.path.expanduser("~/.local/share/applications")
        os.makedirs(applications_dir, exist_ok=True)
        executable = os.path.join(install_dir, _hub_executable_name())
        icon = os.path.join(install_dir, "resources", "icon.png")
        escape = lambda value: value.replace("\\", "\\\\").replace('"', '\\"')
        desktop_entry = (
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=Infernux Hub\n"
            f'Exec="{escape(executable)}"\n'
            f'Icon={escape(icon)}\n'
            "Terminal=false\n"
            "Categories=Development;Game;\n"
        )
        destination = os.path.join(applications_dir, "infernux-hub.desktop")
        with open(destination, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(desktop_entry)
        os.chmod(destination, 0o755)
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

        # Paths are PowerShell literals, never expandable strings or code.
        def literal(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        # Use PowerShell to create .lnk — avoids pywin32 dependency
        ps_script = (
            f'$ws = New-Object -ComObject WScript.Shell; '
            f'$s = $ws.CreateShortcut({literal(shortcut_path)}); '
            f'$s.TargetPath = {literal(exe_path)}; '
            f'$s.WorkingDirectory = {literal(install_dir)}; '
            f'$s.Description = "Infernux Hub"; '
            f'$s.Save()'
        )
        import subprocess

        from hub_utils import merge_child_env_utf8

        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            creationflags=0x08000000,
            env=merge_child_env_utf8(),
        )
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

            with HubInstallTransaction(payload_dir, self.install_dir, progress=self.progress.emit) as installation:
                self.progress.emit(tr("Preparing Infernux Hub files..."))
                installation.prepare()

                self.progress.emit(tr("Replacing the installed Infernux Hub..."))
                installation.activate()

                self.progress.emit(
                    tr(
                        "Deploying private Python {version} runtime...",
                        version=_BUNDLED_PYTHON_VERSION,
                    )
                )
                install_runtime_for_app(self.install_dir, progress_callback=self.progress.emit)

                self.progress.emit(tr("Registering Infernux Hub..."))
                _write_registry(self.install_dir)
                _create_start_menu_shortcut(self.install_dir)
                installation.commit()

            self.finished.emit(self.install_dir)
        except Exception as exc:
            self.error.emit(str(exc))


class _BrandRail(QWidget):
    """Left rail of the installer: icon, product plate and a hazard spine."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        icon_path = os.path.join(_resource_dir(), "icon.png")
        self._icon = QPixmap(icon_path) if os.path.isfile(icon_path) else QPixmap()

    def paintEvent(self, _event):
        from view.forge import caps_font, qcolor, theme
        palette = theme()
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor(palette.sidebar_bg))
        stripe = 8
        painter.fillRect(rect.width() - stripe, 0, stripe, rect.height(), QColor(palette.bg_deep))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(palette.accent_fill))
        x0 = rect.width() - stripe
        for y in range(-stripe * 2, rect.height() + stripe, stripe * 2):
            painter.drawPolygon(QPolygonF([QPointF(x0, y + stripe), QPointF(x0 + stripe, y),
                                           QPointF(x0 + stripe, y + stripe), QPointF(x0, y + stripe * 2)]))
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        if not self._icon.isNull():
            painter.drawPixmap(QRectF(24, 28, 48, 48).toRect(), self._icon)
        painter.setPen(QColor(palette.text_primary))
        painter.setFont(caps_font(17, tracking=3.0))
        painter.drawText(QRectF(24, 92, 180, 24), Qt.AlignmentFlag.AlignLeft, "INFERNUX")
        painter.setPen(QColor(palette.accent))
        painter.setFont(caps_font(10, tracking=1.6))
        painter.drawText(QRectF(24, 120, 180, 16), Qt.AlignmentFlag.AlignLeft, tr("HUB INSTALLER"))
        painter.setPen(qcolor(palette.text_muted, 220))
        painter.setFont(caps_font(9, tracking=1.0))
        lines = (f"PYTHON {_BUNDLED_PYTHON_VERSION}", tr("ISOLATED RUNTIME"), tr("USER-SCOPE INSTALL"))
        for index, line in enumerate(lines):
            painter.drawText(QRectF(24, rect.height() - 92 + index * 18, 180, 14), Qt.AlignmentFlag.AlignLeft, line)
        painter.end()


class InstallerWindow(QWidget):
    def __init__(self):
        super().__init__()
        from view.forge import SegmentMeter, StatusLed, mono_label
        self._installed_dir = ""
        self._thread: QThread | None = None
        self._worker: InstallWorker | None = None

        self.setWindowTitle(tr("Infernux Hub Installer"))
        self.setObjectName("central")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedWidth(780)

        icon_path = os.path.join(_resource_dir(), "icon.png")
        if os.path.isfile(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        default_dir = _default_install_dir()

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(_BrandRail())
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(30, 26, 30, 22)
        root.setSpacing(10)
        outer.addWidget(content, 1)

        root.addWidget(mono_label("§01  /  " + tr("INSTALL"), "pageKicker", spacing=1.8))
        title = QLabel(tr("Install Infernux Hub"))
        title.setObjectName("dialogTitle")
        root.addWidget(title)

        intro = QLabel(
            tr(
                "This installer copies Infernux Hub and its isolated Python {version} "
                "runtime onto your machine. It does not install, upgrade, remove, "
                "register, or modify any Python already on your system.",
                version=_BUNDLED_PYTHON_VERSION,
            )
        )
        intro.setObjectName("dialogSubtitle")
        intro.setWordWrap(True)
        root.addWidget(intro)
        root.addSpacing(6)

        changes_box = QFrame()
        changes_box.setObjectName("subjectCard")
        changes_layout = QVBoxLayout(changes_box)
        changes_layout.setContentsMargins(14, 10, 14, 12)
        changes_layout.setSpacing(4)
        changes_layout.addWidget(mono_label(tr("Installation changes").upper(), "fieldLabel", spacing=1.4))
        changes = QLabel(
            tr(
                "The installer adds Infernux Hub application files, an isolated "
                "Python runtime, an application-menu shortcut, and an uninstall "
                "entry. Automatic update checks are enabled by default. "
                "Installing updates requires confirmation."
            )
        )
        changes.setObjectName("settingsDescription")
        changes.setWordWrap(True)
        changes_layout.addWidget(changes)
        policy = QLabel(
            '<a style="color: #ff6e6e;" href="https://infernux-engine.com/code-signing-policy.html">'
            + tr("Code signing policy and privacy disclosure")
            + "</a>"
        )
        policy.setTextFormat(Qt.TextFormat.RichText)
        policy.setOpenExternalLinks(True)
        changes_layout.addWidget(policy)
        root.addWidget(changes_box)
        root.addSpacing(6)

        root.addWidget(mono_label(tr("Install location").upper(), "fieldLabel", spacing=1.4))
        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        self.path_edit = QLineEdit(default_dir)
        self.path_edit.setFixedHeight(36)
        path_row.addWidget(self.path_edit, 1)
        browse_button = QPushButton(tr("Browse..."))
        browse_button.setFixedSize(104, 36)
        browse_button.clicked.connect(self._browse)
        path_row.addWidget(browse_button)
        root.addLayout(path_row)
        root.addSpacing(8)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self._led = StatusLed("idle")
        status_row.addWidget(self._led, 0, Qt.AlignmentFlag.AlignVCenter)
        self.status_label = QLabel(tr("Ready to install."))
        self.status_label.setObjectName("monoValue")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        root.addLayout(status_row)
        self._meter = SegmentMeter(height=6)
        root.addWidget(self._meter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()

        button_row = QHBoxLayout()
        button_row.setSpacing(8)
        button_row.addStretch()
        self.install_button = QPushButton(tr("Install"))
        self.install_button.setObjectName("primaryBtn")
        self.install_button.setFixedHeight(36)
        self.install_button.setMinimumWidth(120)
        self.install_button.clicked.connect(self._start_install)
        button_row.addWidget(self.install_button)
        self.launch_button = QPushButton(tr("Launch Hub"))
        self.launch_button.setFixedHeight(36)
        self.launch_button.setMinimumWidth(120)
        self.launch_button.setEnabled(False)
        self.launch_button.clicked.connect(self._launch_hub)
        button_row.addWidget(self.launch_button)
        root.addStretch(1)
        root.addLayout(button_row)
        # Word-wrapped disclosure decides the height in either language; polish first
        # so the stylesheet's type sizes are measured, not the platform defaults.
        self.ensurePolished()
        for child in self.findChildren(QWidget):
            child.ensurePolished()
        # Nested frames do not forward height-for-width; pin the wrapped text
        # to its measured height at the fixed content width.
        content_width = self.width() - 220 - 60
        intro.setMinimumHeight(intro.heightForWidth(content_width))
        changes.setMinimumHeight(changes.heightForWidth(content_width - 28))
        # Slack goes to the stretch above the buttons, never into clipped text.
        self.setFixedHeight(max(500, outer.totalHeightForWidth(self.width()) + 48))

    def _set_progress(self, state: str) -> None:
        if state == "busy":
            self._meter.set_fraction(None)
            self._meter.set_tone("accent")
            self._led.set_kind("busy")
        elif state == "done":
            self._meter.set_fraction(1.0)
            self._meter.set_tone("ok")
            self._led.set_kind("ok")
        elif state == "failed":
            self._meter.set_fraction(0.0)
            self._led.set_kind("error")
        else:
            self._meter.set_fraction(0.0)
            self._led.set_kind("idle")

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Select installation directory"), self.path_edit.text())
        if folder:
            self.path_edit.setText(folder)

    def _start_install(self) -> None:
        from view import dialogs
        install_dir = os.path.abspath(self.path_edit.text().strip())
        if not install_dir:
            dialogs.warning(self, tr("Missing Directory"), tr("Please select an installation directory."))
            return

        safety_error = install_target_error(install_dir)
        if safety_error:
            dialogs.critical(self, tr("Unsafe Install Location"), safety_error)
            return

        if os.path.exists(install_dir) and os.listdir(install_dir):
            if not is_recognized_install_dir(install_dir):
                dialogs.critical(
                    self,
                    tr("Unrecognized Install Location"),
                    tr(
                        "The selected directory contains files but is not a recognized Infernux Hub installation. "
                        "Choose the existing Hub application directory or an empty folder."
                    ),
                )
                return
            warning = (
                "The selected Infernux Hub directory already contains files. Updating it will close any running "
                "Infernux Hub process and replace the application files. Continue?"
            )
            answer = dialogs.question(
                self,
                tr("Directory Not Empty"),
                warning,
                dialogs.Yes | dialogs.No, dialogs.No,
                labels={dialogs.Yes: tr("Replace and update"), dialogs.No: tr("Cancel")},
            )
            if answer != dialogs.Yes:
                return

        self.install_button.setEnabled(False)
        self.launch_button.setEnabled(False)
        self.status_label.setText(tr("Starting installation..."))
        self.progress_bar.setRange(0, 0)
        self._set_progress("busy")

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
        self._set_progress("done")
        self.status_label.setText(tr("Installation completed successfully. Installed to: {path}", path=install_dir))
        self.install_button.setEnabled(True)
        self.install_button.setObjectName("normalBtn")
        self.launch_button.setObjectName("primaryBtn")
        for button in (self.install_button, self.launch_button):
            button.style().unpolish(button)
            button.style().polish(button)
        self.launch_button.setEnabled(True)
        self.launch_button.setFocus()

    def _on_install_failed(self, message: str) -> None:
        from view import dialogs
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self._set_progress("failed")
        self.install_button.setEnabled(True)
        self.status_label.setText(tr("Installation failed."))
        summary, _, rest = message.partition("\n")
        dialogs.critical(self, tr("Installation Failed"), summary, detail=message if rest.strip() else "")

    def _launch_hub(self) -> None:
        from view import dialogs
        if not self._installed_dir:
            return
        exe_path = os.path.join(self._installed_dir, _hub_executable_name())
        if not os.path.isfile(exe_path):
            dialogs.warning(self, tr("Launch Failed"), tr("Hub executable not found: {path}", path=exe_path))
            return
        if sys.platform == "win32":
            os.startfile(exe_path)
        elif sys.platform == "linux":
            subprocess.Popen(
                [exe_path],
                cwd=self._installed_dir,
                start_new_session=True,
            )
        else:
            raise RuntimeError(
                f"Infernux Hub has no launch contract for {sys.platform}"
            )


def _apply_installer_theme(app) -> None:
    from PySide6.QtGui import QFontDatabase
    from style import StyleManager
    from view.forge import ui_font
    fonts = os.path.join(_resource_dir(), "fonts")
    if os.path.isdir(fonts):
        for name in sorted(os.listdir(fonts)):
            if name.lower().endswith(".ttf"):
                QFontDatabase.addApplicationFont(os.path.join(fonts, name))
    app.setFont(ui_font(13))
    app.is_dark_theme = True
    app.setStyleSheet(StyleManager.get_stylesheet(True))


def main() -> int:
    from hub_logging import configure_logging
    from hub_network import configure_system_certificates
    configure_logging()
    configure_system_certificates()
    app = QApplication.instance() or QApplication(sys.argv)
    _apply_installer_theme(app)
    window = InstallerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
