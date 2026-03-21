"""Installs page — lists installed engine versions, install from GitHub or locate .whl."""

from __future__ import annotations

import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QFrame, QDialog, QFileDialog, QMessageBox,
    QProgressBar, QApplication,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject

from version_manager import VersionManager, EngineVersion


# ─── Version card (one per installed version) ────────────────────────

class _VersionCard(QFrame):
    """Card showing a single installed engine version."""

    remove_clicked = Signal(str)  # version string

    def __init__(self, version: str, wheel_path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("versionCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(64)
        self._version = version

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(14)

        # Version badge
        badge = QLabel(version)
        badge.setObjectName("versionBadge")
        layout.addWidget(badge)

        # Wheel filename / path
        info_col = QVBoxLayout()
        info_col.setSpacing(2)
        info_col.setContentsMargins(0, 0, 0, 0)

        filename = os.path.basename(wheel_path) if wheel_path else "unknown"
        file_label = QLabel(filename)
        file_label.setObjectName("cardPath")
        file_label.setToolTip(wheel_path)
        info_col.addWidget(file_label)

        size_text = ""
        if wheel_path and os.path.isfile(wheel_path):
            size_mb = os.path.getsize(wheel_path) / (1024 * 1024)
            size_text = f"{size_mb:.1f} MB"
        size_label = QLabel(size_text)
        size_label.setObjectName("cardDate")
        info_col.addWidget(size_label)

        layout.addLayout(info_col, 1)

        # Remove button
        remove_btn = QPushButton("Remove")
        remove_btn.setObjectName("dangerBtn")
        remove_btn.setFixedHeight(30)
        remove_btn.setFixedWidth(80)
        remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self._version))
        layout.addWidget(remove_btn)


# ─── Install Editor dialog (pick version from GitHub releases) ───────

class _FetchWorker(QObject):
    """Fetch available versions on a background thread."""
    finished = Signal(list)  # list[EngineVersion]

    def __init__(self, vm: VersionManager):
        super().__init__()
        self._vm = vm

    def run(self):
        versions = self._vm.list_versions(include_prerelease=True)
        self.finished.emit(versions)


class _DownloadWorker(QObject):
    """Download a version wheel on a background thread."""
    progress = Signal(int, int)  # downloaded, total
    finished = Signal(str)  # wheel path
    error = Signal(str)

    def __init__(self, vm: VersionManager, version: str):
        super().__init__()
        self._vm = vm
        self._version = version

    def run(self):
        try:
            path = self._vm.download_version(
                self._version, on_progress=lambda d, t: self.progress.emit(d, t)
            )
            self.finished.emit(path)
        except Exception as exc:
            self.error.emit(str(exc))


class _VersionRow(QFrame):
    """A selectable row inside the Install Editor dialog."""

    def __init__(self, ev: EngineVersion, parent=None):
        super().__init__(parent)
        self.ev = ev
        self.setObjectName("versionRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 14, 6)
        layout.setSpacing(10)

        ver_label = QLabel(ev.display_name)
        ver_label.setObjectName("cardName")
        layout.addWidget(ver_label)

        if ev.wheel_size:
            size_mb = ev.wheel_size / (1024 * 1024)
            size_label = QLabel(f"{size_mb:.1f} MB")
            size_label.setObjectName("cardDate")
            layout.addWidget(size_label)

        layout.addStretch()

        if ev.installed:
            installed_label = QLabel("Installed")
            installed_label.setObjectName("installedBadge")
            layout.addWidget(installed_label)

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)


class InstallEditorDialog(QDialog):
    """Dialog that lists available versions from GitHub for installation."""

    def __init__(self, version_manager: VersionManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Install Engine Version")
        self.setMinimumSize(520, 420)
        self._vm = version_manager
        self._selected: EngineVersion | None = None
        self._rows: list[tuple[EngineVersion, _VersionRow]] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self._status = QLabel("Fetching available versions...")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        # Scroll area for version rows
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.hide()
        self._container = QWidget()
        self._list_layout = QVBoxLayout(self._container)
        self._list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._list_layout.setSpacing(4)
        self._list_layout.setContentsMargins(0, 0, 4, 0)
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll, 1)

        # Progress bar (hidden until download starts)
        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setObjectName("normalBtn")
        btn_cancel.setFixedHeight(34)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        self._btn_install = QPushButton("Install")
        self._btn_install.setObjectName("primaryBtn")
        self._btn_install.setFixedHeight(34)
        self._btn_install.setMinimumWidth(100)
        self._btn_install.setEnabled(False)
        self._btn_install.clicked.connect(self._on_install)
        btn_row.addWidget(self._btn_install)

        layout.addLayout(btn_row)

        # Kick off fetch in background
        self._fetch_thread = QThread()
        self._fetch_worker = _FetchWorker(self._vm)
        self._fetch_worker.moveToThread(self._fetch_thread)
        self._fetch_thread.started.connect(self._fetch_worker.run)
        self._fetch_worker.finished.connect(self._on_versions_loaded)
        self._fetch_worker.finished.connect(self._fetch_thread.quit)
        self._fetch_thread.start()

    # ── Slots ────────────────────────────────────────────────────────

    def _on_versions_loaded(self, versions: list):
        self._status.hide()
        self._scroll.show()

        if not versions:
            self._status.setText("No versions found.")
            self._status.show()
            return

        for ev in versions:
            row = _VersionRow(ev)
            row.mousePressEvent = lambda _e, v=ev: self._select(v)
            self._list_layout.addWidget(row)
            self._rows.append((ev, row))

        self._list_layout.addStretch()

    def _select(self, ev: EngineVersion):
        self._selected = ev
        self._btn_install.setEnabled(not ev.installed and bool(ev.wheel_url))
        for v, row in self._rows:
            row.set_selected(v.version == ev.version)

    def _on_install(self):
        if not self._selected or self._selected.installed:
            return
        # Start download
        self._btn_install.setEnabled(False)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.show()

        self._dl_thread = QThread()
        self._dl_worker = _DownloadWorker(self._vm, self._selected.version)
        self._dl_worker.moveToThread(self._dl_thread)
        self._dl_thread.started.connect(self._dl_worker.run)
        self._dl_worker.progress.connect(self._on_dl_progress)
        self._dl_worker.finished.connect(self._on_dl_finished)
        self._dl_worker.error.connect(self._on_dl_error)
        self._dl_worker.finished.connect(self._dl_thread.quit)
        self._dl_worker.error.connect(self._dl_thread.quit)
        self._dl_thread.start()

    def _on_dl_progress(self, downloaded: int, total: int):
        if total > 0:
            self._progress_bar.setValue(int(downloaded * 100 / total))

    def _on_dl_finished(self, _path: str):
        self._progress_bar.hide()
        self.accept()

    def _on_dl_error(self, msg: str):
        self._progress_bar.hide()
        QMessageBox.critical(self, "Download Failed", msg)
        self._btn_install.setEnabled(True)


# ─── Main Installs page ─────────────────────────────────────────────

class InstallsView(QWidget):
    """Page showing installed engine versions with install/locate actions."""

    def __init__(self, version_manager: VersionManager, runtime_manager=None, parent=None):
        super().__init__(parent)
        self._vm = version_manager
        self._runtime_manager = runtime_manager

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        self._runtime_card = QFrame()
        self._runtime_card.setObjectName("versionCard")
        self._runtime_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        runtime_layout = QHBoxLayout(self._runtime_card)
        runtime_layout.setContentsMargins(16, 12, 16, 12)
        runtime_layout.setSpacing(14)

        runtime_info = QVBoxLayout()
        runtime_info.setSpacing(4)
        runtime_info.setContentsMargins(0, 0, 0, 0)
        self._runtime_status = QLabel()
        self._runtime_status.setObjectName("cardName")
        runtime_info.addWidget(self._runtime_status)
        self._runtime_path = QLabel()
        self._runtime_path.setObjectName("cardPath")
        self._runtime_path.setWordWrap(True)
        runtime_info.addWidget(self._runtime_path)
        runtime_layout.addLayout(runtime_info, 1)

        self._runtime_button = QPushButton("Install Python 3.12")
        self._runtime_button.setObjectName("primaryBtn")
        self._runtime_button.setFixedHeight(34)
        self._runtime_button.clicked.connect(self._on_install_python)
        runtime_layout.addWidget(self._runtime_button)

        layout.addWidget(self._runtime_card)

        # ── Header ───────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 0, 12)

        title = QLabel("Installs")
        title.setObjectName("pageTitle")
        header.addWidget(title, alignment=Qt.AlignmentFlag.AlignLeft)
        header.addStretch()

        self.btn_locate = QPushButton("Locate")
        self.btn_locate.setObjectName("normalBtn")
        self.btn_locate.setFixedHeight(36)
        self.btn_locate.setMinimumWidth(90)
        self.btn_locate.clicked.connect(self._on_locate)
        header.addWidget(self.btn_locate)

        spacer = QLabel("")
        spacer.setFixedWidth(8)
        header.addWidget(spacer)

        self.btn_install = QPushButton("Install Editor")
        self.btn_install.setObjectName("primaryBtn")
        self.btn_install.setFixedHeight(36)
        self.btn_install.setMinimumWidth(130)
        self.btn_install.clicked.connect(self._on_install_editor)
        header.addWidget(self.btn_install)

        layout.addLayout(header)

        # ── Version list (scrollable) ────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        self._container = QWidget()
        self._container.setStyleSheet("background: transparent;")
        self._card_layout = QVBoxLayout(self._container)
        self._card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._card_layout.setSpacing(6)
        self._card_layout.setContentsMargins(0, 0, 4, 0)
        scroll.setWidget(self._container)

        self.refresh()

    # ── Public API ───────────────────────────────────────────────────

    def refresh(self):
        self._refresh_runtime_status()

        # Clear existing cards
        while self._card_layout.count():
            item = self._card_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        versions = self._vm.installed_versions()
        if not versions:
            empty = QLabel("No engine versions installed.\nClick 'Install Editor' or 'Locate' to add one.")
            empty.setObjectName("emptyHint")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._card_layout.addWidget(empty)
        else:
            for ver in versions:
                wheel = self._vm.get_wheel_path(ver) or ""
                card = _VersionCard(ver, wheel)
                card.remove_clicked.connect(self._on_remove_version)
                self._card_layout.addWidget(card)

        self._card_layout.addStretch()

    def _refresh_runtime_status(self):
        if self._runtime_manager is None:
            self._runtime_card.hide()
            return

        self._runtime_card.show()
        runtime_path = self._runtime_manager.get_runtime_path()
        if runtime_path:
            self._runtime_status.setText("Python 3.12 runtime is ready")
            self._runtime_path.setText(runtime_path)
            self._runtime_button.setText("Reinstall Python 3.12")
        else:
            self._runtime_status.setText("Python 3.12 runtime is missing")
            self._runtime_path.setText(
                "The Hub has a bundled python-3.12.0-amd64.exe installer and will install Python 3.12 into _inner/python312."
            )
            self._runtime_button.setText("Install Python 3.12")

    # ── Actions ──────────────────────────────────────────────────────

    def _on_install_editor(self):
        dlg = InstallEditorDialog(self._vm, parent=self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh()

    def _on_install_python(self):
        if self._runtime_manager is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            python_exe = self._runtime_manager.ensure_runtime()
        except Exception as exc:
            QMessageBox.critical(self, "Python Installation Failed", str(exc))
        else:
            QMessageBox.information(
                self,
                "Python Installed",
                f"Python 3.12 is ready at:\n{python_exe}",
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.refresh()

    def _on_locate(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select InfEngine Wheel",
            "",
            "Wheel files (*.whl)",
        )
        if not path:
            return

        try:
            version = self._vm.install_local_wheel(path)
            QMessageBox.information(
                self, "Version Installed",
                f"InfEngine {version} has been installed from the selected wheel.",
            )
            self.refresh()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid Wheel", str(exc))

    def _on_remove_version(self, version: str):
        confirm = QMessageBox.question(
            self,
            "Remove Version",
            f"Remove InfEngine {version}?\n\n"
            "This deletes the cached wheel. Projects using this version "
            "will need to reinstall it.",
        )
        if confirm != QMessageBox.Yes:
            return
        self._vm.remove_version(version)
        self.refresh()
