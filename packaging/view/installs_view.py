"""Installs page: engine versions, the managed runtime and optional toolchains."""

from __future__ import annotations

import os
import time

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QScrollArea, QFrame, QDialog, QFileDialog, QApplication, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from version_manager import (
    VersionManager, EngineVersion, display_release, hotfix_label, hotfix_number, latest_engine_versions,
    latest_releases, wheel_python_version,
)
from install_queue import InstallQueue
from android_support import AndroidSupportManager
from blender_support import BLENDER_VERSION, BlenderSupportManager
from i18n import tr
from view.forge import HazardStripe, StatusLed, chip, fmt_age, fmt_bytes, mono_label, set_kind, theme, track
from PySide6.QtGui import QColor, QPainter
from view.hover_widgets import AnimatedSurfaceFrame
from view.job_widgets import JobStrip
from view import dialogs
from view.forge import toast


def _update_install_buttons(view, queue):
    for button in view.findChildren(QPushButton):
        key = button.property("installationKey")
        if key:
            pending = queue.is_pending(key)
            button.setEnabled(not pending)
            button.setText(tr("In queue") if pending else button.property("idleText"))


def _configure_install_scroll_area(scroll: QScrollArea, container: QWidget) -> None:
    """Keep install lists on the Hub palette instead of the OS viewport palette."""
    scroll.setObjectName("installScrollArea")
    scroll.viewport().setObjectName("installViewport")
    scroll.viewport().setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    container.setObjectName("installListContainer")
    container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


def _section_header(title: str, description: str, actions: list[QWidget] = ()) -> QWidget:
    header = QWidget()
    layout = QHBoxLayout(header)
    layout.setContentsMargins(0, 4, 0, 4)
    layout.setSpacing(10)
    text = QVBoxLayout()
    text.setSpacing(4)
    title_label = QLabel(title)
    title_label.setObjectName("sectionTitle")
    text.addWidget(title_label)
    if description:
        description_label = QLabel(description)
        description_label.setObjectName("pageSubtitle")
        description_label.setWordWrap(True)
        text.addWidget(description_label)
    layout.addLayout(text, 1)
    for action in actions:
        layout.addWidget(action, 0, Qt.AlignmentFlag.AlignBottom)
    return header


class _ModuleCard(AnimatedSurfaceFrame):
    """Install unit plate: lamp, title, chips, detail text and one action."""

    def __init__(self, title: str, detail: str, *, lamp: str, chips: list[QLabel] = (), parent=None):
        super().__init__("versionCard", parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 16, 14)
        outer.setSpacing(4)
        row = QHBoxLayout()
        row.setSpacing(14)
        lamp_widget = StatusLed(lamp, size=10)
        row.addWidget(lamp_widget, 0, Qt.AlignmentFlag.AlignTop)
        info = QVBoxLayout()
        info.setSpacing(5)
        heading = QHBoxLayout()
        heading.setSpacing(8)
        title_label = QLabel(title)
        title_label.setObjectName("cardName")
        heading.addWidget(title_label)
        for item in chips:
            heading.addWidget(item)
        heading.addStretch()
        info.addLayout(heading)
        detail_label = QLabel(detail)
        detail_label.setObjectName("settingsDescription")
        detail_label.setWordWrap(True)
        detail_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.addWidget(detail_label)
        row.addLayout(info, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        row.addLayout(self.actions)
        row.setAlignment(self.actions, Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(row)
        self._outer = outer

    def attach_job(self, queue, key: str) -> JobStrip:
        strip = JobStrip(queue, key)
        self._outer.addWidget(strip)
        return strip


def _action_button(text: str, key: str, primary: bool) -> QPushButton:
    button = QPushButton(text)
    button.setObjectName("primaryBtn" if primary else "normalBtn")
    button.setFixedHeight(34)
    button.setMinimumWidth(104)
    button.setProperty("installationKey", key)
    button.setProperty("idleText", button.text())
    return button


# ─── Installed engine card ───────────────────────────────────────────

class _VersionCard(AnimatedSurfaceFrame):
    """Tape-spine card for one installed engine version."""

    remove_clicked = Signal(str)

    def __init__(self, version: str, wheel_path: str, parent=None):
        super().__init__("versionCard", parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(72)
        self._version = version

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 14, 12)
        layout.setSpacing(16)

        badge = QLabel(display_release(version))
        badge.setObjectName("versionBadge")
        badge.setToolTip(version)
        badge.setMinimumWidth(92)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(badge)

        info_col = QVBoxLayout()
        info_col.setSpacing(5)
        info_col.setContentsMargins(0, 0, 0, 0)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title = QLabel(f"Infernux {display_release(version)}")
        title.setObjectName("cardName")
        title_row.addWidget(title)
        if hotfix_label(version):
            title_row.addWidget(chip(tr("HOTFIX {number}", number=hotfix_number(version)), "paper"))
        python_version = wheel_python_version(wheel_path) if wheel_path else ""
        if python_version:
            title_row.addWidget(chip(f"PY {python_version}"))
        if wheel_path and os.path.isfile(wheel_path):
            title_row.addWidget(chip(fmt_bytes(os.path.getsize(wheel_path))))
        title_row.addWidget(chip(tr("INSTALLED"), "ok"))
        title_row.addStretch()
        info_col.addLayout(title_row)

        filename = os.path.basename(wheel_path) if wheel_path else "unknown"
        file_label = QLabel(filename)
        file_label.setObjectName("cardPath")
        file_label.setToolTip(wheel_path)
        file_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        info_col.addWidget(file_label)
        layout.addLayout(info_col, 1)

        reveal = QPushButton(tr("Show"))
        reveal.setObjectName("ghostBtn")
        reveal.setFixedHeight(30)
        reveal.setToolTip(tr("Show the engine wheel in its folder"))
        reveal.setEnabled(bool(wheel_path))
        reveal.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(wheel_path)))
        )
        layout.addWidget(reveal)

        remove_btn = QPushButton(tr("Remove"))
        remove_btn.setObjectName("dangerBtn")
        remove_btn.setFixedHeight(30)
        remove_btn.setMinimumWidth(84)
        remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self._version))
        layout.addWidget(remove_btn)


class _RuntimeCard(_ModuleCard):
    install_clicked = Signal(str)

    def __init__(self, version: str, path: str, *, default: bool, parent=None):
        title = (
            tr("Python {version} (default)", version=version)
            if default
            else tr("Python {version}", version=version)
        )
        detail = path if path else tr("Not installed. Install this runtime before using its engine wheels.")
        chips = [chip(tr("READY"), "ok") if path else chip(tr("MISSING"), "error")]
        super().__init__(title, detail, lamp="ok" if path else "error", chips=chips, parent=parent)
        button = _action_button(tr("Reinstall") if path else tr("Install"), f"python:{version}", not path)
        button.clicked.connect(lambda: self.install_clicked.emit(version))
        self.actions.addWidget(button)


class _AndroidSupportCard(_ModuleCard):
    install_clicked = Signal()

    def __init__(self, manager: AndroidSupportManager, parent=None, *, queue=None):
        status = manager.status()
        if status.installed:
            detail_text = tr(
                "SDK, NDK, JDK, Gradle and Android CPython are managed once for every project.\n{path}",
                path=str(status.root),
            )
            chips, lamp = [chip(tr("READY"), "ok")], "ok"
        elif status.error:
            detail_text = tr("Installed files need repair: {message}", message=status.error)
            chips, lamp = [chip(tr("REPAIR"), "warning")], "warn"
        else:
            detail_text = tr(
                "Required before the Android platform plugin can be imported. "
                "Large toolchains are installed once and shared by every project."
            )
            chips, lamp = [chip(tr("OPTIONAL"))], "idle"
        super().__init__(tr("Android compatibility"), detail_text, lamp=lamp, chips=chips, parent=parent)
        install = _action_button(
            tr("Repair") if status.installed or status.error else tr("Install"), "android", not status.installed,
        )
        install.clicked.connect(self.install_clicked.emit)
        self.actions.addWidget(install)
        if queue is not None:
            self.attach_job(queue, "android")


class _BlenderSupportCard(_ModuleCard):
    install_clicked = Signal()

    def __init__(self, manager: BlenderSupportManager, parent=None, *, queue=None):
        status = manager.status()
        if status.installed:
            detail_text = tr(
                "Used only by the Editor to import .blend source assets. Players never include Blender.\n{path}",
                path=str(status.executable),
            )
            chips, lamp = [chip(tr("READY"), "ok")], "ok"
        elif status.error:
            detail_text = tr("Installed files need repair: {message}", message=status.error)
            chips, lamp = [chip(tr("REPAIR"), "warning")], "warn"
        else:
            detail_text = tr(
                "Install the pinned Blender authoring tool once so every Infernux project can import .blend files."
            )
            chips, lamp = [chip(tr("OPTIONAL"))], "idle"
        super().__init__(tr("Blender {version}", version=BLENDER_VERSION), detail_text,
                         lamp=lamp, chips=chips, parent=parent)
        install = _action_button(
            tr("Repair") if status.installed or status.error else tr("Install"), "blender", not status.installed,
        )
        install.clicked.connect(self.install_clicked.emit)
        self.actions.addWidget(install)
        if queue is not None:
            self.attach_job(queue, "blender")


# ─── Install Editor dialog (pick version from public channels) ───────

class _FetchWorker(QThread):
    """Fetch available versions on a background thread."""
    loaded = Signal(list)  # list[EngineVersion]
    failed = Signal(str)

    def __init__(self, vm: VersionManager, parent):
        super().__init__(parent)
        self._vm = vm

    def run(self):
        try:
            versions = self._vm.list_versions(include_prerelease=True)
            self.loaded.emit(versions)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("Could not list engine versions")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def _published_date(value: str) -> str:
    return value[:10] if value else ""


class _VersionRow(AnimatedSurfaceFrame):
    """A selectable catalog row inside the Install Editor dialog."""

    activated = Signal(object)

    def __init__(self, ev: EngineVersion, parent=None):
        super().__init__("versionRow", parent)
        self.ev = ev
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(54)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 14, 8)
        layout.setSpacing(10)

        ver_label = QLabel(display_release(ev.version))
        ver_label.setObjectName("cardName")
        ver_label.setMinimumWidth(64)
        ver_label.setToolTip(ev.version)
        layout.addWidget(ver_label)
        if hotfix_label(ev.version):
            layout.addWidget(chip(tr("HOTFIX {number}", number=hotfix_number(ev.version)), "paper"))
        if ev.prerelease:
            layout.addWidget(chip(tr("PRE-RELEASE"), "warning"))
        if ev.python_version:
            layout.addWidget(chip(f"PY {ev.python_version}"))
        if ev.wheel_size:
            layout.addWidget(chip(fmt_bytes(ev.wheel_size)))
        if ev.sources:
            source_names = {"pypi": "PyPI", "github": "GitHub"}
            sources = mono_label(
                " · ".join(source_names.get(source, source) for source in ev.sources).upper(),
                "monoMeta", spacing=0.8,
            )
            layout.addWidget(sources)
        layout.addStretch()
        published = _published_date(ev.published_at)
        if published:
            layout.addWidget(mono_label(published, "monoMeta", spacing=0.4))
        if ev.compatibility_error:
            layout.addWidget(chip(tr("INCOMPATIBLE"), "error"))
        elif ev.installed:
            installed_label = QLabel(tr("Update available") if ev.update_available else tr("Installed"))
            installed_label.setObjectName("installedBadge")
            if ev.update_available:
                installed_label.setProperty("kind", "warning")
            layout.addWidget(installed_label)

    def mouseDoubleClickEvent(self, event):
        self.activated.emit(self.ev)

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        self.set_selected_animated(selected)


class InstallEditorDialog(QDialog):
    """Lists installable engine versions; cached catalog first, network second."""

    runtime_install_requested = Signal(str)
    AUTO_REFRESH_MS = 5 * 60 * 1000

    def __init__(self, version_manager: VersionManager, queue: InstallQueue, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Install Engine Version"))
        self.setMinimumSize(640, 520)
        self._vm = version_manager
        self._selected: EngineVersion | None = None
        self._rows: list[tuple[EngineVersion, _VersionRow]] = []
        self._queue = queue
        self._queue.job_finished.connect(self._refresh_selection)
        self._showing_cached = False
        self._preselect = ""

        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self._drag_origin = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        outer.addWidget(HazardStripe(5))
        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(24, 18, 24, 20)
        outer.addLayout(layout)

        head = QHBoxLayout()
        head.addWidget(mono_label(tr("ENGINE CATALOG  ·  PYPI + GITHUB"), "pageKicker", spacing=1.6))
        head.addStretch()
        close = QPushButton("×")
        close.setObjectName("iconBtn")
        close.setFixedSize(28, 28)
        close.setToolTip(tr("Close"))
        close.clicked.connect(self.reject)
        head.addWidget(close)
        layout.addLayout(head)
        title = QLabel(tr("Install an engine"))
        title.setObjectName("dialogTitle")
        layout.addWidget(title)
        subtitle = QLabel(tr("Pick a release. Each engine version is pinned to one managed Python runtime."))
        subtitle.setObjectName("dialogSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        catalog_row = QHBoxLayout()
        catalog_row.setSpacing(8)
        self._catalog_led = StatusLed("idle")
        catalog_row.addWidget(self._catalog_led, 0, Qt.AlignmentFlag.AlignVCenter)
        self._catalog_label = mono_label("", "monoMeta", spacing=0.8)
        catalog_row.addWidget(self._catalog_label, 1)
        self._btn_refresh = QPushButton(tr("Refresh"))
        self._btn_refresh.setObjectName("ghostBtn")
        self._btn_refresh.setFixedHeight(28)
        self._btn_refresh.clicked.connect(self._retry_fetch)
        catalog_row.addWidget(self._btn_refresh)
        layout.addLayout(catalog_row)

        self._status = QLabel(tr("Fetching available versions..."))
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        self._status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._status.setObjectName("settingsDescription")
        layout.addWidget(self._status)

        self._btn_retry = QPushButton(tr("Retry"))
        self._btn_retry.setObjectName("normalBtn")
        self._btn_retry.setMinimumHeight(34)
        self._btn_retry.clicked.connect(self._retry_fetch)
        self._btn_retry.hide()
        layout.addWidget(self._btn_retry)

        self._btn_runtime = QPushButton()
        self._btn_runtime.setObjectName("primaryBtn")
        self._btn_runtime.setMinimumHeight(34)
        self._btn_runtime.hide()
        self._btn_runtime.clicked.connect(self._on_install_runtime)
        layout.addWidget(self._btn_runtime)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.hide()
        self._container = QWidget()
        _configure_install_scroll_area(self._scroll, self._container)
        self._list_layout = QVBoxLayout(self._container)
        self._list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._list_layout.setSpacing(4)
        self._list_layout.setContentsMargins(0, 0, 6, 0)
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll, 1)

        btn_row = QHBoxLayout()
        btn_row.addWidget(mono_label(tr("DOUBLE-CLICK A RELEASE TO INSTALL"), "monoMeta", spacing=1.0))
        btn_row.addStretch()
        btn_cancel = QPushButton(tr("Close"))
        btn_cancel.setObjectName("normalBtn")
        btn_cancel.setFixedHeight(34)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        self._btn_install = QPushButton(tr("Install"))
        self._btn_install.setObjectName("primaryBtn")
        self._btn_install.setFixedHeight(34)
        self._btn_install.setMinimumWidth(110)
        self._btn_install.setEnabled(False)
        self._btn_install.clicked.connect(self._on_install)
        btn_row.addWidget(self._btn_install)
        layout.addLayout(btn_row)

        # Show the last known catalog immediately, then refresh in the background.
        cached = getattr(self._vm, "cached_versions", None)
        if cached is not None:
            try:
                versions = cached(include_prerelease=True)
            except Exception:
                versions = []
            if versions:
                self._showing_cached = True
                self._populate(versions)
                self._status.hide()

        self._fetch_thread = _FetchWorker(self._vm, self)
        self._fetch_thread.loaded.connect(self._on_versions_loaded)
        self._fetch_thread.failed.connect(self._on_versions_failed)
        QApplication.instance().aboutToQuit.connect(self._fetch_thread.wait)
        self._auto_refresh = QTimer(self)
        self._auto_refresh.setInterval(self.AUTO_REFRESH_MS)
        self._auto_refresh.timeout.connect(self._retry_fetch)
        self._set_catalog_state("busy")
        self._fetch_thread.start()

    # ── Catalog state ────────────────────────────────────────────────

    def _set_catalog_state(self, kind: str, detail: str = ""):
        self._catalog_led.set_kind(kind)
        stamp_reader = getattr(self._vm, "catalog_timestamp", None)
        stamp = stamp_reader() if stamp_reader is not None else None
        age = fmt_age(stamp).upper() if stamp else tr("NOT YET FETCHED")
        if kind == "busy":
            text = tr("CATALOG  ·  {age}  ·  REFRESHING", age=age)
        elif kind == "error":
            text = tr("OFFLINE  ·  CATALOG {age}", age=age)
        else:
            text = tr("CATALOG  ·  UPDATED {age}", age=age)
        self._catalog_label.setText(text + (f"  ·  {detail}" if detail else ""))
        self._btn_refresh.setEnabled(kind != "busy")

    def showEvent(self, event):
        self._auto_refresh.start()
        super().showEvent(event)
        owner = self.parentWidget().window() if self.parentWidget() is not None else None
        if owner is not None and owner.isVisible():
            center = owner.frameGeometry().center()
            self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)

    def paintEvent(self, event):
        palette = theme()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(palette.bg_surface))
        painter.setPen(QColor(palette.border_hover))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 60:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._btn_install.isEnabled():
            self._on_install()
            return
        super().keyPressEvent(event)

    def hideEvent(self, event):
        self._auto_refresh.stop()
        super().hideEvent(event)

    # ── Slots ────────────────────────────────────────────────────────

    def _clear_rows(self):
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        self._rows.clear()

    def _populate(self, versions: list):
        selected_version = self._selected.version if self._selected is not None else None
        self._clear_rows()
        self._scroll.show()
        # One row per version: its newest hotfix (an exact pin a project needs stays).
        for ev in latest_engine_versions(versions, keep=self._preselect):
            row = _VersionRow(ev)
            row.mousePressEvent = lambda _e, v=ev: self._select(v)
            row.activated.connect(self._activate)
            self._list_layout.addWidget(row)
            self._rows.append((ev, row))
        self._list_layout.addStretch()
        if self._preselect:
            selected_version = self._preselect
        if selected_version is not None:
            match = next((ev for ev, _row in self._rows if ev.version == selected_version), None)
            if match is not None:
                self._select(match)

    def _on_versions_loaded(self, versions: list):
        self._showing_cached = False
        self._status.hide()
        self._btn_retry.hide()
        self._set_catalog_state("ok")
        self._populate(versions)
        if not versions:
            self._status.setText(tr("No versions found."))
            self._status.show()
            return
        if self._selected is not None:
            self._select(self._selected)

    def _on_versions_failed(self, error: str):
        from hub_logging import hub_log_path
        self._set_catalog_state("error")
        message = (tr("Could not fetch engine versions.") + "\n" + error +
                   "\n\n" + tr("Hub log: {path}", path=str(hub_log_path())))
        if self._rows:
            # Keep the cached catalog usable offline; explain why it is stale.
            message = tr("Showing the cached catalog.") + " " + message
        self._status.setText(message)
        self._status.show()
        self._btn_retry.show()

    def _retry_fetch(self):
        if self._fetch_thread.isRunning():
            return
        self._btn_retry.hide()
        if not self._rows:
            self._status.setText(tr("Fetching available versions..."))
            self._status.show()
        self._set_catalog_state("busy")
        self._fetch_thread.start()

    def _select(self, ev: EngineVersion):
        self._selected = ev
        self._btn_runtime.hide()
        block_reason = self._vm.installation_block_reason(ev)
        if block_reason:
            self._btn_install.setEnabled(False)
            if (
                ev.python_version and ev.wheel_url
                and not ev.compatibility_error
                and not self._vm.is_python_runtime_installed(ev.python_version)
            ):
                self._status.setText(
                    tr(
                        "Infernux {engine} needs its managed runtime (Python {version}). "
                        "Install it here; no Python or Conda setup is required.",
                        engine=ev.version,
                        version=ev.python_version,
                    )
                )
                self._btn_runtime.setText(
                    tr("Install required runtime (Python {version})", version=ev.python_version)
                )
                self._btn_runtime.setEnabled(not self._queue.is_pending(f"python:{ev.python_version}"))
                self._btn_runtime.show()
            else:
                self._status.setText(block_reason)
            self._status.show()
            for candidate, row in self._rows:
                row.set_selected(candidate is ev)
            return
        pending = self._queue.is_pending(f"engine:{ev.version}")
        self._btn_install.setEnabled(
            (not ev.installed or ev.update_available) and bool(ev.wheel_url) and not pending
        )
        self._btn_install.setText(
            tr("In queue") if pending
            else tr("Update") if ev.update_available
            else tr("Installed") if ev.installed
            else tr("Install")
        )
        self._status.hide()
        for v, row in self._rows:
            row.set_selected(v is ev)

    def preselect(self, version: str) -> None:
        """Highlight ``version`` now (if listed) and after every catalog refresh."""
        self._preselect = version
        match = next((ev for ev, _row in self._rows if ev.version == version), None)
        if match is not None:
            self._select(match)
            row = next(row for ev, row in self._rows if ev is match)
            self._scroll.ensureWidgetVisible(row)

    def _activate(self, ev: EngineVersion):
        self._select(ev)
        if self._btn_install.isEnabled():
            self._on_install()

    def _on_install_runtime(self):
        engine = self._selected
        self.runtime_install_requested.emit(engine.python_version)
        self._select(engine)

    def _refresh_selection(self, _job):
        if self._selected is not None:
            self._select(self._selected)

    def _on_install(self):
        engine = self._selected
        if engine is None or not self._btn_install.isEnabled():
            return
        manager = self._vm
        self._queue.submit(
            f"engine:{engine.version}", f"Infernux {engine.version}",
            lambda report: manager.download_version(
                engine.version,
                on_progress=lambda done, total: report(tr("Downloading"), done, total),
                should_cancel=getattr(report, "should_cancel", None),
            ),
            cancellable=True,
        )
        self.accept()


# ─── Main Installs page ─────────────────────────────────────────────

class _EngineEmptyState(QFrame):
    def __init__(self, on_install, on_locate, parent=None):
        super().__init__(parent)
        self.setObjectName("emptyState")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        kicker = mono_label(tr("BAY EMPTY  ·  0 ENGINES"), "monoKicker", spacing=2.0)
        kicker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(kicker)
        label = QLabel(tr("No engine versions installed.\nClick 'Install Editor' or 'Locate' to add one."))
        label.setObjectName("emptyHint")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        buttons = QHBoxLayout()
        buttons.addStretch()
        locate = QPushButton(tr("Locate"))
        locate.setObjectName("normalBtn")
        locate.setFixedHeight(34)
        locate.clicked.connect(on_locate)
        buttons.addWidget(locate)
        install = QPushButton(tr("Install Editor"))
        install.setObjectName("primaryBtn")
        install.setFixedHeight(34)
        install.clicked.connect(on_install)
        buttons.addWidget(install)
        buttons.addStretch()
        layout.addLayout(buttons)


class _PendingEngineCard(AnimatedSurfaceFrame):
    """A transfer in flight, shown above installed engines until it lands."""

    def __init__(self, queue, job, parent=None):
        super().__init__("versionCard", parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 14, 12)
        layout.setSpacing(2)
        title = QLabel(job.title)
        title.setObjectName("cardName")
        layout.addWidget(title)
        layout.addWidget(JobStrip(queue, job.key))


class InstallsView(QWidget):
    """Engine installations only; all writes are owned by the shared queue."""

    runtime_install_requested = Signal(str)

    def __init__(self, version_manager, queue: InstallQueue, parent=None):
        super().__init__(parent)
        self._vm = version_manager
        self._queue = queue
        self._install_dialog = None
        self._pending_keys: tuple[str, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)
        self.btn_locate = QPushButton(tr("Locate"))
        self.btn_locate.setObjectName("normalBtn")
        self.btn_locate.setFixedHeight(34)
        self.btn_locate.setToolTip(tr("Add an Infernux .whl you already downloaded"))
        self.btn_locate.clicked.connect(self._on_locate)
        self.btn_install = QPushButton("+  " + tr("Install Editor"))
        self.btn_install.setObjectName("primaryBtn")
        self.btn_install.setFixedHeight(34)
        self.btn_install.clicked.connect(self._on_install_editor)
        layout.addWidget(_section_header(
            tr("Engine versions"),
            tr("Each engine version is cached once and shared by every project pinned to it."),
            [self.btn_locate, self.btn_install],
        ))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)
        self._container = QWidget()
        _configure_install_scroll_area(scroll, self._container)
        self._card_layout = QVBoxLayout(self._container)
        self._card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._card_layout.setContentsMargins(0, 0, 6, 0)
        self._card_layout.setSpacing(6)
        scroll.setWidget(self._container)
        queue.job_finished.connect(self._on_job_finished)
        queue.changed.connect(self._on_queue_changed)
        self.refresh()

    def _on_job_finished(self, _job):
        self.refresh()

    def _active_engine_keys(self) -> tuple[str, ...]:
        return tuple(job.key for job in self._queue.jobs
                     if job.key.startswith(("engine:", "engine-file:")) and job.active)

    def _on_queue_changed(self):
        if self._active_engine_keys() != self._pending_keys:
            self.refresh()

    def refresh(self):
        while self._card_layout.count():
            item = self._card_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        self._pending_keys = self._active_engine_keys()
        for job in self._queue.jobs:
            if job.key in self._pending_keys:
                self._card_layout.addWidget(_PendingEngineCard(self._queue, job))
        versions = latest_releases(self._vm.installed_versions())
        if not versions and not self._pending_keys:
            self._card_layout.addWidget(_EngineEmptyState(self._on_install_editor, self._on_locate))
        for version in versions:
            card = _VersionCard(version, self._vm.get_wheel_path(version) or "")
            card.remove_clicked.connect(self._on_remove_version)
            self._card_layout.addWidget(card)

    def _on_install_editor(self):
        if self._install_dialog is None:
            self._install_dialog = InstallEditorDialog(self._vm, self._queue, self)
            self._install_dialog.runtime_install_requested.connect(self.runtime_install_requested)
        elif not self._install_dialog.isVisible():
            self._install_dialog._retry_fetch()
        self._install_dialog.show()
        self._install_dialog.raise_()
        self._install_dialog.activateWindow()

    def open_install_dialog(self, preselect: str = "") -> None:
        self._on_install_editor()
        if preselect:
            self._install_dialog.preselect(preselect)

    def _on_locate(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Select Infernux Wheel"), "", "Wheel files (*.whl)",
        )
        if path:
            manager = self._vm
            self._queue.submit(
                f"engine-file:{os.path.normcase(os.path.abspath(path))}",
                os.path.basename(path),
                lambda report: manager.install_local_wheel(path),
            )

    def _on_remove_version(self, version):
        if self._queue.busy:
            dialogs.information(self, tr("Installation in progress"),
                                tr("Wait for installations to finish before removing an engine."))
            return
        wheel = self._vm.get_wheel_path(version) or ""
        base = display_release(version)
        releases = [item for item in self._vm.installed_versions() if display_release(item) == base]
        if dialogs.confirm(
            self, tr("Remove Infernux {version}?", version=base),
            tr("This deletes the cached wheel. Projects using this version will need to reinstall it."),
            confirm_label=tr("Remove engine"), destructive=True, kicker=tr("REMOVE ENGINE"),
            subject=(f"Infernux {base}", wheel),
        ):
            for release in releases or [version]:
                self._vm.remove_version(release)
            self.refresh()
            toast(self, tr("Removed Infernux {version}", version=base), "info")


class _ModuleView(QWidget):
    """Shared layout for the runtime and toolchain tabs."""

    def __init__(self, title: str, description: str, queue: InstallQueue, parent=None):
        super().__init__(parent)
        self._queue = queue
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(_section_header(title, description))
        self._cards = QVBoxLayout()
        self._cards.setSpacing(6)
        layout.addLayout(self._cards)
        layout.addStretch()
        queue.job_finished.connect(self._on_job_finished)
        queue.changed.connect(self._update_actions)

    def _update_actions(self):
        _update_install_buttons(self, self._queue)

    def _on_job_finished(self, _job):
        self.refresh()

    def _clear_cards(self):
        while self._cards.count():
            widget = self._cards.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()


class PythonRuntimesView(_ModuleView):
    def __init__(self, manager, queue: InstallQueue, parent=None):
        super().__init__(
            tr("Runtime environment"),
            tr("Required to run the editor. Hub installs and manages Python for you; no programming or environment setup is needed."),
            queue, parent,
        )
        self._manager = manager
        self.refresh()

    def refresh(self):
        self._clear_cards()
        for version in self._manager.supported_versions():
            card = _RuntimeCard(
                version, self._manager.get_runtime_path(version) or "",
                default=version == self._manager.default_version,
            )
            card.install_clicked.connect(self.install)
            card.attach_job(self._queue, f"python:{version}")
            self._cards.addWidget(card)
        self._update_actions()

    def install(self, version):
        manager = self._manager
        reinstall = manager.has_runtime(version)
        def prepare(report):
            report(tr("Preparing runtime"), 0, 0)
            status = lambda text: report(text, 0, 0)
            if reinstall:
                return manager.reinstall_runtime(version, on_status=status)
            return manager.ensure_runtime(version=version, on_status=status)
        return self._queue.submit(f"python:{version}", f"Python {version}", prepare)


class AndroidSupportView(_ModuleView):
    def __init__(self, manager, queue: InstallQueue, parent=None):
        super().__init__(
            tr("Android support"),
            tr("Only needed to build Android games. Install from the official channel; Hub manages the shared build tools for all projects."),
            queue, parent,
        )
        self._manager = manager
        self.refresh()

    def refresh(self):
        self._clear_cards()
        card = _AndroidSupportCard(self._manager, queue=self._queue)
        card.install_clicked.connect(self.install)
        self._cards.addWidget(card)
        self._update_actions()

    def install(self):
        manager = self._manager
        return self._queue.submit(
            "android", tr("Android support"),
            lambda report: manager.install(
                on_progress=lambda done, total: report(tr("Downloading"), done, total),
            ),
            # Interrupted Android downloads resume from the cached partial file.
            cancellable=True,
        )


class BlenderSupportView(_ModuleView):
    def __init__(self, manager, queue: InstallQueue, parent=None):
        super().__init__(
            tr("Model authoring"),
            tr(
                "Optional Editor tooling for importing Blender source assets. "
                "Hub owns one compatible installation shared by all projects."
            ),
            queue, parent,
        )
        self._manager = manager
        self.refresh()

    def refresh(self):
        self._clear_cards()
        card = _BlenderSupportCard(self._manager, queue=self._queue)
        card.install_clicked.connect(self.install)
        self._cards.addWidget(card)
        self._update_actions()

    def install(self):
        manager = self._manager
        return self._queue.submit(
            "blender",
            tr("Blender authoring support"),
            lambda report: manager.install(
                on_progress=lambda done, total: report(tr("Downloading"), done, total),
                on_stage=lambda stage: report(tr(stage), 0, 0),
            ),
            cancellable=True,
        )
