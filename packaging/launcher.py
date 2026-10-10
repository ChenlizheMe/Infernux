import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _REPO_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

# Hub runs before an engine is installed. Its startup must stay stdlib/Qt-only;
# importing a submodule of infernux would execute the engine's native facade.
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.dont_write_bytecode = True

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog,
    QHBoxLayout, QVBoxLayout, QSizePolicy, QStackedWidget,
    QGraphicsOpacityEffect, QSystemTrayIcon, QMenu, QTabWidget, QLabel,
)
from PySide6.QtCore import QByteArray, QObject, QRect, QThread, Qt, QTimer, QPropertyAnimation, QEasingCurve, Slot
from PySide6.QtGui import QIcon, QFontDatabase, QKeySequence, QShortcut

from ui_project_list import ProjectListPane
from database import ProjectDatabase
from style import StyleManager
from hub_resources import ICON_PATH, FONT_PATHS
from hub_utils import HubLaunchContext, get_app_dir, is_frozen
from python_runtime import PythonRuntimeManager
from android_support import AndroidSupportManager
from blender_support import BlenderSupportManager
from version_manager import VersionManager

from model.project_model import ProjectModel
from viewmodel.control_pane_viewmodel import ControlPaneViewModel
from view.control_pane_view import ControlPane
from view.sidebar_view import SidebarView
from view.installs_view import (
    AndroidSupportView,
    BlenderSupportView,
    InstallsView,
    PythonRuntimesView,
)
from view.install_queue_panel import InstallQueuePanel
from install_queue import InstallQueue
from installer_safety import can_remove_install_dir
from hub_uninstall import remove_application
from i18n import configure_language, tr
from view.hover_widgets import ensure_hover_animation_filter
from view.forge import Backdrop, PageHeader, ToastHost, ui_font
from view import dialogs
import time
import logging


class GameEngineLauncher(QMainWindow):
    def __init__(self, launch_context: HubLaunchContext | None = None, *, install_queue=None) -> None:
        self.launch_context = launch_context or HubLaunchContext.current()
        self._own_app = False
        if QApplication.instance() is None:
            self._own_app = True
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        super().__init__()

        # Configure localization before constructing any visible widget.
        self.db = ProjectDatabase()
        configure_language(self.db.get_setting("language", "system"))

        install_hub_fonts(self.app)
        self._started = time.monotonic()

        # Apply the persisted Hub theme before constructing visible pages.
        self.app.is_dark_theme = self.db.get_setting("theme", "dark") != "light"
        self.app.setStyleSheet(StyleManager.get_stylesheet(self.app.is_dark_theme))
        ensure_hover_animation_filter(self.app)

        self.setWindowTitle("Infernux Hub")
        self.setWindowIcon(QIcon(ICON_PATH))
        self._apply_initial_geometry()
        self.setAcceptDrops(True)

        # Version and runtime managers
        self.runtime_manager = PythonRuntimeManager()
        self.android_support_manager = AndroidSupportManager()
        self.android_support_manager.activate_environment()
        self.blender_support_manager = BlenderSupportManager()
        self.blender_support_manager.activate_environment()
        self.version_manager = VersionManager(self.runtime_manager)
        self.install_queue = install_queue if install_queue is not None else InstallQueue(self.app)
        self._exit_when_idle = False
        self.install_queue.idle.connect(self._on_queue_idle)
        self.install_queue.job_finished.connect(self._on_installation_finished)

        # Services belong to this window, independently of translated pages.
        model = ProjectModel(self.db, self.version_manager, self.runtime_manager)
        self.viewmodel = ControlPaneViewModel(
            model, None, self.version_manager, self.runtime_manager,
            launch_context=self.launch_context,
        )
        self.viewmodel.setParent(self)
        from view.update_dialog import UpdateController
        self.update_controller = UpdateController(self)
        self.update_controller.check_finished.connect(self._on_startup_update_check_finished)
        self._startup_update_pending = False
        from view.notification_dialog import HubNotificationController
        self.notification_controller = HubNotificationController(
            self, self.db, open_installs=lambda: self.sidebar.select_page(1),
        )
        self._language_wait_connections = []
        self._language_refresh_timer = QTimer(self)
        self._language_refresh_timer.setSingleShot(True)
        self._language_refresh_timer.timeout.connect(self._refresh_language_pages)

        self._build_pages()
        self.update_controller.update_available.connect(self._on_update_available)
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(6 * 60 * 60 * 1000)
        self._update_timer.timeout.connect(self._periodic_update_check)
        self._update_timer.start()
        self._install_shortcuts()
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(60_000)
        self._telemetry_timer.timeout.connect(self._refresh_telemetry)
        self._telemetry_timer.start()
        self._catalog_prefetch = None
        self.tray = QSystemTrayIcon(QIcon(ICON_PATH), self)
        self.tray.setToolTip("Infernux Hub")
        self._build_tray_menu()
        self.tray.activated.connect(self._on_tray_activated)
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.app.setQuitOnLastWindowClosed(False)
            self.tray.show()
        self.app.aboutToQuit.connect(self._on_close)

    def _build_pages(self):
        # ── Root layout: sidebar | content ───────────────────────────
        central = Backdrop(self)
        central.setObjectName("central")
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Sidebar
        self.sidebar = SidebarView(parent=central, started=getattr(self, "_started", None))
        root_layout.addWidget(self.sidebar)
        self.sidebar.update_requested.connect(self._show_update)
        self.sidebar.ignited.connect(self._afterburner)

        # Stacked pages
        self.pages = QStackedWidget()
        root_layout.addWidget(self.pages, 1)

        # ── Page 0: Projects ─────────────────────────────────────────
        projects_page = QWidget()
        projects_layout = QVBoxLayout(projects_page)
        projects_layout.setContentsMargins(36, 28, 36, 24)
        projects_layout.setSpacing(18)

        self.project_list = ProjectListPane(
            self.db, self.version_manager, parent=projects_page,
        )
        self.viewmodel.project_list = self.project_list
        self.project_list.remove_requested.connect(self._remove_project_from_card)
        self.project_list.launch_requested.connect(self._launch_project_from_card)
        self.project_list.install_requested.connect(self._install_engine_for_project)
        self.project_list.relocate_requested.connect(
            lambda project_id: self.viewmodel.relocate_project(self.controls, project_id))
        self.controls = ControlPane(self.viewmodel, parent=projects_page)
        self.project_list.create_requested.connect(lambda: self.viewmodel.create_project(self.controls))
        self.project_list.open_requested.connect(lambda: self.viewmodel.open_existing_project(self.controls))

        self.controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.project_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        projects_layout.addWidget(self.controls, 0)
        projects_layout.addWidget(self.project_list, 1)

        self.pages.addWidget(projects_page)

        # ── Page 1: Installs ─────────────────────────────────────────
        installs_page = QWidget()
        installs_layout = QVBoxLayout(installs_page)
        installs_layout.setContentsMargins(36, 28, 36, 24)
        installs_layout.setSpacing(18)
        installs_layout.addWidget(PageHeader(
            "02", tr("INSTALLS"), tr("Installs"),
            tr("Engines, runtimes and optional toolchains. Everything here is shared by all projects."),
        ))
        self.install_tabs = QTabWidget()
        self.install_tabs.setObjectName("installTabs")
        installs_layout.addWidget(self.install_tabs)

        self.installs_view = InstallsView(
            self.version_manager,
            self.install_queue,
            parent=installs_page,
        )
        self.python_view = PythonRuntimesView(
            self.runtime_manager, self.install_queue, settings=self.db
        )
        self.android_view = AndroidSupportView(self.android_support_manager, self.install_queue)
        self.blender_view = BlenderSupportView(self.blender_support_manager, self.install_queue)
        for view, label in (
            (self.installs_view, tr("Engine versions")),
            (self.python_view, tr("Runtime environment")),
            (self.android_view, tr("Android support")),
            (self.blender_view, tr("Model authoring")),
        ):
            self.install_tabs.addTab(view, label)
        self.install_tabs.currentChanged.connect(self._on_install_tab_changed)

        self.pages.addWidget(installs_page)

        # ── Page 2: Settings ─────────────────────────────────────────
        from view.settings_view import SettingsView

        settings_page = QWidget()
        settings_layout = QVBoxLayout(settings_page)
        settings_layout.setContentsMargins(36, 28, 36, 24)
        self.settings_view = SettingsView(self.db, parent=settings_page)
        settings_layout.addWidget(self.settings_view)
        self.pages.addWidget(settings_page)

        self.settings_view.update_check_requested.connect(
            lambda: self.update_controller.check(silent=False)
        )
        self.settings_view.language_changed.connect(self._on_language_changed)

        # ── Page 3: Discussion ──────────────────────────────────────
        from view.discussion_view import DiscussionView

        self.discussion_view = DiscussionView(parent=self.pages)
        self.pages.addWidget(self.discussion_view)

        self.installs_view.runtime_install_requested.connect(self._install_required_runtime)

        self.install_panel = InstallQueuePanel(self.install_queue, central)
        self.install_panel.layout_changed.connect(self._position_install_panel)
        self._position_install_panel()
        self.toast_host = ToastHost(central, left_inset=self.sidebar.width())
        pending = getattr(getattr(self, "update_controller", None), "pending_update", None)
        if pending is not None and pending.update is not None:
            self.sidebar.show_update(pending.update.target_version)
        self.sidebar.page_changed.connect(self._on_page_changed)
        self.install_queue.changed.connect(self._update_queue_badge)
        self._update_queue_badge()
        QTimer.singleShot(0, self._refresh_telemetry)

    def _build_tray_menu(self):
        old_menu = self.tray.contextMenu()
        tray_menu = QMenu(self)
        tray_menu.addAction(tr("Open Infernux Hub"), self._restore_window)
        tray_menu.addAction(tr("Downloads and installs"), self._show_installs)
        tray_menu.addSeparator()
        tray_menu.addAction(tr("Exit"), self.request_quit)
        self.tray.setContextMenu(tray_menu)
        if old_menu is not None:
            old_menu.deleteLater()

    @Slot(object)
    def _on_installation_finished(self, job):
        # Exactly one subscription follows the current project page.
        self.project_list.refresh()
        self._refresh_telemetry()
        if getattr(job, "key", "").startswith("python:") and job.state == "succeeded":
            self._seed_bundled_engines()

    def _update_queue_badge(self):
        active = sum(job.active for job in self.install_queue.jobs)
        self.sidebar.set_badge(1, str(active) if active else "")
        self.sidebar.set_transfer_active(bool(active))

    # ── Window geometry ──────────────────────────────────────────────

    GEOMETRY_SETTING = "window_geometry"

    def _apply_initial_geometry(self):
        """Size the window from the screen it opens on, then restore the user's layout.

        Fixed pixel sizes looked tiny on 4K panels and overflowed 1366×768
        laptops at 125% scaling. The default now follows the available area
        in device-independent pixels, so every display shows the same layout.
        """
        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 800)
        minimum_w = min(980, available.width() - 40)
        minimum_h = min(620, available.height() - 40)
        self.setMinimumSize(max(720, minimum_w), max(520, minimum_h))
        width = max(self.minimumWidth(), min(1520, round(available.width() * 0.78)))
        height = max(self.minimumHeight(), min(960, round(available.height() * 0.82), round(width / 1.52)))
        self.resize(width, height)
        self.move(available.x() + (available.width() - width) // 2,
                  available.y() + (available.height() - height) // 2)
        saved = self.db.get_setting(self.GEOMETRY_SETTING, "")
        if saved:
            try:
                restored = self.restoreGeometry(QByteArray.fromBase64(saved.encode("ascii")))
            except (ValueError, UnicodeError):
                restored = False
            if restored and not any(
                candidate.availableGeometry().intersects(self.frameGeometry())
                for candidate in QApplication.screens()
            ):
                self.resize(width, height)
                self.move(available.x() + (available.width() - width) // 2,
                          available.y() + (available.height() - height) // 2)

    def _save_geometry(self):
        try:
            self.db.set_setting(self.GEOMETRY_SETTING, bytes(self.saveGeometry().toBase64()).decode("ascii"))
        except Exception:
            logging.getLogger(__name__).debug("Could not persist the Hub window geometry", exc_info=True)

    # ── Shortcuts, drag and drop, small delights ─────────────────────

    def _install_shortcuts(self):
        bindings = (
            (QKeySequence.StandardKey.New, lambda: self.viewmodel.create_project(self.controls)),
            (QKeySequence.StandardKey.Open, lambda: self.viewmodel.open_existing_project(self.controls)),
            (QKeySequence("Ctrl+1"), lambda: self.sidebar.select_page(0)),
            (QKeySequence("Ctrl+2"), lambda: self.sidebar.select_page(1)),
            (QKeySequence("Ctrl+3"), lambda: self.sidebar.select_page(3)),
            (QKeySequence("Ctrl+,"), lambda: self.sidebar.select_page(2)),
        )
        self._shortcuts = []
        for sequence, callback in bindings:
            shortcut = QShortcut(sequence, self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    @staticmethod
    def _dropped_folders(event) -> list[str]:
        data = event.mimeData()
        if not data.hasUrls():
            return []
        return [url.toLocalFile() for url in data.urls()
                if url.isLocalFile() and os.path.isdir(url.toLocalFile())]

    def dragEnterEvent(self, event):
        if self._dropped_folders(event):
            event.acceptProposedAction()
            if hasattr(self, "toast_host"):
                from view.forge import toast
                if not getattr(self, "_drop_hint_shown", False):
                    self._drop_hint_shown = True
                    toast(self, tr("Drop to add the project folder to Hub"), "info", timeout_ms=1800)
        else:
            event.ignore()

    def dropEvent(self, event):
        folders = self._dropped_folders(event)
        self._drop_hint_shown = False
        if not folders:
            return
        event.acceptProposedAction()
        self.sidebar.select_page(0)
        for folder in folders:
            self.viewmodel.add_project_folder(self.controls, folder)

    def _afterburner(self):
        from view.forge import toast
        self.sidebar.set_transfer_active(True)
        QTimer.singleShot(2600, self._update_queue_badge)
        toast(self, tr("Afterburner engaged. Nothing got faster, but it looked great."), "info", timeout_ms=2600)

    def _install_engine_for_project(self, version: str):
        self.install_tabs.setCurrentWidget(self.installs_view)
        self.sidebar.select_page(1)
        self.installs_view.open_install_dialog(preselect=version)

    # ── Updates ──────────────────────────────────────────────────────

    def _on_update_available(self, result):
        version = result.update.target_version if result is not None and result.update is not None else None
        self.sidebar.show_update(version)
        if version:
            from view.forge import toast
            toast(self, tr("Infernux Hub {version} is available.", version=version), "info",
                  action=(tr("Review"), self._show_update), timeout_ms=8000)

    def _show_update(self):
        if self.update_controller.prompt_update():
            self.sidebar.show_update(None)

    def _periodic_update_check(self):
        if self.db.get_setting("automatic_update_checks", "enabled") == "enabled":
            self.update_controller.check(silent=True)

    def _refresh_telemetry(self):
        """Sidebar readouts; local state only, never a network request."""
        if not hasattr(self, "sidebar"):
            return
        try:
            from version_manager import latest_releases
            engines = len(latest_releases(self.version_manager.installed_versions()))
        except Exception:
            engines = 0
        self.sidebar.set_telemetry(
            "engines", tr("ENGINES") + f"  {engines:02d}", "ok" if engines else "warn")
        default_version = self.runtime_manager.default_version
        has_runtime = self.runtime_manager.has_runtime(default_version)
        self.sidebar.set_telemetry(
            "runtime", tr("RUNTIME") + f"  PY {default_version}", "ok" if has_runtime else "error")
        from view.forge import fmt_age
        stamp = self.version_manager.catalog_timestamp()
        self.sidebar.set_telemetry(
            "network", tr("CATALOG") + "  " + fmt_age(stamp).upper(), "ok" if stamp else "idle")

    def _launch_project_from_card(self, project_id: str):
        self.project_list.select_project(project_id)
        if self.project_list.get_selected_project_id() == project_id:
            self.viewmodel.launch_project(self.controls)

    def _on_page_changed(self, index: int):
        self.pages.setCurrentIndex(index)
        effect = self.pages.graphicsEffect()
        if effect is None:
            effect = QGraphicsOpacityEffect(self.pages)
            self.pages.setGraphicsEffect(effect)
        effect.setOpacity(0.0)
        self._page_transition = QPropertyAnimation(effect, b"opacity", self)
        self._page_transition.setDuration(180)
        self._page_transition.setStartValue(0.0)
        self._page_transition.setEndValue(1.0)
        self._page_transition.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._page_transition.start()
        # Refresh installs when switching to that page
        if index == 1:
            self._on_install_tab_changed(self.install_tabs.currentIndex())
        elif index == 2:
            self.settings_view.refresh()

    def _on_install_tab_changed(self, index):
        self.install_tabs.widget(index).refresh()

    def _show_runtime_installs(self):
        self.install_tabs.setCurrentWidget(self.python_view)
        self.sidebar.select_page(1)

    def _install_required_runtime(self, version):
        self._show_runtime_installs()
        self.python_view.install(version)

    def _position_install_panel(self):
        panel = self.install_panel
        panel.setFixedWidth(max(240, min(320, self.pages.width() - 32)))
        panel.move(self.centralWidget().width() - panel.width() - 16,
                   max(16, self.centralWidget().height() - panel.height() - 16))
        panel.raise_()
        panel.position_details()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "install_panel"):
            self._position_install_panel()
        if hasattr(self, "toast_host"):
            self.toast_host.relayout()

    def _restore_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _show_installs(self):
        self._restore_window()
        self.sidebar.select_page(1)
        self.install_panel.refresh()

    def _on_tray_activated(self, reason):
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self._restore_window()

    def closeEvent(self, event):
        if self.tray.isVisible():
            event.ignore()
            self.hide()
        elif self.install_queue.busy:
            event.ignore()
            self.showMinimized()
        else:
            super().closeEvent(event)

    def request_quit(self):
        if self.install_queue.busy:
            if dialogs.question(
                self, tr("Installation in progress"),
                tr("Exit after the installation queue finishes? Installations will continue in the background."),
                dialogs.Yes | dialogs.No, dialogs.No,
                labels={dialogs.Yes: tr("Exit when finished"), dialogs.No: tr("Keep Hub open")},
            ) != dialogs.Yes:
                return
            self._exit_when_idle = True
            self.close()
        else:
            self.app.quit()

    def _on_queue_idle(self):
        if self._exit_when_idle:
            self.app.quit()

    def _remove_project_from_card(self, project_id: str):
        self.viewmodel.remove_project(self.controls, project_id)

    def _on_language_changed(self, _mode: str):
        # Leave the Settings signal stack before retiring its widgets.
        self._schedule_language_refresh()

    @Slot()
    def _schedule_language_refresh(self):
        self._language_refresh_timer.start(0)

    @Slot()
    def _clear_language_waits(self):
        for connection in self._language_wait_connections:
            QObject.disconnect(connection)
        self._language_wait_connections.clear()

    @Slot()
    def _refresh_language_pages(self):
        self._clear_language_waits()
        central = self.centralWidget()
        waiting = False
        # Query dialogs own their QThreads. Keep the pages alive until those
        # threads finish; completion signals schedule one more event turn.
        for thread in central.findChildren(QThread):
            connection = thread.finished.connect(self._schedule_language_refresh)
            if thread.isRunning():
                waiting = True
                self._language_wait_connections.append(connection)
            else:
                thread.wait()
                QObject.disconnect(connection)
        creation = self.viewmodel._creation_dialog
        if creation is not None:
            self._language_wait_connections.append(
                creation.finished.connect(self._schedule_language_refresh)
            )
            waiting = True
        if waiting:
            return

        selected = self.project_list.get_selected_project_id()
        search = self.project_list.search_edit.text()
        page, tab = self.pages.currentIndex(), self.install_tabs.currentIndex()
        transition = getattr(self, "_page_transition", None)
        if transition is not None:
            transition.stop()
            transition.deleteLater()
        self.discussion_view.shutdown()
        old_central = self.takeCentralWidget()
        old_central.hide()
        self._build_pages()
        self._build_tray_menu()
        self.project_list.search_edit.setText(search)
        self.project_list.select_project(selected)
        self.install_tabs.setCurrentIndex(tab)
        self.sidebar.select_page(page)
        old_central.deleteLater()

    def run(self):
        self.show()
        if is_frozen():
            QTimer.singleShot(0, self._bootstrap_hub)
        if self.db.get_setting("automatic_update_checks", "enabled") == "enabled":
            # Warm the engine catalog so Installs opens with fresh data.
            QTimer.singleShot(1500, self._prefetch_catalog)
        if self._own_app:
            sys.exit(self.app.exec())

    def _prefetch_catalog(self):
        from view.installs_view import _FetchWorker
        if self._catalog_prefetch is not None and self._catalog_prefetch.isRunning():
            return
        worker = _FetchWorker(self.version_manager, self)
        worker.loaded.connect(lambda _versions: self._refresh_telemetry())
        worker.finished.connect(lambda: setattr(self, "_catalog_prefetch", None))
        worker.finished.connect(worker.deleteLater)
        self.app.aboutToQuit.connect(worker.wait)
        self._catalog_prefetch = worker
        worker.start()

    def _bootstrap_hub(self):
        if self.db.get_setting("automatic_update_checks", "enabled") == "enabled":
            self._startup_update_pending = True
            self.update_controller.check(silent=True)
            return
        self._bootstrap_python_runtime()

    def _on_startup_update_check_finished(self):
        if not self._startup_update_pending:
            return
        self._startup_update_pending = False
        self._bootstrap_python_runtime()

    def _bootstrap_python_runtime(self):
        # A fresh installer provisions the default runtime before launching the
        # Hub. An in-place upgrade from an older Hub deliberately does not
        # mutate Python behind the user's back, so make the missing requirement
        # explicit and take the user directly to the runtime controls.
        default_version = self.runtime_manager.default_version
        if not self.runtime_manager.has_runtime(default_version):
            self._show_runtime_installs()
        QTimer.singleShot(0, self._finish_startup)

    def _seed_bundled_engines(self):
        """Install the engine wheel shipped with the installer, off the UI thread."""
        import bundled_engine
        try:
            source = bundled_engine.bundled_engines_dir()
            if not source.is_dir():
                return
            seeded = bundled_engine._read_seeded(bundled_engine._default_state_path())
            if all(wheel.name in seeded for wheel in source.glob("infernux-*.whl")):
                return
        except OSError:
            return
        if not self.runtime_manager.has_runtime(self.runtime_manager.default_version):
            return
        manager = self.version_manager

        def seed(report):
            report(tr("Unpacking the bundled engine"), 0, 0)
            return bundled_engine.seed_bundled_engines(manager)

        self.install_queue.submit("engine:bundled", tr("Bundled engine"), seed)

    def _finish_startup(self):
        self._seed_bundled_engines()
        self.installs_view.refresh()
        if self.db.get_setting("automatic_update_checks", "enabled") == "enabled":
            self.notification_controller.show_pending()

    def _on_close(self):
        self._save_geometry()
        self._language_refresh_timer.stop()
        self._clear_language_waits()
        self.viewmodel._wait_for_creation()
        self.db.close()


def install_hub_fonts(app) -> None:
    """Register the Hub typeface and make it the application default."""
    for font_path in FONT_PATHS:
        QFontDatabase.addApplicationFont(font_path)
    app.setFont(ui_font(13))


def _standalone_application():
    """QApplication for prompts shown without the Hub window (uninstall)."""
    app = QApplication.instance() or QApplication(sys.argv)
    install_hub_fonts(app)
    app.is_dark_theme = True
    app.setStyleSheet(StyleManager.get_stylesheet(True))
    return app


def _schedule_windows_application_removal(install_dir: str) -> None:
    """Run outside the Hub so its executable is no longer locked during removal."""
    import ctypes
    from ctypes import wintypes
    import subprocess

    powershell = str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    helper = (
        Path(get_app_dir()) / "InfernuxHubData/uninstaller/hub_uninstall.ps1"
        if is_frozen() else Path(__file__).with_name("hub_uninstall.ps1")
    )
    if not helper.is_file():
        raise FileNotFoundError(f"Hub uninstall helper is missing: {helper}")
    shell_execute = ctypes.windll.shell32.ShellExecuteW
    shell_execute.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR,
                              wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_int]
    shell_execute.restype = wintypes.HINSTANCE
    result = shell_execute(
        None, "runas", powershell,
        subprocess.list2cmdline([
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(helper),
            "-InstallDir", install_dir, "-ParentPid", str(os.getpid()),
        ]),
        install_dir, 0,
    )
    if int(result or 0) <= 32:
        raise OSError(int(result or 0), "Could not start the Hub uninstaller")


def _handle_uninstall() -> int:
    """Remove registry entries, Start Menu shortcut, and optionally the install directory."""
    if sys.platform == "darwin":
        return _handle_uninstall_macos()
    if sys.platform.startswith("linux"):
        return _handle_uninstall_linux()
    if sys.platform != "win32":
        return 1
    import winreg

    # Read install location from registry before removing the key.
    install_dir = ""
    reg_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\InfernuxHub"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_key) as key:
            install_dir, _ = winreg.QueryValueEx(key, "InstallLocation")
    except OSError as _exc:
        logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
        pass

    # Remove registry entry
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, reg_key)
    except OSError as _exc:
        logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
        pass

    # Remove Start Menu shortcut
    try:
        import ctypes.wintypes
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(None, 0x0002, None, 0, buf)
        if buf.value:
            import shutil as _shutil
            _shutil.rmtree(os.path.join(buf.value, "Infernux Hub"), ignore_errors=True)
    except Exception as _exc:
        logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
        pass

    # Ask user if they want to remove install files
    app = _standalone_application()
    answer = dialogs.question(
        None,
        tr("Uninstall Infernux Hub"),
        tr("Remove Hub application files after this window closes?\n{path}\n\nProjects and Shared resources (plugins, SDKs, runtimes and engines) are preserved.", path=install_dir),
    )
    if answer == dialogs.Yes and install_dir and os.path.isdir(install_dir):
        if can_remove_install_dir(install_dir):
            try:
                _schedule_windows_application_removal(install_dir)
            except OSError as exc:
                dialogs.critical(None, tr("Uninstall Failed"), str(exc))
                return 1
            return 0
        else:
            dialogs.warning(
                None,
                tr("Install Folder Preserved"),
                tr("The installation folder was not deleted because it is not marked as a safe Infernux Hub install directory.\n\n"
                "Your projects and downloaded engine versions are preserved. Remove application files manually only if "
                "you are sure this folder does not contain user data."),
            )

    dialogs.information(None, tr("Uninstall Complete"), tr("Infernux Hub has been uninstalled."), level="ok")
    return 0


def _handle_uninstall_macos() -> int:
    """Remove Infernux Hub from macOS."""
    import shutil as _shutil

    app = _standalone_application()

    # Typical macOS install / config locations
    config_dir = os.path.expanduser("~/.config/Infernux")
    app_link = os.path.expanduser("~/Applications/Infernux Hub")
    dirs_to_remove = [d for d in (config_dir, app_link) if os.path.exists(d)]

    if dirs_to_remove:
        answer = dialogs.question(
            None,
            "Uninstall Infernux Hub",
            "Do you want to remove Infernux Hub application configuration?\n\n"
            + "\n".join(dirs_to_remove),
        )
        if answer == dialogs.Yes:
            for d in dirs_to_remove:
                _shutil.rmtree(d, ignore_errors=True)

    dialogs.information(None, "Uninstall Complete", "Infernux Hub has been uninstalled.", level="ok")
    return 0


def _handle_uninstall_linux() -> int:
    """Remove the Linux application while preserving Hub user data."""
    app = _standalone_application()

    desktop_entry = os.path.expanduser("~/.local/share/applications/infernux-hub.desktop")
    install_dir = get_app_dir()
    targets = [p for p in (desktop_entry, install_dir) if os.path.exists(p)]

    if targets:
        answer = dialogs.question(
            None,
            "Uninstall Infernux Hub",
            "Do you want to remove the Infernux Hub application?\n\n"
            + "\n".join(targets)
            + "\n\nProjects, downloaded engines, Python runtimes, and the shared "
            "plugin library are preserved.",
        )
        if answer == dialogs.Yes:
            for p in targets:
                if os.path.isdir(p):
                    if p == install_dir and not can_remove_install_dir(p):
                        raise RuntimeError(
                            "The Hub application directory is not a recognized install: "
                            f"{p}"
                        )
                    remove_application(p)
                else:
                    os.remove(p)

    dialogs.information(None, "Uninstall Complete", "Infernux Hub has been uninstalled.", level="ok")
    return 0


if __name__ == "__main__":
    from hub_logging import configure_logging
    configure_logging()
    from hub_network import configure_system_certificates
    configure_system_certificates()
    if "--uninstall" in sys.argv:
        raise SystemExit(_handle_uninstall())
    launcher = GameEngineLauncher(HubLaunchContext.current())
    launcher.run()
