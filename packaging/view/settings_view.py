"""Early Hub settings page."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from about_content import ABOUT_DESCRIPTION, ABOUT_TITLE
from hub_updater import current_hub_version
from i18n import current_language, detect_system_locale, tr
from plugin_library import inspect_plugin_library, prune_unreferenced_packages
from hub_utils import get_hub_shared_data_dir
from hub_logging import hub_log_path
from shared_storage_migration import inspect_legacy_storage
from view.storage_migration_dialog import StorageMigrationDialog
from view.sidebar_view import ToggleSwitch, apply_theme
from view.hover_widgets import AnimatedSurfaceFrame
from view.forge import ForgeComboBox, PageHeader, mono_label, toast
from view import dialogs


class _SettingsGroup(AnimatedSurfaceFrame):
    """A titled plate of setting rows separated by hairlines."""

    def __init__(self, code: str, title: str, parent=None):
        super().__init__("settingsCard", parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 14, 18, 8)
        self._layout.setSpacing(0)
        heading = mono_label(f"{code}  /  {title}", "pageKicker", spacing=1.6)
        self._layout.addWidget(heading)
        self._layout.addSpacing(6)
        self._rows = 0

    def add_row(self, title: str, description: QLabel | str, *controls: QWidget) -> QLabel:
        if self._rows:
            line = QFrame()
            line.setObjectName("settingsRule")
            line.setFixedHeight(1)
            self._layout.addWidget(line)
        row = QHBoxLayout()
        row.setContentsMargins(0, 12, 0, 12)
        row.setSpacing(14)
        text = QVBoxLayout()
        text.setSpacing(4)
        label = QLabel(title)
        label.setObjectName("settingsLabel")
        label.setWordWrap(True)
        text.addWidget(label)
        if isinstance(description, str):
            description_label = QLabel(description)
            description_label.setObjectName("settingsDescription")
            description_label.setWordWrap(True)
        else:
            description_label = description
        description_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text.addWidget(description_label)
        row.addLayout(text, 1)
        for control in controls:
            row.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
        self._layout.addLayout(row)
        self._rows += 1
        return description_label


class SettingsView(QWidget):
    update_check_requested = Signal()
    language_changed = Signal(str)

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self._db = db

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(16)
        root_layout.addWidget(PageHeader(
            "04", tr("SETTINGS"), tr("Settings"),
            tr("Hub preferences, updates and project-independent information."),
        ))
        scroll = QScrollArea(self)
        scroll.setObjectName("settingsScrollArea")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.viewport().setObjectName("settingsViewport")
        content = QWidget()
        content.setObjectName("settingsContent")
        scroll.setWidget(content)
        root_layout.addWidget(scroll, 1)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(12)

        general = _SettingsGroup("A", tr("GENERAL"))
        self.language_combo = ForgeComboBox()
        self.language_combo.setMinimumWidth(150)
        self.language_combo.addItem(tr("System"), "system", meta=detect_system_locale().upper())
        self.language_combo.addItem("中文", "zh", meta="ZH-CN")
        self.language_combo.addItem("English", "en", meta="EN")
        saved = self._db.get_setting("language", "system") if self._db else "system"
        index = self.language_combo.findData(saved)
        self.language_combo.setCurrentIndex(max(index, 0))
        self.language_combo.currentIndexChanged.connect(self._save_language)
        general.add_row(
            tr("Language"),
            f"{tr('Current language')}: {'中文' if current_language() == 'zh' else 'English'} "
            f"({detect_system_locale()}) · {tr('Language changes apply immediately.')}",
            self.language_combo,
        )
        self.theme_toggle = ToggleSwitch()
        self.theme_toggle.setChecked(bool(getattr(QApplication.instance(), "is_dark_theme", True)))
        self.theme_toggle.stateChanged.connect(self._toggle_theme)
        general.add_row(
            tr("Appearance"),
            tr("Dark control room or light paper. Both keep the Infernux red."),
            self.theme_toggle,
        )

        self._runtime_ca_bundle = (
            self._db.get_setting("python_runtime_ca_bundle", "").strip() if self._db else ""
        )
        certificate_controls = QWidget()
        certificate_row = QHBoxLayout(certificate_controls)
        certificate_row.setContentsMargins(0, 0, 0, 0)
        self.runtime_ca_bundle_edit = QLineEdit(self._runtime_ca_bundle)
        self.runtime_ca_bundle_edit.setPlaceholderText(tr("No custom certificate"))
        self.runtime_ca_bundle_edit.setClearButtonEnabled(True)
        self.runtime_ca_bundle_edit.setAccessibleName(tr("Python runtime download CA certificate"))
        self.runtime_ca_bundle_edit.editingFinished.connect(self._save_runtime_ca_bundle)
        certificate_row.addWidget(self.runtime_ca_bundle_edit, 1)
        self.runtime_ca_browse_button = QPushButton(tr("Browse"))
        self.runtime_ca_browse_button.setObjectName("normalBtn")
        self.runtime_ca_browse_button.setFixedHeight(34)
        self.runtime_ca_browse_button.clicked.connect(self._choose_runtime_ca_bundle)
        certificate_row.addWidget(self.runtime_ca_browse_button)
        self.runtime_ca_clear_button = QPushButton(tr("Clear"))
        self.runtime_ca_clear_button.setObjectName("normalBtn")
        self.runtime_ca_clear_button.setFixedHeight(34)
        self.runtime_ca_clear_button.clicked.connect(self._clear_runtime_ca_bundle)
        certificate_row.addWidget(self.runtime_ca_clear_button)
        general.add_row(
            tr("Python runtime downloads"),
            tr("Optional PEM CA certificate for Python runtime downloads."),
            certificate_controls,
        )
        layout.addWidget(general)

        updates = _SettingsGroup("B", tr("UPDATES"))
        self.automatic_update_toggle = ToggleSwitch()
        self.automatic_update_toggle.setChecked(
            bool(self._db)
            and self._db.get_setting("automatic_update_checks", "enabled")
            == "enabled"
        )
        self.automatic_update_toggle.stateChanged.connect(
            self._save_automatic_update_checks
        )
        updates.add_row(
            tr("Automatic update and release-notice checks"),
            tr(
                "When enabled, Hub contacts infernux-engine.com at startup for "
                "updates and release notices. No project content is sent. "
                "Installing updates requires confirmation."
            ),
            self.automatic_update_toggle,
        )
        update_button = QPushButton(tr("Check for Updates"))
        update_button.setObjectName("normalBtn")
        update_button.setFixedHeight(34)
        update_button.setMinimumWidth(130)
        update_button.clicked.connect(self.update_check_requested)
        updates.add_row(
            tr("Hub Update"),
            tr("Hub version: {version}", version=current_hub_version()) + " · "
            + tr("Check the Infernux release catalog for a Hub update."),
            update_button,
        )
        layout.addWidget(updates)

        storage = _SettingsGroup("C", tr("STORAGE"))
        self.storage_description = QLabel()
        self.storage_description.setObjectName("settingsDescription")
        self.storage_description.setWordWrap(True)
        self.storage_description.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.clean_plugins_button = QPushButton(tr("Clean Unused Packages"))
        self.clean_plugins_button.setObjectName("normalBtn")
        self.clean_plugins_button.setFixedHeight(34)
        self.clean_plugins_button.clicked.connect(self._clean_plugin_library)
        storage.add_row(tr("Plugin Library"), self.storage_description, self.clean_plugins_button)
        shared_path = QLabel(tr("Shared resources: {path}", path=get_hub_shared_data_dir()))
        shared_path.setObjectName("settingsPath")
        shared_path.setWordWrap(True)
        shared_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.migrate_storage_button = QPushButton(tr("Migrate Legacy Resources"))
        self.migrate_storage_button.setObjectName("normalBtn")
        self.migrate_storage_button.setFixedHeight(34)
        self.migrate_storage_button.clicked.connect(self._migrate_legacy_storage)
        storage.add_row(tr("Shared resources"), shared_path, self.migrate_storage_button)
        logs = QPushButton(tr("Open Hub Logs"))
        logs.setObjectName("normalBtn")
        logs.setFixedHeight(34)
        logs.clicked.connect(self._open_logs)
        log_path = QLabel(str(hub_log_path().parent))
        log_path.setObjectName("settingsPath")
        log_path.setWordWrap(True)
        log_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        storage.add_row(tr("Diagnostics"), log_path, logs)
        layout.addWidget(storage)
        self._refresh_plugin_library()

        about = _SettingsGroup("D", tr("ABOUT"))
        about.add_row(tr(ABOUT_TITLE), tr(ABOUT_DESCRIPTION))
        layout.addWidget(about)
        layout.addStretch()

    def _open_logs(self):
        directory = hub_log_path().parent
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _save_language(self):
        if not self._db:
            return
        mode = self.language_combo.currentData()
        if mode == self._db.get_setting("language", "system"):
            return
        self._db.set_setting("language", mode)
        from i18n import configure_language
        configure_language(mode)
        self.language_changed.emit(mode)

    def _save_automatic_update_checks(self, state: int):
        if self._db:
            self._db.set_setting(
                "automatic_update_checks", "enabled" if state else "disabled"
            )

    def _choose_runtime_ca_bundle(self):
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            tr("Select a CA certificate"),
            str(Path(self._runtime_ca_bundle).parent)
            if self._runtime_ca_bundle
            else "",
            tr("Certificate files (*.pem);;All files (*)"),
        )
        if path:
            self.runtime_ca_bundle_edit.setText(path)
            self._save_runtime_ca_bundle()

    def _clear_runtime_ca_bundle(self):
        self.runtime_ca_bundle_edit.clear()
        self._save_runtime_ca_bundle()

    def _save_runtime_ca_bundle(self):
        raw_path = self.runtime_ca_bundle_edit.text().strip()
        if raw_path:
            try:
                path = str(Path(raw_path).expanduser().resolve())
            except (OSError, RuntimeError, ValueError) as exc:
                QMessageBox.critical(
                    self,
                    tr("Settings"),
                    f"{tr('Python runtime download CA certificate')}: {exc}",
                )
                return
        else:
            path = ""

        if path == self._runtime_ca_bundle:
            return
        self._runtime_ca_bundle = path
        if self._db:
            self._db.set_setting("python_runtime_ca_bundle", path)

    def refresh(self):
        """Refresh state owned outside the Hub process."""

        self._refresh_plugin_library()

    def _toggle_theme(self, state: int):
        if self._db:
            self._db.set_setting("theme", "dark" if state else "light")
        apply_theme(self.window(), bool(state))

    def _project_roots(self) -> tuple[str, ...]:
        if not self._db:
            return ()
        return tuple(record.path for record in self._db.all_projects())

    @staticmethod
    def _format_bytes(value: int) -> str:
        amount = float(max(0, int(value)))
        units = ("B", "KiB", "MiB", "GiB", "TiB")
        unit = units[0]
        for unit in units:
            if amount < 1024.0 or unit == units[-1]:
                break
            amount /= 1024.0
        return f"{int(amount)} {unit}" if unit == "B" else f"{amount:.1f} {unit}"

    def _refresh_plugin_library(self):
        try:
            stats = inspect_plugin_library(self._project_roots())
        except (OSError, RuntimeError, ValueError) as exc:
            self.storage_description.setText(
                tr("Plugin library cleanup is unavailable: {message}", message=str(exc))
            )
            self.clean_plugins_button.setEnabled(False)
            return
        self.storage_description.setText(
            tr(
                "{count} packages · {size} · {path}",
                count=stats.package_count,
                size=self._format_bytes(stats.total_bytes),
                path=str(stats.root),
            )
        )
        self.clean_plugins_button.setEnabled(bool(stats.removable))
        self.clean_plugins_button.setToolTip(
            tr(
                "{count} unused packages can release {size}.",
                count=len(stats.removable),
                size=self._format_bytes(stats.removable_bytes),
            )
            if stats.removable
            else tr("Every downloaded package is still referenced by a Hub project.")
        )

    def _clean_plugin_library(self):
        try:
            before = inspect_plugin_library(self._project_roots())
        except (OSError, RuntimeError, ValueError) as exc:
            dialogs.critical(self, tr("Plugin Library"), str(exc))
            self._refresh_plugin_library()
            return
        if not before.removable:
            self._refresh_plugin_library()
            return
        answer = dialogs.question(
            self,
            tr("Clean Unused Packages"),
            tr(
                "Delete {count} unreferenced plugin packages and release {size}?",
                count=len(before.removable),
                size=self._format_bytes(before.removable_bytes),
            ),
            dialogs.Yes | dialogs.No, dialogs.No,
            labels={dialogs.Yes: tr("Delete packages"), dialogs.No: tr("Cancel")},
            destructive=True, kicker=tr("PLUGIN LIBRARY"),
            detail="\n".join(str(item) for item in before.removable),
        )
        if answer != dialogs.Yes:
            return
        try:
            prune_unreferenced_packages(self._project_roots())
        except (OSError, RuntimeError, ValueError) as exc:
            dialogs.critical(self, tr("Plugin Library"), str(exc))
        else:
            toast(self, tr("Released {size}", size=self._format_bytes(before.removable_bytes)))
        self._refresh_plugin_library()

    def _migrate_legacy_storage(self):
        try:
            plan = inspect_legacy_storage()
        except (OSError, RuntimeError, ValueError) as exc:
            dialogs.critical(self, tr("Migrate Legacy Resources"), str(exc))
            return
        preview = dialogs.HubDialog(
            self,
            level="question" if plan.items else "info",
            title=tr("Migrate Legacy Resources"),
            text=tr(
                "Move {count} complete resources from {source} to {destination}?\n"
                "Close all Editors, builds and downloads first. Existing targets ({conflicts}) "
                "will be skipped and retained at the old location. Projects, settings and "
                "unfinished downloads are not moved. See details for the exact list.",
                count=len(plan.items), source=str(plan.source), destination=str(plan.destination),
                conflicts=len(plan.conflicts),
            ),
            kicker=tr("STORAGE"),
            detail=(
                tr("Move:") + "\n" + "\n".join(path.as_posix() for path in plan.items)
                + "\n\n" + tr("Keep at old location (target exists):") + "\n"
                + "\n".join(path.as_posix() for path in plan.conflicts)
            ),
        )
        if plan.items:
            cancel = preview.addButton(tr("Cancel"), "reject")
            move = preview.addButton(tr("Move resources"), "accept")
            preview.setDefaultButton(cancel)
        else:
            move = None
            preview.setDefaultButton(preview.addButton(tr("OK"), "reject"))
        preview.exec()
        if move is None or preview.clickedButton() is not move:
            return
        dialog = StorageMigrationDialog(plan, self._project_roots(), self)
        dialog.exec()
        if dialog.worker.error:
            dialogs.critical(self, tr("Migrate Legacy Resources"), dialog.worker.error)
        else:
            dialogs.information(self, tr("Migrate Legacy Resources"), tr(
                "Moved {count} resources. {conflicts} existing targets were skipped; "
                "their old copies have not been deleted.",
                count=len(dialog.worker.result), conflicts=len(plan.conflicts),
            ), level="ok")
        self._refresh_plugin_library()


__all__ = ["SettingsView"]
