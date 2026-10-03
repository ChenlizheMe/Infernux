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
from hub_network import create_download_ssl_context
from shared_storage_migration import inspect_legacy_storage
from view.storage_migration_dialog import StorageMigrationDialog
from view.sidebar_view import ToggleSwitch, apply_theme
from view.hover_widgets import AnimatedSurfaceFrame


class SettingsView(QWidget):
    update_check_requested = Signal()
    language_changed = Signal(str)
    runtime_ca_bundle_changed = Signal(str)

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self._db = db

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setObjectName("settingsScrollArea")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.viewport().setObjectName("settingsViewport")
        content = QWidget()
        content.setObjectName("settingsContent")
        scroll.setWidget(content)
        root_layout.addWidget(scroll)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)

        title = QLabel(tr("Settings"))
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        subtitle = QLabel(tr("Hub preferences, updates and project-independent information."))
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        card = AnimatedSurfaceFrame("settingsCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(12)

        row = QHBoxLayout()
        label_block = QVBoxLayout()
        language_label = QLabel(tr("Language"))
        language_label.setObjectName("settingsLabel")
        label_block.addWidget(language_label)
        detected = QLabel(
            f"{tr('Current language')}: {'中文' if current_language() == 'zh' else 'English'} "
            f"({detect_system_locale()})"
        )
        detected.setObjectName("settingsDescription")
        label_block.addWidget(detected)
        row.addLayout(label_block, 1)

        self.language_combo = QComboBox()
        self.language_combo.addItem(tr("System"), "system")
        self.language_combo.addItem(tr("Chinese"), "zh")
        self.language_combo.addItem(tr("English"), "en")
        saved = self._db.get_setting("language", "system") if self._db else "system"
        index = self.language_combo.findData(saved)
        self.language_combo.setCurrentIndex(max(index, 0))
        self.language_combo.currentIndexChanged.connect(self._save_language)
        row.addWidget(self.language_combo)
        card_layout.addLayout(row)

        hint = QLabel(tr("Language changes apply immediately."))
        hint.setObjectName("settingsDescription")
        card_layout.addWidget(hint)
        layout.addWidget(card)

        appearance_card = AnimatedSurfaceFrame("settingsCard")
        appearance_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        appearance_layout = QHBoxLayout(appearance_card)
        appearance_layout.setContentsMargins(20, 16, 20, 16)
        appearance_text = QVBoxLayout()
        appearance_label = QLabel(tr("Appearance"))
        appearance_label.setObjectName("settingsLabel")
        appearance_text.addWidget(appearance_label)
        appearance_hint = QLabel(tr("Switch between the neutral dark and light Hub themes."))
        appearance_hint.setObjectName("settingsDescription")
        appearance_hint.setWordWrap(True)
        appearance_text.addWidget(appearance_hint)
        appearance_layout.addLayout(appearance_text, 1)
        self.theme_toggle = ToggleSwitch()
        self.theme_toggle.setChecked(bool(getattr(QApplication.instance(), "is_dark_theme", True)))
        self.theme_toggle.stateChanged.connect(self._toggle_theme)
        appearance_layout.addWidget(self.theme_toggle)
        layout.addWidget(appearance_card)

        network_card = AnimatedSurfaceFrame("settingsCard")
        network_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        network_layout = QVBoxLayout(network_card)
        network_layout.setContentsMargins(20, 18, 20, 18)
        network_layout.setSpacing(10)
        network_title = QLabel(tr("Python runtime downloads"))
        network_title.setObjectName("settingsLabel")
        network_layout.addWidget(network_title)
        network_hint = QLabel(
            tr(
                "Optional CA certificate for Python runtime downloads. "
                "It is added to the default trusted certificates."
            )
        )
        network_hint.setObjectName("settingsDescription")
        network_hint.setWordWrap(True)
        network_layout.addWidget(network_hint)

        self._runtime_ca_bundle = (
            self._db.get_setting("python_runtime_ca_bundle", "").strip()
            if self._db
            else ""
        )
        certificate_row = QHBoxLayout()
        self.runtime_ca_bundle_edit = QLineEdit(self._runtime_ca_bundle)
        self.runtime_ca_bundle_edit.setPlaceholderText(tr("No custom certificate"))
        self.runtime_ca_bundle_edit.setClearButtonEnabled(True)
        self.runtime_ca_bundle_edit.setAccessibleName(
            tr("Python runtime download CA certificate")
        )
        self.runtime_ca_bundle_edit.editingFinished.connect(
            self._save_runtime_ca_bundle
        )
        certificate_row.addWidget(self.runtime_ca_bundle_edit, 1)
        self.runtime_ca_browse_button = QPushButton(tr("Browse"))
        self.runtime_ca_browse_button.setObjectName("normalBtn")
        self.runtime_ca_browse_button.setFixedHeight(34)
        self.runtime_ca_browse_button.clicked.connect(
            self._choose_runtime_ca_bundle
        )
        certificate_row.addWidget(self.runtime_ca_browse_button)
        self.runtime_ca_clear_button = QPushButton(tr("Clear"))
        self.runtime_ca_clear_button.setObjectName("normalBtn")
        self.runtime_ca_clear_button.setFixedHeight(34)
        self.runtime_ca_clear_button.clicked.connect(self._clear_runtime_ca_bundle)
        certificate_row.addWidget(self.runtime_ca_clear_button)
        network_layout.addLayout(certificate_row)
        self.runtime_ca_bundle_status = QLabel()
        self.runtime_ca_bundle_status.setObjectName("settingsDescription")
        self.runtime_ca_bundle_status.setWordWrap(True)
        network_layout.addWidget(self.runtime_ca_bundle_status)
        self._refresh_runtime_ca_bundle_status()
        layout.addWidget(network_card)

        storage_card = AnimatedSurfaceFrame("settingsCard")
        storage_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        storage_layout = QHBoxLayout(storage_card)
        storage_layout.setContentsMargins(20, 18, 20, 18)
        storage_text = QVBoxLayout()
        storage_label = QLabel(tr("Plugin Library"))
        storage_label.setObjectName("settingsLabel")
        storage_text.addWidget(storage_label)
        self.storage_description = QLabel()
        self.storage_description.setObjectName("settingsDescription")
        self.storage_description.setWordWrap(True)
        self.storage_description.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.storage_description.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        storage_text.addWidget(self.storage_description)
        storage_layout.addLayout(storage_text, 1)
        self.clean_plugins_button = QPushButton(tr("Clean Unused Packages"))
        self.clean_plugins_button.setObjectName("normalBtn")
        self.clean_plugins_button.setFixedHeight(34)
        self.clean_plugins_button.clicked.connect(self._clean_plugin_library)
        storage_layout.addWidget(self.clean_plugins_button)
        layout.addWidget(storage_card)
        self._refresh_plugin_library()

        migration_card = AnimatedSurfaceFrame("settingsCard")
        migration_layout = QVBoxLayout(migration_card)
        migration_layout.setContentsMargins(20, 18, 20, 18)
        shared_path = QLabel(tr("Shared resources: {path}", path=get_hub_shared_data_dir()))
        shared_path.setWordWrap(True)
        shared_path.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        shared_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        migration_layout.addWidget(shared_path)
        self.migrate_storage_button = QPushButton(tr("Migrate Legacy Resources"))
        self.migrate_storage_button.setObjectName("normalBtn")
        self.migrate_storage_button.setFixedHeight(34)
        self.migrate_storage_button.clicked.connect(self._migrate_legacy_storage)
        migration_layout.addWidget(self.migrate_storage_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(migration_card)

        logs = QPushButton(tr("Open Hub Logs"))
        logs.setObjectName("normalBtn")
        logs.setMinimumHeight(34)
        logs.clicked.connect(self._open_logs)
        layout.addWidget(logs, 0, Qt.AlignmentFlag.AlignLeft)

        update_card = AnimatedSurfaceFrame("settingsCard")
        update_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        update_layout = QHBoxLayout(update_card)
        update_layout.setContentsMargins(20, 18, 20, 18)
        update_text = QVBoxLayout()
        update_label = QLabel(tr("Hub Update"))
        update_label.setObjectName("settingsLabel")
        update_text.addWidget(update_label)
        update_description = QLabel(tr("Check the Infernux release catalog for a Hub update."))
        update_description.setObjectName("settingsDescription")
        update_description.setWordWrap(True)
        update_text.addWidget(update_description)
        automatic_label = QLabel(tr("Automatic update and release-notice checks"))
        automatic_label.setObjectName("settingsLabel")
        automatic_label.setWordWrap(True)
        update_text.addWidget(automatic_label)
        automatic_description = QLabel(
            tr(
                "When enabled, Hub contacts infernux-engine.com at startup for "
                "updates and release notices. No project content is sent. "
                "Installing updates requires confirmation."
            )
        )
        automatic_description.setObjectName("settingsDescription")
        automatic_description.setWordWrap(True)
        update_text.addWidget(automatic_description)
        update_layout.addLayout(update_text, 1)
        self.automatic_update_toggle = ToggleSwitch()
        self.automatic_update_toggle.setChecked(
            bool(self._db)
            and self._db.get_setting("automatic_update_checks", "enabled")
            == "enabled"
        )
        self.automatic_update_toggle.stateChanged.connect(
            self._save_automatic_update_checks
        )
        update_layout.addWidget(self.automatic_update_toggle)
        update_button = QPushButton(tr("Check for Updates"))
        update_button.setObjectName("normalBtn")
        update_button.setFixedHeight(34)
        update_button.setMinimumWidth(118)
        update_button.clicked.connect(self.update_check_requested)
        update_layout.addWidget(update_button)
        layout.addWidget(update_card)

        about_card = AnimatedSurfaceFrame("settingsCard")
        about_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        about_layout = QVBoxLayout(about_card)
        about_layout.setContentsMargins(20, 18, 20, 18)
        about_layout.setSpacing(6)
        about_title = QLabel(tr(ABOUT_TITLE))
        about_title.setObjectName("settingsLabel")
        about_layout.addWidget(about_title)
        about_text = QLabel(tr(ABOUT_DESCRIPTION))
        about_text.setObjectName("settingsDescription")
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)
        version = QLabel(tr("Hub version: {version}", version=current_hub_version()))
        version.setObjectName("settingsDescription")
        about_layout.addWidget(version)
        layout.addWidget(about_card)
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
            tr("Certificate files (*.pem *.crt *.cer);;All files (*)"),
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
            path = str(Path(raw_path).expanduser().resolve())
            try:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
                create_download_ssl_context(path)
            except (OSError, ValueError) as exc:
                self.runtime_ca_bundle_edit.setText(self._runtime_ca_bundle)
                self.runtime_ca_bundle_status.setText(
                    tr("The certificate could not be loaded: {message}", message=str(exc))
                )
                return
        else:
            path = ""

        if path == self._runtime_ca_bundle:
            self._refresh_runtime_ca_bundle_status()
            return
        self._runtime_ca_bundle = path
        if self._db:
            self._db.set_setting("python_runtime_ca_bundle", path)
        self.runtime_ca_bundle_changed.emit(path)
        self._refresh_runtime_ca_bundle_status()

    def _refresh_runtime_ca_bundle_status(self):
        if not self._runtime_ca_bundle:
            self.runtime_ca_bundle_status.clear()
            return
        try:
            if not Path(self._runtime_ca_bundle).is_file():
                raise FileNotFoundError(self._runtime_ca_bundle)
            create_download_ssl_context(self._runtime_ca_bundle)
        except (OSError, ValueError) as exc:
            self.runtime_ca_bundle_status.setText(
                tr("The certificate could not be loaded: {message}", message=str(exc))
            )
            return
        self.runtime_ca_bundle_status.setText(
            tr("This certificate is added to the default trust store for Python runtime downloads.")
        )

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
            QMessageBox.critical(self, tr("Plugin Library"), str(exc))
            self._refresh_plugin_library()
            return
        if not before.removable:
            self._refresh_plugin_library()
            return
        answer = QMessageBox.question(
            self,
            tr("Clean Unused Packages"),
            tr(
                "Delete {count} unreferenced plugin packages and release {size}?",
                count=len(before.removable),
                size=self._format_bytes(before.removable_bytes),
            ),
        )
        if answer != QMessageBox.Yes:
            return
        try:
            prune_unreferenced_packages(self._project_roots())
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(self, tr("Plugin Library"), str(exc))
        self._refresh_plugin_library()

    def _migrate_legacy_storage(self):
        try:
            plan = inspect_legacy_storage()
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.critical(self, tr("Migrate Legacy Resources"), str(exc))
            return
        preview = QMessageBox(self)
        preview.setWindowTitle(tr("Migrate Legacy Resources"))
        preview.setText(tr(
            "Move {count} complete resources from {source} to {destination}?\n"
            "Close all Editors, builds and downloads first. Existing targets ({conflicts}) "
            "will be skipped and retained at the old location. Projects, settings and "
            "unfinished downloads are not moved. See details for the exact list.",
            count=len(plan.items), source=str(plan.source), destination=str(plan.destination),
            conflicts=len(plan.conflicts),
        ))
        preview.setDetailedText(
            tr("Move:") + "\n" + "\n".join(path.as_posix() for path in plan.items)
            + "\n\n" + tr("Keep at old location (target exists):") + "\n"
            + "\n".join(path.as_posix() for path in plan.conflicts)
        )
        preview.setStandardButtons(
            QMessageBox.Yes | QMessageBox.No if plan.items else QMessageBox.Ok
        )
        preview.setDefaultButton(QMessageBox.No if plan.items else QMessageBox.Ok)
        if preview.exec() != QMessageBox.Yes or not plan.items:
            return
        dialog = StorageMigrationDialog(plan, self._project_roots(), self)
        dialog.exec()
        if dialog.worker.error:
            QMessageBox.critical(self, tr("Migrate Legacy Resources"), dialog.worker.error)
        else:
            QMessageBox.information(self, tr("Migrate Legacy Resources"), tr(
                "Moved {count} resources. {conflicts} existing targets were skipped; "
                "their old copies have not been deleted.",
                count=len(dialog.worker.result), conflicts=len(plan.conflicts),
            ))
        self._refresh_plugin_library()


__all__ = ["SettingsView"]
