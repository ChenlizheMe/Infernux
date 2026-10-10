"""Infernux Hub 'Create New Project' plate."""

import math
import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QFrame, QWidget, QSizePolicy,
)
from PySide6.QtCore import QPoint, QPointF, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF

from model.new_project_model import NewProjectModel
from viewmodel.new_project_viewmodel import NewProjectViewModel
from hub_utils import is_frozen
from i18n import tr
from project_paths import ProjectPathError, new_project_target, validate_project_name
from view.forge import ForgeComboBox, HazardStripe, caps_font, chip, mono_label, qcolor, repolish, theme


class TemplatePreview(QWidget):
    """Painted thumbnail of the default scene: ground grid, light and a cube."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._advance)

    def showEvent(self, event):
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _advance(self):
        self._phase = (self._phase + 0.02) % (2 * math.pi)
        self.update()

    def paintEvent(self, _event):
        palette = theme()
        painter = QPainter(self)
        rect = QRectF(self.rect())
        painter.fillRect(rect, QColor(palette.bg_deep))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        horizon = rect.height() * 0.46
        cx = rect.width() / 2
        grid = qcolor(palette.grid, 40)
        painter.setPen(QPen(grid, 1))
        for index in range(-8, 9):
            painter.drawLine(QPointF(cx + index * 12, horizon), QPointF(cx + index * 60, rect.height()))
        y = horizon
        step = 4.0
        while y < rect.height():
            painter.drawLine(QPointF(0, y), QPointF(rect.width(), y))
            y += step
            step *= 1.35
        # Directional light: a sun disc with scan bars.
        sun = QPointF(rect.width() * 0.78, horizon - 34)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(qcolor(palette.accent_fill, 200))
        painter.drawEllipse(sun, 15, 15)
        for offset in range(-9, 12, 5):
            painter.fillRect(QRectF(sun.x() - 16, sun.y() + offset, 32, 1.6), QColor(palette.bg_deep))
        # A slowly turning wireframe cube on the ground plane.
        angle = self._phase
        size = 26
        base = QPointF(cx - 8, horizon + 46)

        def project(x, y, z):
            rx = x * math.cos(angle) - z * math.sin(angle)
            rz = x * math.sin(angle) + z * math.cos(angle)
            return QPointF(base.x() + rx, base.y() - y + rz * 0.35)

        corners = [(x, y, z) for x in (-size, size) for y in (0, size * 2) for z in (-size, size)]
        points = [project(*corner) for corner in corners]
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
                 if sum(c1 != c2 for c1, c2 in zip(corners[a], corners[b])) == 1]
        shadow = QPolygonF([project(-size, 0, -size), project(size, 0, -size), project(size, 0, size), project(-size, 0, size)])
        painter.setBrush(qcolor(palette.accent_deep, 90))
        painter.drawPolygon(shadow)
        painter.setPen(QPen(QColor(palette.accent), 1.4))
        for a, b in edges:
            painter.drawLine(points[a], points[b])
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(QPen(QColor(palette.accent), 1))
        painter.setFont(caps_font(9, tracking=1.2))
        painter.drawText(QRectF(10, 8, 200, 14), Qt.AlignmentFlag.AlignLeft, "SCENE  ·  MAIN")
        painter.setPen(qcolor(palette.text_muted, 200))
        painter.drawText(QRectF(10, rect.height() - 20, rect.width() - 20, 14), Qt.AlignmentFlag.AlignRight,
                         "CAM  ·  LIGHT  ·  GROUND")
        painter.end()


class NewProjectView(QDialog):
    SETTINGS_GROUP = "NewProjectDialog"
    LAST_PATH_KEY = "lastProjectPath"

    def __init__(self, version_manager=None, runtime_manager=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Create New Project"))
        self.setObjectName("newProjectDialog")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setMinimumWidth(820)
        self.setModal(True)
        self._version_manager = version_manager
        self._has_installed_versions = False
        self._drag_origin = None
        self._scrim = None

        self.settings = QSettings("InfernuxEngine", "InfernuxEngine")
        self.model = NewProjectModel()
        self.viewmodel = NewProjectViewModel(self.model)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        outer.addWidget(HazardStripe(5))
        columns = QHBoxLayout()
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(0)
        outer.addLayout(columns, 1)

        # ── Template column ──
        template = QFrame()
        template.setObjectName("templateColumn")
        template.setFixedWidth(300)
        template_layout = QVBoxLayout(template)
        template_layout.setContentsMargins(22, 22, 22, 22)
        template_layout.setSpacing(10)
        template_layout.addWidget(mono_label(tr("TEMPLATE"), "pageKicker", spacing=1.8))
        template_layout.addWidget(TemplatePreview())
        template_title = QLabel(tr("Default scene"))
        template_title.setObjectName("sectionTitle")
        template_layout.addWidget(template_title)
        template_body = QLabel(tr(
            "A main scene with a camera, a directional light and a ground plane, "
            "plus the engine's default plugins and editor settings."
        ))
        template_body.setObjectName("settingsDescription")
        template_body.setWordWrap(True)
        template_layout.addWidget(template_body)
        chips = QHBoxLayout()
        chips.setSpacing(6)
        for text in ("3D", "URP", "PYTHON"):
            chips.addWidget(chip(text))
        chips.addStretch()
        template_layout.addLayout(chips)
        template_layout.addStretch()
        self._template_note = mono_label(tr("TEMPLATES SHIP WITH EACH ENGINE"), "monoMeta", spacing=1.0)
        self._template_note.setWordWrap(True)
        template_layout.addWidget(self._template_note)
        columns.addWidget(template)

        # ── Form column ──
        form = QVBoxLayout()
        form.setContentsMargins(28, 22, 28, 0)
        form.setSpacing(6)
        form.addWidget(mono_label(tr("NEW PROJECT"), "pageKicker", spacing=1.8))
        title = QLabel(tr("Create New Project"))
        title.setObjectName("dialogTitle")
        form.addWidget(title)
        subtitle = QLabel(tr("Set the project name, location and Infernux version."))
        subtitle.setObjectName("dialogSubtitle")
        form.addWidget(subtitle)
        form.addSpacing(14)

        form.addWidget(mono_label(tr("PROJECT NAME"), "fieldLabel", spacing=1.4))
        self.name_edit = QLineEdit()
        self.name_edit.setFixedHeight(36)
        self.name_edit.setPlaceholderText(tr("Enter a name for your project"))
        self.name_edit.textChanged.connect(self.viewmodel.set_name)
        self.name_edit.textChanged.connect(self._update_create_button_state)
        form.addWidget(self.name_edit)
        self._name_hint = mono_label("", "engineCaption", spacing=0.4)
        form.addWidget(self._name_hint)
        form.addSpacing(8)

        form.addWidget(mono_label(tr("LOCATION"), "fieldLabel", spacing=1.4))
        chooser_row = QHBoxLayout()
        chooser_row.setSpacing(8)
        self.path_edit = QLineEdit()
        self.path_edit.setFixedHeight(36)
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(tr("No path selected"))
        path_button = QPushButton(tr("Browse..."))
        path_button.setObjectName("normalBtn")
        path_button.setFixedSize(104, 36)
        path_button.clicked.connect(self._on_choose_path)
        chooser_row.addWidget(self.path_edit)
        chooser_row.addWidget(path_button)
        form.addLayout(chooser_row)
        form.addSpacing(14)

        last_path = self.settings.value(f"{self.SETTINGS_GROUP}/{self.LAST_PATH_KEY}", "")
        if last_path:
            self.path_edit.setText(last_path)
            self.viewmodel.set_path(last_path)

        form.addWidget(mono_label(tr("ENGINE"), "fieldLabel", spacing=1.4))
        self.version_combo = ForgeComboBox()
        self.version_combo.setFixedHeight(36)
        form.addWidget(self.version_combo)
        self._no_version_hint = QLabel(tr(
            "No engine versions installed. Go to the Installs tab to download one first."
        ))
        self._no_version_hint.setWordWrap(True)
        self._no_version_hint.setVisible(False)
        self._no_version_hint.setObjectName("dialogErrorHint")
        form.addWidget(self._no_version_hint)
        self.version_combo.currentIndexChanged.connect(self._update_create_button_state)
        form.addSpacing(14)

        target = QFrame()
        target.setObjectName("subjectCard")
        target_layout = QVBoxLayout(target)
        target_layout.setContentsMargins(12, 9, 12, 10)
        target_layout.setSpacing(3)
        target_layout.addWidget(mono_label(tr("TARGET"), "fieldLabel", spacing=1.4))
        self._target_label = QLabel()
        self._target_label.setObjectName("cardPath")
        self._target_label.setWordWrap(True)
        self._target_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        target_layout.addWidget(self._target_label)
        form.addWidget(target)
        form.addStretch()

        form_holder = QWidget()
        form_holder.setLayout(form)
        columns.addWidget(form_holder, 1)

        rule = QFrame()
        rule.setObjectName("dialogRule")
        rule.setFixedHeight(1)
        outer.addWidget(rule)
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(24, 12, 16, 14)
        btn_layout.setSpacing(8)
        btn_layout.addWidget(mono_label("ESC · " + tr("CANCEL") + "     ENTER · " + tr("CREATE"),
                                        "monoMeta", spacing=1.2))
        btn_layout.addStretch()
        btn_cancel = QPushButton(tr("Cancel"))
        btn_cancel.setObjectName("normalBtn")
        btn_cancel.setFixedSize(100, 36)
        btn_cancel.clicked.connect(self.reject)
        btn_create = QPushButton(tr("Create"))
        btn_create.setObjectName("primaryBtn")
        btn_create.setFixedSize(132, 36)
        btn_create.setDefault(True)
        btn_create.clicked.connect(self.accept)
        self._create_btn = btn_create
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_create)
        outer.addLayout(btn_layout)

        self.name_edit.textChanged.connect(self._update_target)
        self._populate_versions()
        self._suggest_name()
        self._update_target()
        self.name_edit.setFocus()

    # ── Data ─────────────────────────────────────────────────────────

    def _suggest_name(self):
        base = self.path_edit.text().strip()
        candidate = tr("My Project")
        if base:
            index = 1
            while os.path.exists(os.path.join(base, candidate)) and index < 99:
                index += 1
                candidate = f"{tr('My Project')} {index}"
        self.name_edit.setText(candidate)
        self.name_edit.selectAll()

    def _populate_versions(self):
        """Fill the version combo box."""
        self.version_combo.clear()
        installed: list[str] = []
        dev_mode = not is_frozen()
        if self._version_manager is not None:
            installed = self._version_manager.installed_versions()
            self._has_installed_versions = bool(installed)
            for index, version in enumerate(installed):
                python = ""
                try:
                    python = self._version_manager.python_version_for_engine(version)
                except Exception:
                    python = ""
                self.version_combo.addItem(
                    version, version, meta=f"PY {python}" if python else "",
                    tag=tr("LATEST") if index == 0 else "",
                )
            if not installed and not dev_mode:
                self.version_combo.addItem(tr("(no versions installed)"), "")

        if dev_mode:
            self.version_combo.insertItem(0, tr("dev (current environment)"), "")
            self.version_combo.setCurrentIndex(0)

        self._update_create_button_state()

    def _has_selected_version(self) -> bool:
        if not is_frozen():
            return self.version_combo.currentIndex() >= 0
        return bool(self.version_combo.currentData())

    def _name_error(self) -> str:
        try:
            validate_project_name(self.name_edit.text())
        except ProjectPathError as exc:
            return str(exc) if self.name_edit.text().strip() else ""
        return ""

    def _update_create_button_state(self):
        has_version = self._has_selected_version()
        if is_frozen():
            self._no_version_hint.setText(tr(
                "No engine versions installed. Go to the Installs tab to download one first."
                if not self._has_installed_versions
                else "Select an installed engine version before creating a project."
            ))
            self._no_version_hint.setVisible(not has_version)
        else:
            self._no_version_hint.setVisible(False)
        error = self._name_error()
        self._name_hint.setText(error)
        self._name_hint.setProperty("kind", "error" if error else "")
        repolish(self._name_hint)
        self.name_edit.setProperty("invalid", bool(error))
        repolish(self.name_edit)
        self._create_btn.setEnabled(self.viewmodel.is_valid() and has_version)

    def _update_target(self):
        name = self.name_edit.text().strip()
        base = self.path_edit.text().strip()
        if not (name and base):
            self._target_label.setText(tr("Choose a name and a location"))
            return
        try:
            _parent, target = new_project_target(base, name)
        except ProjectPathError as exc:
            self._target_label.setText(str(exc).splitlines()[0])
            return
        prefix = tr("Already exists") + "  ·  " if os.path.exists(target) else ""
        self._target_label.setText(prefix + target)

    def _on_choose_path(self):
        current_path = self.path_edit.text() or ""
        folder = QFileDialog.getExistingDirectory(self, tr("Choose Project Location"), current_path)
        if folder:
            self.path_edit.setText(folder)
            self.viewmodel.set_path(folder)
            self.settings.setValue(f"{self.SETTINGS_GROUP}/{self.LAST_PATH_KEY}", folder)
            self._update_create_button_state()
            self._update_target()

    def accept(self):
        self._update_create_button_state()
        if not self._create_btn.isEnabled():
            return
        super().accept()

    def get_data(self):
        name, path = self.viewmodel.get_data()
        version = self.version_combo.currentData() or ""
        return name, path, version

    # ── Frameless plate behaviour ────────────────────────────────────

    def paintEvent(self, event):
        palette = theme()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(palette.bg_surface))
        painter.fillRect(QRectF(1, 6, 300, self.height() - 7), QColor(palette.bg_input))
        painter.setPen(QColor(palette.border_hover))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()
        super().paintEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 64:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        owner = self.parentWidget().window() if self.parentWidget() is not None else None
        if owner is not None and owner.isVisible() and self._scrim is None:
            from view.dialogs import _Scrim
            self._scrim = _Scrim(owner)
            self._scrim.fade(1.0)
            center = owner.frameGeometry().center()
            self.adjustSize()
            self.move(QPoint(center.x() - self.width() // 2, center.y() - self.height() // 2))

    def done(self, result):
        if self._scrim is not None:
            self._scrim.fade(0.0, then_delete=True)
            self._scrim = None
        super().done(result)
