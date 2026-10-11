"""Project list: tape-label rows with engine binding, pins, sorting and quick fixes."""

import json
import os
import time

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QFrame, QPushButton, QLineEdit, QMenu,
)
from PySide6.QtCore import QPoint, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut

from database import ProjectDatabase
from hub_utils import is_frozen, is_project_open
from i18n import tr
from project_python_runtime import read_project_python_version
from version_manager import VersionManager, display_release
from view.forge import Monogram, StatusLed, chip, fmt_age, mono_label, repolish, toast, track
from view.hover_widgets import AnimatedSurfaceFrame
from model.project_model import source_engine_version


# Fixed column widths shared by the header and every row.
ENGINE_COLUMN = 196
MODIFIED_COLUMN = 96
LAUNCH_COLUMN = 96
MENU_COLUMN = 34

PIN_SETTING = "pinned_projects"
SORT_SETTING = "project_sort"


def project_modified_time(path: str) -> float:
    """Newest mtime among the folders an editor session writes; cheap stats only."""
    newest = 0.0
    for relative in ("", "Assets", "ProjectSettings", "Library"):
        try:
            newest = max(newest, os.stat(os.path.join(path, relative)).st_mtime)
        except OSError:
            continue
    return newest


def _modified_text(timestamp: float) -> str:
    if not timestamp:
        return "—"
    if time.time() - timestamp < 7 * 86400:
        return fmt_age(timestamp)
    return time.strftime("%Y-%m-%d", time.localtime(timestamp))


class _ProjectCard(AnimatedSurfaceFrame):
    """One project row. Unavailable rows stay interactive; only launching is blocked."""

    launch_clicked = Signal(str)
    fix_clicked = Signal(str, str)  # project id, action ("install" or "locate")

    def __init__(self, project_id: str, name: str, created_at: str, path: str,
                 version_manager=None, on_remove_requested=None, parent=None, *, pinned: bool = False):
        super().__init__("projectCard", parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setMouseTracking(True)
        self.project_name = name
        self.project_id = project_id
        self.project_path = path
        self.engine_version = ""
        self.unavailable_reason = ""
        self.pinned = pinned
        self.modified = project_modified_time(path)
        self._launchable = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(64)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 8, 10)
        layout.setSpacing(14)

        self._monogram = Monogram(name)
        layout.addWidget(self._monogram)

        text_col = QVBoxLayout()
        text_col.setSpacing(3)
        text_col.setContentsMargins(0, 0, 0, 0)
        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        name_label = QLabel(name)
        name_label.setObjectName("cardName")
        name_label.setToolTip(name)
        name_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        name_row.addWidget(name_label, 1)
        self._pin_mark = chip(tr("PINNED"), "paper")
        self._pin_mark.setVisible(pinned)
        name_row.addWidget(self._pin_mark)
        text_col.addLayout(name_row)
        path_label = QLabel(path)
        path_label.setObjectName("cardPath")
        path_label.setToolTip(path)
        path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        text_col.addWidget(path_label)
        layout.addLayout(text_col, 1)

        engine_cell = QHBoxLayout()
        engine_cell.setSpacing(8)
        engine_cell.setContentsMargins(0, 0, 0, 0)
        self._led = StatusLed("idle")
        engine_cell.addWidget(self._led, 0, Qt.AlignmentFlag.AlignVCenter)
        engine_stack = QVBoxLayout()
        engine_stack.setSpacing(3)
        engine_stack.setContentsMargins(0, 0, 0, 0)
        engine_cell.addLayout(engine_stack, 1)
        engine_holder = QWidget()
        engine_holder.setFixedWidth(ENGINE_COLUMN)
        engine_holder.setLayout(engine_cell)

        def caption(text: str, kind: str) -> QLabel:
            label = mono_label(text, "engineCaption", spacing=0.4)
            label.setProperty("kind", kind)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            return label

        fix_action = ""
        if not os.path.isdir(path):
            status_label = QLabel(tr("MISSING"))
            status_label.setObjectName("projectStatus")
            status_label.setProperty("kind", "error")
            status_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            engine_stack.addWidget(status_label, 0, Qt.AlignmentFlag.AlignLeft)
            engine_stack.addWidget(caption(tr("Project path missing"), "error"))
            self.unavailable_reason = tr("Project path missing") + "\n" + path
            self._led.set_kind("error")
            fix_action = "locate"
        else:
            version = VersionManager.read_project_version(path) or ""
            self.engine_version = version
            version_label = QLabel(display_release(version) if version else tr("Unversioned"))
            version_label.setObjectName("projectVersion")
            version_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            unavailable_reason = ""
            if is_frozen() and version_manager is not None:
                try:
                    python_version = read_project_python_version(path)
                except (OSError, RuntimeError, ValueError) as exc:
                    python_version = ""
                    unavailable_reason = str(exc)
                if not version:
                    unavailable_reason = tr("Project engine version is missing")
                elif not python_version:
                    unavailable_reason = tr("Project Python version is missing")
                elif not version_manager.is_installed(version, python_version):
                    unavailable_reason = (
                        f"Infernux {version} / Python {python_version} "
                        f"{tr('is not installed')}"
                    )
                    fix_action = "install"
            elif version and not is_frozen() and version != source_engine_version():
                unavailable_reason = (
                    f"Project requires Infernux {version}; the source engine is {source_engine_version()}"
                )

            engine_stack.addWidget(version_label, 0, Qt.AlignmentFlag.AlignLeft)
            if unavailable_reason:
                self.unavailable_reason = unavailable_reason
                version_label.setText(display_release(version) if version else tr("Unknown"))
                version_label.setProperty("kind", "warning")
                version_label.setToolTip(unavailable_reason)
                required = caption(tr("Install required version"), "warning")
                required.setToolTip(unavailable_reason)
                engine_stack.addWidget(required)
                self._led.set_kind("warn")
            elif is_project_open(path):
                version_label.setProperty("kind", "active")
                version_label.setToolTip(tr("Project Already Open"))
                engine_stack.addWidget(caption(tr("Open in the editor"), "ok"))
                self._led.set_kind("busy")
            else:
                version_label.setProperty("kind", "ready")
                version_label.setToolTip(f"Infernux {version}" if version else "")  # exact pin
                self._led.set_kind("ok")
                self._launchable = True
        layout.addWidget(engine_holder)

        if self.unavailable_reason:
            self.setProperty("unavailable", True)
            self.setToolTip(self.unavailable_reason)
            self._monogram.set_dim(True)

        modified = mono_label(_modified_text(self.modified), "cardDate", spacing=0.4)
        modified.setFixedWidth(MODIFIED_COLUMN)
        layout.addWidget(modified)

        slot = QWidget()
        slot.setFixedWidth(LAUNCH_COLUMN)
        slot_layout = QHBoxLayout(slot)
        slot_layout.setContentsMargins(0, 0, 0, 0)
        self._launch_button = QPushButton(tr("LAUNCH"))
        self._launch_button.setObjectName("launchBtn")
        self._launch_button.setFixedSize(LAUNCH_COLUMN, 30)
        self._launch_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._launch_button.setToolTip(tr("Open this project in the Infernux editor"))
        self._launch_button.setVisible(self._launchable)
        self._launch_button.clicked.connect(lambda: self.launch_clicked.emit(self.project_id))
        slot_layout.addWidget(self._launch_button)
        self._fix_button = QPushButton(tr("INSTALL") if fix_action == "install" else tr("LOCATE"))
        self._fix_button.setObjectName("ghostBtn")
        self._fix_button.setFixedSize(LAUNCH_COLUMN, 30)
        self._fix_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fix_button.setToolTip(
            tr("Install the engine version this project needs") if fix_action == "install"
            else tr("Point Hub at the project's new folder")
        )
        self._fix_button.setVisible(bool(fix_action))
        self._fix_button.clicked.connect(lambda: self.fix_clicked.emit(self.project_id, fix_action))
        slot_layout.addWidget(self._fix_button)
        layout.addWidget(slot)

        open_btn = QPushButton("···")
        open_btn.setObjectName("cardOpenBtn")
        open_btn.setFixedSize(MENU_COLUMN, 30)
        open_btn.setToolTip(tr("Project actions"))
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._actions_button = open_btn
        self._actions_menu = QMenu(open_btn)
        self._remove_callback = on_remove_requested
        self._fix_action = fix_action
        self._build_menu()
        open_btn.clicked.connect(self._show_actions_menu)
        layout.addWidget(open_btn)

    def _build_menu(self):
        menu = self._actions_menu
        menu.clear()
        if self._launchable:
            menu.addAction(tr("Launch"), lambda: self.launch_clicked.emit(self.project_id))
            menu.addSeparator()
        if self._fix_action == "install":
            menu.addAction(tr("Install required engine"), lambda: self.fix_clicked.emit(self.project_id, "install"))
        show = menu.addAction(tr("Show in Explorer"),
                              lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.project_path)))
        show.setEnabled(os.path.isdir(self.project_path))
        menu.addAction(tr("Copy path"), self._copy_path)
        menu.addAction(tr("Unpin") if self.pinned else tr("Pin to top"),
                       lambda: self.fix_clicked.emit(self.project_id, "pin"))
        menu.addAction(tr("Relocate..."), lambda: self.fix_clicked.emit(self.project_id, "locate"))
        menu.addSeparator()
        menu.addAction(
            tr("Remove from Hub"),
            lambda: self._remove_callback(self.project_id) if self._remove_callback is not None else None,
        )

    def _copy_path(self):
        QApplication.clipboard().setText(self.project_path)
        toast(self, tr("Copied the project path"), "info", timeout_ms=2400)

    def _show_actions_menu(self):
        # Right-align the menu under the button so it never covers the row's text.
        anchor = self._actions_button.mapToGlobal(self._actions_button.rect().bottomRight())
        self._actions_menu.popup(anchor + QPoint(-self._actions_menu.sizeHint().width(), 2))

    def contextMenuEvent(self, event):
        self._actions_menu.popup(event.globalPos())

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        self.set_selected_animated(selected)

    @property
    def can_select(self) -> bool:
        return self._launchable

    @property
    def launchable(self) -> bool:
        return self._launchable


class _EmptyState(QFrame):
    create_clicked = Signal()
    open_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("emptyState")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)
        kicker = mono_label(tr("NO SIGNAL  ·  0 PROJECTS"), "monoKicker", spacing=2.0)
        kicker.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(kicker)
        empty_title = QLabel(tr("No projects yet"))
        empty_title.setObjectName("emptyTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(empty_title)
        empty_detail = QLabel(tr("Create a new project, open an existing one, or drop a project folder here."))
        empty_detail.setObjectName("emptyHint")
        empty_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_detail.setWordWrap(True)
        layout.addWidget(empty_detail)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch()
        open_button = QPushButton(tr("Open"))
        open_button.setObjectName("normalBtn")
        open_button.setFixedHeight(34)
        open_button.clicked.connect(self.open_clicked)
        buttons.addWidget(open_button)
        create = QPushButton(tr("New project"))
        create.setObjectName("primaryBtn")
        create.setFixedHeight(34)
        create.clicked.connect(self.create_clicked)
        buttons.addWidget(create)
        buttons.addStretch()
        layout.addSpacing(6)
        layout.addLayout(buttons)


class ProjectListPane(QWidget):
    """Scrollable list of project rows with search, sort, pins and keyboard control."""

    remove_requested = Signal(str)
    launch_requested = Signal(str)
    create_requested = Signal()
    open_requested = Signal()
    install_requested = Signal(str)   # engine version a project needs
    relocate_requested = Signal(str)  # project id

    SORTS = ("recent", "name")

    def __init__(self, db: ProjectDatabase, version_manager=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.version_manager = version_manager
        self.selected_project_id = None
        self.project_cards: dict[str, _ProjectCard] = {}
        self._all_projects = []
        self._sort = self._setting(SORT_SETTING, "recent")
        if self._sort not in self.SORTS:
            self._sort = "recent"
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(12)
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("searchBox")
        self.search_edit.setPlaceholderText(tr("Search projects...") + "   Ctrl+F")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedHeight(34)
        self.search_edit.setMaximumWidth(340)
        self.search_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.search_edit.textChanged.connect(self._apply_filter)
        toolbar.addWidget(self.search_edit, 1)
        self._count_label = mono_label("", "monoMeta", spacing=1.2)
        toolbar.addWidget(self._count_label)
        toolbar.addStretch()
        toolbar.addWidget(mono_label(tr("SORT"), "columnHeader", spacing=1.4))
        self._sort_buttons = {}
        segment = QHBoxLayout()
        segment.setSpacing(0)
        for index, (key, label) in enumerate((("recent", tr("RECENT")), ("name", tr("A–Z")))):
            button = QPushButton(label)
            button.setObjectName("segmentBtn")
            button.setCheckable(True)
            button.setChecked(key == self._sort)
            button.setProperty("last", index == 1)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, value=key: self.set_sort(value))
            segment.addWidget(button)
            self._sort_buttons[key] = button
        toolbar.addLayout(segment)
        main_layout.addLayout(toolbar)
        shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        shortcut.activated.connect(lambda: (self.search_edit.setFocus(), self.search_edit.selectAll()))

        header = QHBoxLayout()
        header.setContentsMargins(14, 0, 8 + 4, 0)
        header.setSpacing(14)
        spacer = QWidget()
        spacer.setFixedWidth(40)
        header.addWidget(spacer)
        header.addWidget(self._column(tr("PROJECT")), 1)
        header.addWidget(self._column(tr("ENGINE"), ENGINE_COLUMN))
        header.addWidget(self._column(tr("MODIFIED"), MODIFIED_COLUMN))
        header.addSpacing(LAUNCH_COLUMN + MENU_COLUMN + 14)
        self._header = QWidget()
        self._header.setLayout(header)
        main_layout.addWidget(self._header)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("projectScrollArea")
        scroll_area.viewport().setObjectName("projectViewport")
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_layout.addWidget(scroll_area)

        self.container = QWidget()
        self.container.setObjectName("projectListContainer")
        self.card_layout = QVBoxLayout(self.container)
        self.card_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.card_layout.setSpacing(4)
        self.card_layout.setContentsMargins(0, 0, 4, 0)
        scroll_area.setWidget(self.container)
        self._scroll = scroll_area

        self._footer = mono_label(
            tr("ENTER LAUNCH  ·  UP/DOWN SELECT  ·  DEL REMOVE  ·  RIGHT-CLICK FOR ACTIONS  ·  DROP A FOLDER TO ADD"),
            "monoMeta", spacing=1.0,
        )
        main_layout.addWidget(self._footer)

        self.refresh()

    @staticmethod
    def _column(text: str, width: int | None = None) -> QLabel:
        label = track(QLabel(text), 1.4)
        label.setObjectName("columnHeader")
        if width is not None:
            label.setFixedWidth(width)
        return label

    # ── Preferences ──────────────────────────────────────────────────

    def _setting(self, key: str, default: str = "") -> str:
        getter = getattr(self.db, "get_setting", None)
        try:
            return getter(key, default) if getter is not None else default
        except Exception:
            return default

    def _save_setting(self, key: str, value: str) -> None:
        setter = getattr(self.db, "set_setting", None)
        if setter is not None:
            setter(key, value)

    def pinned_ids(self) -> list[str]:
        try:
            value = json.loads(self._setting(PIN_SETTING, "[]") or "[]")
        except ValueError:
            return []
        return [str(item) for item in value] if isinstance(value, list) else []

    def toggle_pin(self, project_id: str) -> None:
        pinned = self.pinned_ids()
        if project_id in pinned:
            pinned.remove(project_id)
        else:
            pinned.insert(0, project_id)
        self._save_setting(PIN_SETTING, json.dumps(pinned))
        self.refresh()

    def set_sort(self, sort: str) -> None:
        self._sort = sort if sort in self.SORTS else "recent"
        for key, button in self._sort_buttons.items():
            button.setChecked(key == self._sort)
        self._save_setting(SORT_SETTING, self._sort)
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self):
        previous_selection = self.selected_project_id
        self.project_cards.clear()
        while self.card_layout.count():
            item = self.card_layout.takeAt(0)
            w = item.widget()
            if w:
                w.hide()
                w.deleteLater()

        self._all_projects = self.db.all_projects()
        pinned = self.pinned_ids()
        cards = []
        for record in self._all_projects:
            card = _ProjectCard(
                record.project_id, record.name, record.created_at, record.path,
                self.version_manager, self.remove_requested.emit, pinned=record.project_id in pinned,
            )
            card.mousePressEvent = lambda ev, pid=record.project_id, owner=card: self._on_press(ev, pid, owner)
            card.mouseDoubleClickEvent = lambda _ev, pid=record.project_id: self._on_double_click(pid)
            card.launch_clicked.connect(self._on_double_click)
            card.fix_clicked.connect(self._on_fix)
            cards.append(card)

        def order(card):
            pin_rank = pinned.index(card.project_id) if card.project_id in pinned else len(pinned)
            if self._sort == "name":
                return (pin_rank, card.project_name.casefold())
            return (pin_rank, -card.modified, card.project_name.casefold())

        for card in sorted(cards, key=order):
            self.card_layout.addWidget(card)
            self.project_cards[card.project_id] = card

        if not self._all_projects:
            empty = _EmptyState()
            empty.create_clicked.connect(self.create_requested)
            empty.open_clicked.connect(self.open_requested)
            self.card_layout.addWidget(empty)
        self._header.setVisible(bool(self._all_projects))
        self._footer.setVisible(bool(self._all_projects))
        self.card_layout.addStretch()
        self._apply_filter(self.search_edit.text())
        if (
            previous_selection in self.project_cards
            and self.project_cards[previous_selection].can_select
        ):
            self._on_select(previous_selection)
        else:
            self.selected_project_id = None

    # ------------------------------------------------------------------
    def _apply_filter(self, text: str):
        needle = text.strip().lower()
        shown = 0
        for card in self.project_cards.values():
            searchable = f"{card.project_name} {card.project_path} {card.engine_version}".lower()
            visible = needle in searchable if needle else True
            card.setVisible(visible)
            shown += visible
        total = len(self.project_cards)
        if needle:
            self._count_label.setText(tr("{shown} / {total} PROJECTS", shown=f"{shown:02d}", total=f"{total:02d}"))
        else:
            self._count_label.setText(tr("{total} PROJECTS", total=f"{total:02d}"))

    def _on_press(self, event, project_id: str, card: _ProjectCard):
        if event is not None and event.button() == Qt.MouseButton.RightButton:
            return
        if card.can_select:
            self._on_select(project_id)
        else:
            self.setFocus(Qt.FocusReason.MouseFocusReason)

    def _on_select(self, project_id: str):
        card = self.project_cards.get(project_id)
        if card is None:
            self.selected_project_id = None
            for candidate in self.project_cards.values():
                candidate.set_selected(False)
            return
        if not card.can_select:
            return
        self.selected_project_id = project_id
        for candidate_id, candidate in self.project_cards.items():
            candidate.set_selected(candidate_id == project_id)
        self._scroll.ensureWidgetVisible(card, 0, 8)
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_double_click(self, project_id: str):
        """Select the row and ask the window to launch it."""
        self._on_select(project_id)
        if self.selected_project_id == project_id:
            self.launch_requested.emit(project_id)

    def _on_fix(self, project_id: str, action: str):
        card = self.project_cards.get(project_id)
        if action == "pin":
            self.toggle_pin(project_id)
        elif action == "install" and card is not None:
            self.install_requested.emit(card.engine_version)
        elif action == "locate":
            self.relocate_requested.emit(project_id)

    def keyPressEvent(self, event):
        visible = [pid for pid, card in self.project_cards.items() if not card.isHidden() and card.can_select]
        key = event.key()
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Up) and visible:
            if self.selected_project_id in visible:
                index = visible.index(self.selected_project_id) + (1 if key == Qt.Key.Key_Down else -1)
            else:
                index = 0
            self._on_select(visible[max(0, min(len(visible) - 1, index))])
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.selected_project_id:
            self.launch_requested.emit(self.selected_project_id)
            return
        if key == Qt.Key.Key_Delete and self.selected_project_id:
            self.remove_requested.emit(self.selected_project_id)
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_selected_project(self):
        record = self.get_selected_record()
        return record.name if record else None

    def get_selected_project_id(self):
        return self.selected_project_id

    def get_selected_record(self):
        if self.selected_project_id:
            return self.db.get_project(self.selected_project_id)
        return None

    def get_selected_project_path(self):
        record = self.get_selected_record()
        return record.path if record else None

    def select_project(self, project_id: str):
        if project_id in self.project_cards:
            self._on_select(project_id)
