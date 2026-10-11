"""Hub dialogs: frameless FORGE SIGNAL modals replacing QMessageBox.

``question/information/warning/critical`` mirror the QMessageBox static API
(same arguments, StandardButton results) so call sites read the same, while
``confirm`` and ``HubDialog`` expose the richer layout: a level glyph,
hazard band, optional subject card, collapsible details and keyboard hints.
The parent window is dimmed by a scanline scrim while a dialog is open.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QParallelAnimationGroup, QPoint, QPropertyAnimation, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QDialog, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from i18n import tr
from view.forge import GlyphPlate, HazardStripe, Monogram, level_color, mono_label, qcolor, theme

StandardButton = QMessageBox.StandardButton

_LEVEL_GLYPH = {"info": "info", "question": "question", "warning": "warning", "critical": "critical",
                "ok": "ok", "remove": "remove"}
_LEVEL_KICKER = {"info": "NOTICE", "question": "CONFIRM", "warning": "CAUTION", "critical": "FAULT",
                 "ok": "COMPLETE", "remove": "REMOVE"}
_ROLE_OBJECT = {"accept": "primaryBtn", "destructive": "dangerSolidBtn", "reject": "normalBtn", "action": "normalBtn"}
_STANDARD_LABELS = {
    StandardButton.Yes: "Yes", StandardButton.No: "No", StandardButton.Ok: "OK",
    StandardButton.Cancel: "Cancel", StandardButton.Close: "Close", StandardButton.Retry: "Retry",
    StandardButton.Ignore: "Ignore",
}


class _Scrim(QWidget):
    """Dims the owning window behind a modal; painted scanlines keep it in style."""

    def __init__(self, window: QWidget):
        super().__init__(window)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setGeometry(window.rect())
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.0)
        self.setGraphicsEffect(effect)
        self._fade = QPropertyAnimation(effect, b"opacity", self)
        self._fade.setDuration(180)
        self.show()
        self.raise_()

    def fade(self, target: float, *, then_delete: bool = False):
        self._fade.stop()
        self._fade.setEndValue(target)
        if then_delete:
            self._fade.finished.connect(self.deleteLater)
        self._fade.start()

    def paintEvent(self, _event):
        palette = theme()
        painter = QPainter(self)
        painter.fillRect(self.rect(), qcolor(palette.overlay, 165))
        line = qcolor(palette.grid, 7)
        for y in range(0, self.height(), 3):
            painter.fillRect(0, y, self.width(), 1, line)
        painter.end()


class HubDialog(QDialog):
    """A modal plate. QMessageBox-like accessors keep tests and callers simple."""

    def __init__(self, parent=None, *, level: str = "info", title: str = "", text: str = "",
                 kicker: str = "", detail: str = "", subject: tuple[str, str] | None = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle(title or "Infernux Hub")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setMaximumWidth(760)
        self._level = level
        self._text = text
        self._detail = detail
        self._buttons: list[tuple[QPushButton, str]] = []
        self._default: QPushButton | None = None
        self._escape: QPushButton | None = None
        self._clicked: QPushButton | None = None
        self._drag_origin: QPoint | None = None
        self._scrim: _Scrim | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        stripe_color = level_color(level).name()
        outer.addWidget(HazardStripe(5, color=stripe_color))

        body = QHBoxLayout()
        body.setContentsMargins(24, 20, 24, 18)
        body.setSpacing(18)
        body.addWidget(GlyphPlate(_LEVEL_GLYPH.get(level, "info"), level), 0, Qt.AlignmentFlag.AlignTop)
        column = QVBoxLayout()
        column.setSpacing(6)
        kicker_text = kicker or tr(_LEVEL_KICKER.get(level, "NOTICE"))
        self._kicker = mono_label(f"§  {kicker_text}", "pageKicker", spacing=1.8)
        column.addWidget(self._kicker)
        self._title = QLabel(title)
        self._title.setObjectName("dialogTitle")
        self._title.setWordWrap(True)
        self._title.setVisible(bool(title))
        column.addWidget(self._title)
        self._body = QLabel(text)
        self._body.setObjectName("dialogBody")
        self._body.setWordWrap(True)
        self._body.setTextFormat(Qt.TextFormat.PlainText)
        self._body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._body.setVisible(bool(text))
        column.addWidget(self._body)
        if subject is not None:
            column.addSpacing(4)
            column.addWidget(self._subject_card(*subject))
        if detail:
            column.addSpacing(2)
            self._detail_toggle = QPushButton(tr("Show details"))
            self._detail_toggle.setObjectName("ghostBtn")
            self._detail_toggle.setFixedHeight(26)
            self._detail_toggle.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            self._detail_toggle.clicked.connect(self._toggle_detail)
            column.addWidget(self._detail_toggle, 0, Qt.AlignmentFlag.AlignLeft)
            self._detail_frame = QFrame()
            self._detail_frame.setObjectName("dialogDetail")
            frame_layout = QVBoxLayout(self._detail_frame)
            frame_layout.setContentsMargins(10, 8, 6, 8)
            view = QPlainTextEdit(detail)
            view.setObjectName("dialogDetailText")
            view.setReadOnly(True)
            view.setMinimumHeight(120)
            view.setMaximumHeight(220)
            frame_layout.addWidget(view)
            self._detail_frame.hide()
            column.addWidget(self._detail_frame)
        body.addLayout(column, 1)
        outer.addLayout(body)

        rule = QFrame()
        rule.setObjectName("dialogRule")
        rule.setFixedHeight(1)
        outer.addWidget(rule)
        footer = QHBoxLayout()
        footer.setContentsMargins(24, 12, 16, 14)
        footer.setSpacing(8)
        self._hint = mono_label("", "monoMeta", spacing=1.2)
        footer.addWidget(self._hint)
        footer.addStretch()
        self._footer = footer
        outer.addLayout(footer)

    # ── QMessageBox-compatible surface ───────────────────────────────

    def text(self) -> str:
        return self._text

    def detailedText(self) -> str:  # noqa: N802 - Qt naming
        return self._detail

    def addButton(self, label: str, role: str = "action") -> QPushButton:  # noqa: N802
        button = QPushButton(label)
        button.setObjectName(_ROLE_OBJECT.get(role, "normalBtn"))
        button.setFixedHeight(34)
        button.setMinimumWidth(96)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setAutoDefault(False)
        button.clicked.connect(lambda _checked=False, item=button: self._finish(item))
        self._footer.addWidget(button)
        self._buttons.append((button, role))
        # Three or more actions need room for the keyboard hint as well.
        self.setMinimumWidth(max(self.minimumWidth(), 360 + 112 * len(self._buttons)))
        if role == "reject" and self._escape is None:
            self._escape = button
        self._update_hint()
        return button

    def setDefaultButton(self, button: QPushButton) -> None:  # noqa: N802
        self._default = button
        for candidate, _role in self._buttons:
            candidate.setDefault(candidate is button)
        self._update_hint()

    def defaultButton(self) -> QPushButton | None:  # noqa: N802
        return self._default

    def setEscapeButton(self, button: QPushButton) -> None:  # noqa: N802
        self._escape = button
        self._update_hint()

    def clickedButton(self) -> QPushButton | None:  # noqa: N802
        return self._clicked

    def add_link(self, label: str, url: str) -> QPushButton:
        """A footer link that opens ``url`` without closing the dialog."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        button = QPushButton(label)
        button.setObjectName("ghostBtn")
        button.setFixedHeight(30)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
        self._footer.insertWidget(1, button)
        return button

    # ── Behaviour ────────────────────────────────────────────────────

    def _subject_card(self, name: str, detail: str) -> QFrame:
        card = QFrame()
        card.setObjectName("subjectCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(12)
        layout.addWidget(Monogram(name, size=34))
        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel(name)
        title.setObjectName("cardName")
        text.addWidget(title)
        path = QLabel(detail)
        path.setObjectName("cardPath")
        path.setWordWrap(True)
        path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text.addWidget(path)
        layout.addLayout(text, 1)
        return card

    def _toggle_detail(self):
        visible = not self._detail_frame.isVisible()
        self._detail_frame.setVisible(visible)
        self._detail_toggle.setText(tr("Hide details") if visible else tr("Show details"))
        self.adjustSize()

    def _update_hint(self):
        parts = []
        if self._escape is not None:
            parts.append("ESC · " + self._escape.text().upper())
        if self._default is not None and self._default is not self._escape:
            parts.append("ENTER · " + self._default.text().upper())
        self._hint.setText("     ".join(parts))

    def _finish(self, button: QPushButton):
        self._clicked = button
        role = next((role for candidate, role in self._buttons if candidate is button), "action")
        if role == "reject":
            self.reject()
        else:
            self.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Escape:
            if self._escape is not None:
                self._finish(self._escape)
            else:
                self.reject()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._default is not None:
            self._finish(self._default)
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() < 72:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        palette = theme()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(palette.bg_surface))
        painter.setPen(QColor(palette.border_hover))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        accent = level_color(self._level, palette)
        # Corner registration marks on the footer.
        painter.fillRect(self.width() - 7, self.height() - 7, 4, 4, accent)
        painter.fillRect(3, self.height() - 7, 4, 4, qcolor(palette.text_muted, 140))
        painter.end()
        super().paintEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        owner = self.parentWidget().window() if self.parentWidget() is not None else None
        if owner is not None and owner.isVisible() and self._scrim is None:
            self._scrim = _Scrim(owner)
            self._scrim.fade(1.0)
        self.adjustSize()
        if owner is not None and owner.isVisible():
            center = owner.frameGeometry().center()
        else:
            screen = self.screen() or (owner.screen() if owner is not None else None)
            center = screen.availableGeometry().center() if screen is not None else self.pos()
        target = QPoint(center.x() - self.width() // 2, center.y() - self.height() // 2 - 24)
        self.move(target + QPoint(0, 14))
        self.setWindowOpacity(0.0)
        group = QParallelAnimationGroup(self)
        slide = QPropertyAnimation(self, b"pos", group)
        slide.setDuration(210)
        slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        slide.setEndValue(target)
        fade = QPropertyAnimation(self, b"windowOpacity", group)
        fade.setDuration(170)
        fade.setEndValue(1.0)
        group.addAnimation(slide)
        group.addAnimation(fade)
        group.start()
        self._entrance = group
        if self._default is not None:
            self._default.setFocus()

    def done(self, result):
        if self._scrim is not None:
            self._scrim.fade(0.0, then_delete=True)
            self._scrim = None
        super().done(result)


# ── QMessageBox-compatible helpers ───────────────────────────────────

def _standard_dialog(level: str, parent, title: str, text: str, buttons, default_button, *,
                     kicker: str = "", detail: str = "", subject=None, labels: dict | None = None,
                     destructive: bool = False, link: tuple[str, str] | None = None) -> StandardButton:
    dialog = HubDialog(parent, level=level, title=title, text=text, kicker=kicker, detail=detail, subject=subject)
    if link is not None:
        dialog.add_link(*link)
    order = (StandardButton.Ignore, StandardButton.Cancel, StandardButton.No, StandardButton.Close,
             StandardButton.Retry, StandardButton.Ok, StandardButton.Yes)
    labels = labels or {}
    mapping: dict[int, StandardButton] = {}
    flags = StandardButton(buttons)
    created = []
    for standard in order:
        if flags & standard:
            negative = standard in (StandardButton.Cancel, StandardButton.No, StandardButton.Close)
            role = ("reject" if negative else "action" if standard == StandardButton.Ignore
                    else "destructive" if destructive else "accept")
            button = dialog.addButton(labels.get(standard, tr(_STANDARD_LABELS.get(standard, "OK"))), role)
            mapping[id(button)] = standard
            created.append((standard, button))
    if not created:
        button = dialog.addButton(tr("OK"), "accept")
        mapping[id(button)] = StandardButton.Ok
        created.append((StandardButton.Ok, button))
    default = next((button for standard, button in created if standard == default_button), None)
    if default is None:
        # Questions default to the safe answer; notices to their only action.
        negative = next((button for standard, button in created
                         if standard in (StandardButton.No, StandardButton.Cancel)), None)
        default = negative if level in ("question", "remove") and negative is not None else created[-1][1]
    dialog.setDefaultButton(default)
    if len(created) == 1:
        dialog.setEscapeButton(created[0][1])
    dialog.exec()
    clicked = dialog.clickedButton()
    if clicked is None:
        negative = next((standard for standard, _button in created
                         if standard in (StandardButton.No, StandardButton.Cancel, StandardButton.Close)), None)
        return negative or created[0][0]
    return mapping.get(id(clicked), StandardButton.NoButton)


def question(parent, title, text, buttons=StandardButton.Yes | StandardButton.No,
             defaultButton=StandardButton.NoButton, **options) -> StandardButton:  # noqa: N803
    return _standard_dialog(options.pop("level", "question"), parent, title, text, buttons, defaultButton, **options)


def information(parent, title, text, buttons=StandardButton.Ok,
                defaultButton=StandardButton.NoButton, **options) -> StandardButton:  # noqa: N803
    return _standard_dialog(options.pop("level", "info"), parent, title, text, buttons, defaultButton, **options)


def warning(parent, title, text, buttons=StandardButton.Ok,
            defaultButton=StandardButton.NoButton, **options) -> StandardButton:  # noqa: N803
    return _standard_dialog("warning", parent, title, text, buttons, defaultButton, **options)


def critical(parent, title, text, buttons=StandardButton.Ok,
             defaultButton=StandardButton.NoButton, **options) -> StandardButton:  # noqa: N803
    return _standard_dialog("critical", parent, title, text, buttons, defaultButton, **options)


def confirm(parent, title: str, text: str, *, confirm_label: str, cancel_label: str = "",
            destructive: bool = False, kicker: str = "", detail: str = "", subject=None,
            level: str = "") -> bool:
    """Two-button confirmation with explicit verbs; Enter defaults to cancel when destructive."""
    answer = _standard_dialog(
        level or ("remove" if destructive else "question"), parent, title, text,
        StandardButton.Yes | StandardButton.No, StandardButton.No if destructive else StandardButton.Yes,
        kicker=kicker, detail=detail, subject=subject, destructive=destructive,
        labels={StandardButton.Yes: confirm_label, StandardButton.No: cancel_label or tr("Cancel")},
    )
    return answer == StandardButton.Yes


Yes, No, Ok, Cancel = StandardButton.Yes, StandardButton.No, StandardButton.Ok, StandardButton.Cancel

__all__ = ["HubDialog", "Cancel", "No", "Ok", "Yes", "confirm", "critical", "information", "question", "warning"]
