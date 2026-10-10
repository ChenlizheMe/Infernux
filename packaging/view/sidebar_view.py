"""Left sidebar navigation for Infernux Hub."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QApplication,
    QGraphicsOpacityEffect, QComboBox, QLineEdit, QFrame,
)
from PySide6.QtCore import Signal, Qt, QPropertyAnimation, Property, QEasingCurve, QRectF, QSize
from PySide6.QtGui import QPainter, QColor, QPaintEvent, QPixmap, QPen

from style import StyleManager
from i18n import tr
from hub_resources import ICON_PATH
import time

from view.forge import (
    HazardStripe, IgnitionLogo, MissionClock, StatusLed, mix, mono_font, mono_label, qcolor, theme, track,
)


def apply_theme(window, is_dark: bool) -> None:
    """Apply the one global Hub theme used by the Settings page."""
    app = QApplication.instance()
    if app is None or getattr(app, "is_dark_theme", True) == is_dark:
        return
    pixmap = window.grab()
    overlay = QLabel(window)
    overlay.setPixmap(pixmap)
    overlay.setGeometry(window.rect())
    overlay.show()
    app.is_dark_theme = is_dark
    app.setStyleSheet(StyleManager.get_stylesheet(is_dark))
    for widget_type in (QPushButton, QComboBox, QLineEdit):
        for widget in window.findChildren(widget_type):
            if hasattr(widget, "_hub_hover_progress"):
                widget._hub_hover_progress = 0.0
                widget.setStyleSheet("")
    app.processEvents()
    effect = QGraphicsOpacityEffect(overlay)
    overlay.setGraphicsEffect(effect)
    animation = QPropertyAnimation(effect, b"opacity", overlay)
    animation.setDuration(180)
    animation.setStartValue(1.0)
    animation.setEndValue(0.0)
    animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
    animation.finished.connect(overlay.deleteLater)
    animation.start()
    window._theme_anim = animation


class ToggleSwitch(QWidget):
    """Hardware rocker: a square thumb travels across a recessed slot."""
    stateChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(52, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._checked = True
        self._position = 1.0
        self._anim = QPropertyAnimation(self, b"position")
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.setDuration(180)

    @Property(float)
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos
        self.update()

    def isChecked(self):
        return self._checked

    def setChecked(self, checked: bool):
        self._checked = checked
        self._position = 1.0 if checked else 0.0
        self.update()

    def _toggle(self):
        self._checked = not self._checked
        self._anim.stop()
        self._anim.setStartValue(self._position)
        self._anim.setEndValue(1.0 if self._checked else 0.0)
        self._anim.start()
        self.stateChanged.emit(int(self._checked))

    def mousePressEvent(self, ev):
        self._toggle()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._toggle()
        else:
            super().keyPressEvent(ev)

    def paintEvent(self, e: QPaintEvent):
        palette = theme()
        p = QPainter(self)
        rect = self.rect()
        slot = mix(QColor(palette.bg_input), QColor(palette.accent_deep), self._position)
        p.fillRect(rect, slot)
        border = QColor(palette.accent if self.hasFocus() else palette.border_hover)
        p.setPen(QPen(border, 1))
        p.drawRect(rect.adjusted(0, 0, -1, -1))
        p.setFont(mono_font(8, tracking=0.5))
        p.setPen(qcolor(palette.text_muted, 200))
        p.drawText(QRectF(4, 0, 22, rect.height()), Qt.AlignmentFlag.AlignCenter, "I")
        p.drawText(QRectF(rect.width() - 26, 0, 22, rect.height()), Qt.AlignmentFlag.AlignCenter, "O")
        thumb_w = 22
        x = 2 + (rect.width() - thumb_w - 4) * self._position
        thumb = mix(QColor(palette.text_muted), QColor(palette.accent_fill), self._position)
        p.fillRect(QRectF(x, 2, thumb_w, rect.height() - 4), thumb)
        grip = qcolor(palette.accent_text if self._checked else palette.bg_base, 150)
        for offset in (7, 11, 15):
            p.fillRect(QRectF(x + offset, 7, 1, rect.height() - 14), grip)
        p.end()


class NavButton(QPushButton):
    """Numbered navigation plate with an indicator lamp and active bar."""

    def __init__(self, index: int, label: str, parent=None):
        super().__init__(label, parent)
        self._number = f"{index:02d}"
        self.setObjectName("navItem")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(44)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._badge = ""

    def set_badge(self, text: str) -> None:
        if text != self._badge:
            self._badge = text
            self.update()

    def paintEvent(self, _event):
        palette = theme()
        active = bool(self.property("active"))
        hover = float(getattr(self, "_hub_hover_progress", 0.0))
        p = QPainter(self)
        rect = self.rect()
        background = QColor(palette.nav_active) if active else mix(
            qcolor(palette.nav_hover, 0), QColor(palette.nav_hover), hover)
        p.fillRect(rect, background)
        if active:
            p.fillRect(0, 0, 3, rect.height(), QColor(palette.accent_fill))
        elif hover > 0.01:
            p.fillRect(0, rect.height() // 2 - 6, 2, 12, qcolor(palette.accent_fill, round(200 * hover)))
        p.setFont(mono_font(10, tracking=0.6))
        p.setPen(QColor(palette.accent if active else palette.text_muted))
        p.drawText(QRectF(20, 0, 26, rect.height()), Qt.AlignmentFlag.AlignVCenter, self._number)
        label_font = self.font()
        label_font.setPixelSize(14)
        label_font.setWeight(label_font.Weight.DemiBold if active else label_font.Weight.Medium)
        p.setFont(label_font)
        text = mix(QColor(palette.text_secondary), QColor(palette.text_primary), 1.0 if active else hover)
        p.setPen(text)
        p.drawText(QRectF(50, 0, rect.width() - 90, rect.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        if self._badge:
            p.setFont(mono_font(10, tracking=0.4))
            metrics = p.fontMetrics()
            width = metrics.horizontalAdvance(self._badge) + 10
            badge = QRectF(rect.width() - width - 14, rect.height() / 2 - 9, width, 18)
            p.fillRect(badge, QColor(palette.accent_fill))
            p.setPen(QColor(palette.accent_text))
            p.drawText(badge, Qt.AlignmentFlag.AlignCenter, self._badge)
        elif active:
            p.fillRect(rect.width() - 20, rect.height() // 2 - 3, 6, 6, QColor(palette.accent))
        if self.hasFocus():
            p.setPen(QPen(QColor(palette.accent), 1, Qt.PenStyle.DotLine))
            p.drawRect(rect.adjusted(1, 1, -2, -2))
        p.end()


class SidebarView(QWidget):
    """Infernux Hub rail: brand plate, numbered navigation and telemetry."""

    page_changed = Signal(int)
    update_requested = Signal()
    ignited = Signal()

    def __init__(self, parent=None, *, started: float | None = None):
        super().__init__(parent)
        self.setFixedWidth(248)
        self.setObjectName("sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title_container = QWidget()
        title_container.setObjectName("sidebarHeader")
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(18, 20, 18, 16)
        title_layout.setSpacing(12)
        self.logo = IgnitionLogo(QPixmap(ICON_PATH), size=42)
        self.logo.setToolTip("Infernux Hub")
        self.logo.ignited.connect(self.ignited)
        title_layout.addWidget(self.logo)

        brand_text = QVBoxLayout()
        brand_text.setSpacing(2)
        title = QLabel("INFERNUX")
        title.setObjectName("sidebarTitle")
        track(title, 3.0)
        brand_text.addWidget(title)
        self._subtitle = mono_label(tr("HUB") + "  ·  " + _hub_version(), "sidebarSubtitle", spacing=1.4)
        brand_text.addWidget(self._subtitle)
        title_layout.addLayout(brand_text, 1)
        layout.addWidget(title_container)
        self.stripe = HazardStripe(6)
        self.stripe.setToolTip(tr("Stripes move while downloads or installs run"))
        layout.addWidget(self.stripe)
        layout.addSpacing(18)

        layout.addWidget(mono_label(tr("NAVIGATION"), "sidebarSection", spacing=2.0))
        layout.addSpacing(6)

        # Button order is the page index order; Settings is pinned to the bottom.
        self._nav_buttons: list[QPushButton] = []
        entries = [(tr("Projects"), 0, 1), (tr("Installs"), 1, 2), (tr("Settings"), 2, 4), (tr("Community"), 3, 3)]
        for label, index, number in entries:
            btn = NavButton(number, label)
            btn.setProperty("active", index == 0)
            btn.setProperty("page", index)
            btn.clicked.connect(lambda _checked, i=index: self._switch_page(i))
            self._nav_buttons.append(btn)
        for btn in (self._nav_buttons[0], self._nav_buttons[1], self._nav_buttons[3]):
            layout.addWidget(btn)
        layout.addStretch()

        self.update_pill = QPushButton()
        self.update_pill.setObjectName("updatePill")
        self.update_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update_pill.clicked.connect(self.update_requested)
        self.update_pill.hide()
        pill_row = QHBoxLayout()
        pill_row.setContentsMargins(16, 0, 16, 10)
        pill_row.addWidget(self.update_pill)
        layout.addLayout(pill_row)

        layout.addWidget(self._nav_buttons[2])
        layout.addSpacing(10)

        telemetry = QWidget()
        telemetry_layout = QVBoxLayout(telemetry)
        telemetry_layout.setContentsMargins(20, 12, 18, 18)
        telemetry_layout.setSpacing(6)
        rule = QFrame()
        rule.setFixedHeight(1)
        rule.setObjectName("telemetryRule")
        telemetry_layout.addWidget(rule)
        telemetry_layout.addSpacing(2)
        self.clock = MissionClock(started if started is not None else time.monotonic())
        self.clock.setToolTip(tr("Mission elapsed time since Hub started"))
        telemetry_layout.addWidget(self.clock)
        self._telemetry_rows: dict[str, tuple[StatusLed, QLabel]] = {}
        for key, text in (("engines", tr("ENGINES") + "  --"), ("runtime", tr("RUNTIME") + "  --"),
                          ("network", tr("CATALOG") + "  --")):
            row = QHBoxLayout()
            row.setSpacing(8)
            led = StatusLed("idle")
            label = mono_label(text, "telemetryLine", spacing=0.8)
            row.addWidget(led, 0, Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(label, 1)
            telemetry_layout.addLayout(row)
            self._telemetry_rows[key] = (led, label)
        layout.addWidget(telemetry)

    def set_telemetry(self, key: str, text: str, kind: str = "idle") -> None:
        led, label = self._telemetry_rows[key]
        led.set_kind(kind)
        if label.text() != text:
            label.setText(text)

    def set_badge(self, page: int, text: str) -> None:
        for button in self._nav_buttons:
            if button.property("page") == page:
                button.set_badge(text)

    def set_transfer_active(self, active: bool) -> None:
        self.stripe.set_active(active)

    def show_update(self, version: str | None) -> None:
        if version:
            self.update_pill.setText(tr("UPDATE  ·  HUB {version}", version=version))
            self.update_pill.setToolTip(tr("Review and install the Hub update"))
        self.update_pill.setVisible(bool(version))

    def paintEvent(self, event):
        super().paintEvent(event)
        palette = theme()
        p = QPainter(self)
        # Registration ticks along the rail's right edge.
        tick = qcolor(palette.text_muted, 70)
        for y in range(0, self.height(), 24):
            p.fillRect(self.width() - 5, y, 4 if y % 96 == 0 else 2, 1, tick)
        p.end()

    # ── Internal ─────────────────────────────────────────────────────

    def _switch_page(self, index: int):
        for btn in self._nav_buttons:
            animation = getattr(btn, "_hub_hover_animation", None)
            if animation is not None:
                animation.stop()
            btn._hub_hover_progress = 0.0
            btn.setStyleSheet("")
            btn.setProperty("active", btn.property("page") == index)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()
        self.page_changed.emit(index)

    def select_page(self, index: int) -> None:
        self._switch_page(index)

    def _toggle_theme(self, state):
        apply_theme(self.window(), bool(state))


def _hub_version() -> str:
    try:
        from hub_updater import current_hub_version
        return current_hub_version()
    except Exception:
        return "DEV"
