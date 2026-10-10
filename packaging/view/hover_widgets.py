"""Reusable, native Qt hover transitions for Infernux Hub surfaces and controls."""

from __future__ import annotations

from weakref import ref

from PySide6.QtCore import Property, QEasingCurve, QEvent, QObject, QPropertyAnimation, QRectF, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QApplication, QComboBox, QFrame, QLineEdit, QPushButton, QWidget

from style import StyleManager


def _mix(start: QColor, end: QColor, amount: float) -> QColor:
    amount = max(0.0, min(1.0, amount))
    return QColor(
        round(start.red() + (end.red() - start.red()) * amount),
        round(start.green() + (end.green() - start.green()) * amount),
        round(start.blue() + (end.blue() - start.blue()) * amount),
        round(start.alpha() + (end.alpha() - start.alpha()) * amount),
    )


def _hex(color: QColor) -> str:
    return color.name(QColor.NameFormat.HexRgb)


def _rgba(color: QColor) -> str:
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()})"


def _is_dark() -> bool:
    return bool(getattr(QApplication.instance(), "is_dark_theme", True))


class AnimatedSurfaceFrame(QFrame):
    """Flat Hub surface with animated hover gradient and optional inner selection ring."""

    def __init__(self, object_name: str, parent=None):
        super().__init__(parent)
        self.setObjectName(object_name)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._hover_progress = 0.0
        self._selection_progress = 0.0
        self._hover_animation = QPropertyAnimation(self, b"hoverProgress", self)
        self._hover_animation.setDuration(170)
        self._hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._selection_animation = QPropertyAnimation(self, b"selectionProgress", self)
        self._selection_animation.setDuration(210)
        self._selection_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    @Property(float)
    def hoverProgress(self):
        return self._hover_progress

    @hoverProgress.setter
    def hoverProgress(self, value):
        self._hover_progress = float(value)
        self.update()

    @Property(float)
    def selectionProgress(self):
        return self._selection_progress

    @selectionProgress.setter
    def selectionProgress(self, value):
        self._selection_progress = float(value)
        self.update()

    def set_selected_animated(self, selected: bool):
        self._selection_animation.stop()
        self._selection_animation.setStartValue(self._selection_progress)
        self._selection_animation.setEndValue(1.0 if selected else 0.0)
        self._selection_animation.start()

    def enterEvent(self, event):
        self._animate_hover(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_hover(0.0)
        super().leaveEvent(event)

    def _animate_hover(self, target: float):
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._hover_progress)
        self._hover_animation.setEndValue(target)
        self._hover_animation.start()

    def paintEvent(self, event):
        palette = StyleManager.palette(_is_dark())
        hover, selected = self._hover_progress, self._selection_progress
        rect = self.rect()
        painter = QPainter(self)
        fill = _mix(QColor(palette.bg_surface), QColor(palette.bg_surface_hover), hover)
        painter.fillRect(rect, fill)
        if not self.isEnabled() or self.property("unavailable"):
            # Unavailable rows are hatched like a locked-out panel.
            hatch = QColor(palette.grid)
            hatch.setAlpha(14)
            painter.setPen(QPen(hatch, 1))
            for x in range(-rect.height(), rect.width(), 9):
                painter.drawLine(x, rect.height(), x + rect.height(), 0)
        border = _mix(QColor(palette.border), QColor(palette.border_hover), hover)
        painter.setPen(QPen(border, 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        edge = max(hover * 0.75, selected)
        if edge > 0.001:
            accent = QColor(palette.accent_fill)
            accent.setAlpha(round(255 * edge))
            painter.fillRect(0, 0, 2 + round(2 * edge), rect.height(), accent)
        if selected > 0.001:
            accent = QColor(palette.accent)
            accent.setAlpha(round(255 * selected))
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            line = QColor(accent)
            line.setAlpha(round(120 * selected))
            painter.setPen(QPen(line, 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            from view.forge import draw_brackets
            draw_brackets(painter, QRectF(rect.adjusted(1, 1, -2, -2)), accent, 9.0 * selected + 1.0, 2.0)
        painter.end()
        super().paintEvent(event)


_HOVER_FILTER_OBJECT_NAME = "infernuxHubHoverAnimationFilter"


class HoverAnimationFilter(QObject):
    """Animate common Hub controls that cannot transition through QSS alone."""

    def __init__(self, app: QApplication):
        # The application owns the filter.  Keeping this relationship in the
        # QObject tree is important: Python attributes are not a reliable
        # registry when PySide creates a new wrapper for the same QApplication.
        super().__init__(app)
        self.setObjectName(_HOVER_FILTER_OBJECT_NAME)
        self._app_ref = ref(app)
        self._installed = False
        app.aboutToQuit.connect(self._on_application_quit)
        self._install_once()

    def _install_once(self) -> None:
        app = self._app_ref()
        if app is None or self._installed:
            return
        # Qt normally de-duplicates an event filter, but remove first also
        # repairs a filter that was temporarily removed by a test or a host.
        app.removeEventFilter(self)
        app.installEventFilter(self)
        self._installed = True

    def uninstall(self) -> None:
        app = self._app_ref()
        if app is not None and self._installed:
            app.removeEventFilter(self)
        self._installed = False

    def _on_application_quit(self) -> None:
        self.uninstall()

    def eventFilter(self, watched, event):
        if isinstance(watched, (QPushButton, QLineEdit, QComboBox)):
            if event.type() in (QEvent.Type.Enter, QEvent.Type.HoverEnter):
                self._animate(watched, 1.0)
            elif event.type() in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
                self._animate(watched, 0.0)
            elif event.type() in (QEvent.Type.FocusIn, QEvent.Type.FocusOut):
                self._apply(watched, float(getattr(watched, "_hub_hover_progress", 0.0)))
        return super().eventFilter(watched, event)

    def _animate(self, widget: QWidget, target: float):
        # These controls carry state colours that QSS :hover already handles.
        if not widget.isEnabled() or widget.objectName() in {
            "cardAvatar", "dangerSolidBtn", "updatePill", "segmentBtn",
        }:
            return
        animation = getattr(widget, "_hub_hover_animation", None)
        if animation is None:
            animation = QVariantAnimation(widget)
            animation.setDuration(160)
            animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            animation.valueChanged.connect(lambda value, item=widget: self._apply(item, float(value)))
            animation.finished.connect(lambda item=widget: self._finish_animation(item))
            widget._hub_hover_animation = animation
        animation.stop()
        widget._hub_hover_target = target
        animation.setStartValue(float(getattr(widget, "_hub_hover_progress", 0.0)))
        animation.setEndValue(target)
        animation.start()

    @staticmethod
    def _finish_animation(widget: QWidget):
        """Return controls to QSS after a leave transition finishes.

        Keeping an inline background on a navigation item can visually pin its
        hover state even when the numeric animation has already returned to 0.
        Clearing it here restores the transparent inactive and active QSS states.
        """
        if float(getattr(widget, "_hub_hover_target", 1.0)) > 0.0:
            return
        widget._hub_hover_progress = 0.0
        widget.setStyleSheet("")

    def _apply(self, widget: QWidget, progress: float):
        widget._hub_hover_progress = progress
        palette = StyleManager.palette(_is_dark())
        if isinstance(widget, QPushButton):
            name = widget.objectName()
            text_base = QColor(palette.text_primary)
            text_hover = text_base
            if name in {"primaryBtn", "createBtn"}:
                base, hover = QColor(palette.accent_fill), QColor(palette.accent_hover)
                text_base = text_hover = QColor(palette.accent_text)
            elif name in {"dangerBtn", "launchBtn"}:
                base = QColor(palette.bg_surface)
                base.setAlpha(0)
                hover = QColor(palette.accent_fill if name == "launchBtn" else palette.accent_pressed)
                text_base = QColor(palette.accent if name == "launchBtn" else palette.danger)
                text_hover = QColor(palette.accent_text)
            elif name in {"ghostBtn", "iconBtn", "cardOpenBtn", "segmentBtn"}:
                if name == "segmentBtn" and widget.isChecked():
                    return
                base = QColor(palette.button_hover)
                base.setAlpha(0)
                hover = QColor(palette.button_hover)
                text_base = QColor(palette.text_secondary)
                text_hover = QColor(palette.text_primary)
            elif name == "navItem":
                if bool(widget.property("active")):
                    base = QColor(palette.nav_active)
                else:
                    base = QColor(palette.sidebar_bg)
                hover = QColor(palette.nav_hover)
                text_base = QColor(palette.text_secondary)
                text_hover = QColor(palette.text_primary)
            else:
                base = QColor(palette.button_surface)
                hover = QColor(palette.button_hover)
            background = _mix(base, hover, progress)
            widget.setStyleSheet(
                f"background-color: {_rgba(background)};"
                f"color: {_hex(_mix(text_base, text_hover, progress))};"
            )
            return

        base = QColor(palette.bg_input)
        hover = QColor(palette.bg_surface_hover)
        border_base = QColor(palette.border)
        border_hover = QColor(palette.border_hover)
        border = QColor(palette.accent) if widget.hasFocus() else _mix(border_base, border_hover, progress)
        widget.setStyleSheet(
            f"background-color: {_hex(_mix(base, hover, progress))};"
            f"border-color: {_hex(border)};"
        )


def ensure_hover_animation_filter(app: QApplication | None = None) -> HoverAnimationFilter:
    """Return the one hover filter owned by the live Hub QApplication.

    ``QApplication.instance()`` is canonical here.  Accepting a caller's
    wrapper only for validation avoids attaching a filter to a stale or
    foreign application object, which was the source of order-dependent
    tests after another Qt consumer had created and released an app.
    """
    current = QApplication.instance()
    if current is None:
        raise RuntimeError("Hub hover animation requires a live QApplication")
    if app is not None and app is not current:
        # Different Python wrappers can represent the same C++ object.  The
        # QObject child lookup below is deliberately based on the canonical
        # current application rather than Python object identity.
        app = current

    animator = current.findChild(HoverAnimationFilter, _HOVER_FILTER_OBJECT_NAME)
    if animator is None:
        animator = HoverAnimationFilter(current)
    else:
        animator._install_once()

    return animator


def release_hover_animation_filter(app: QApplication | None = None) -> None:
    """Remove the Hub filter without destroying the owning QApplication."""
    current = QApplication.instance()
    if current is None:
        return
    animator = current.findChild(HoverAnimationFilter, _HOVER_FILTER_OBJECT_NAME)
    if animator is not None:
        animator.uninstall()


__all__ = [
    "AnimatedSurfaceFrame",
    "HoverAnimationFilter",
    "ensure_hover_animation_filter",
    "release_hover_animation_filter",
]
