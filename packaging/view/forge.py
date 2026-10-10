"""FORGE SIGNAL primitives: painted, asset-free building blocks for the Hub.

Everything is drawn with QPainter in the Hub's one typeface, so the Hub ships
no images beyond its icon. Animated pieces only run timers while visible.
"""

from __future__ import annotations

import math
import random
import time
import zlib

from PySide6.QtCore import (
    Property, QEasingCurve, QLineF, QPoint, QPointF, QPropertyAnimation, QRect, QRectF,
    QSize, Qt, QTimer, Signal,
)
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget,
)

from i18n import tr
from style import FONT_FAMILIES, StyleManager, ThemePalette


def is_dark() -> bool:
    return bool(getattr(QApplication.instance(), "is_dark_theme", True))


def theme() -> ThemePalette:
    return StyleManager.palette(is_dark())


def qcolor(value: str, alpha: int | None = None) -> QColor:
    color = QColor(value)
    if alpha is not None:
        color.setAlpha(alpha)
    return color


def mix(start: QColor, end: QColor, amount: float) -> QColor:
    amount = max(0.0, min(1.0, amount))
    return QColor(
        round(start.red() + (end.red() - start.red()) * amount),
        round(start.green() + (end.green() - start.green()) * amount),
        round(start.blue() + (end.blue() - start.blue()) * amount),
        round(start.alpha() + (end.alpha() - start.alpha()) * amount),
    )


# ── Type: one family, hierarchy from size, weight and tracking ───────

def ui_font(pixel_size: int = 13, *, weight: QFont.Weight = QFont.Weight.Medium, tracking: float = 0.0) -> QFont:
    font = QFont()
    font.setFamilies(list(FONT_FAMILIES))
    font.setPixelSize(pixel_size)
    font.setWeight(weight)
    if tracking:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, tracking)
    return font


def caps_font(pixel_size: int = 11, *, tracking: float = 1.4) -> QFont:
    """Tracked capitals for labels and readouts."""
    return ui_font(pixel_size, weight=QFont.Weight.Bold, tracking=tracking)


# Compatibility names used across the views.
def mono_font(pixel_size: int = 11, *, bold: bool = True, tracking: float = 1.0) -> QFont:
    return ui_font(pixel_size, weight=QFont.Weight.Bold if bold else QFont.Weight.Medium, tracking=tracking)


def display_font(pixel_size: int = 16, *, bold: bool = True) -> QFont:
    return ui_font(pixel_size, weight=QFont.Weight.Bold if bold else QFont.Weight.Medium)


def track(label: QLabel, spacing: float = 1.2) -> QLabel:
    """Letter-space a label; QSS has no letter-spacing property."""
    font = label.font()
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    label.setFont(font)
    return label


def mono_label(text: str, object_name: str = "monoLabel", parent=None, *, spacing: float = 1.0) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName(object_name)
    return track(label, spacing)


caps_label = mono_label


# ── Pixel glyphs (7×7 bitmaps) ───────────────────────────────────────

GLYPHS = {
    "warning": ("..###..", "..###..", "..###..", "..###..", ".......", "..###..", "..###.."),
    "question": (".#####.", "##...##", "....##.", "...##..", "...##..", ".......", "...##.."),
    "info": ("...##..", ".......", "..###..", "...##..", "...##..", "...##..", "..####."),
    "critical": ("##...##", ".##.##.", "..###..", "..###..", ".##.##.", "##...##", "......."),
    "ok": (".......", "......#", ".....##", "#...##.", "##.##..", ".###...", "..#...."),
    "remove": (".......", ".......", ".......", "#######", ".......", ".......", "......."),
    "folder": ("###....", "#.####.", "#.....#", "#.....#", "#.....#", "#######", "......."),
    "play": (".#.....", ".##....", ".###...", ".####..", ".###...", ".##....", ".#....."),
    "star": ("...#...", "...#...", "#######", ".#####.", "..###..", ".##.##.", ".#...#."),
    "update": ("...#...", "..###..", ".#####.", "...#...", "...#...", "...#...", "......."),
}


def level_color(level: str, palette: ThemePalette | None = None) -> QColor:
    palette = palette or theme()
    return QColor({
        "warning": palette.warn, "critical": palette.danger, "ok": palette.signal,
        "info": palette.text_mono, "question": palette.accent, "remove": palette.danger,
    }.get(level, palette.accent))


def draw_glyph(painter: QPainter, rect: QRectF | QRect, glyph: str, color: QColor) -> None:
    rows = GLYPHS.get(glyph, GLYPHS["info"])
    rect = QRectF(rect)
    cell = min(rect.width(), rect.height()) / 7.0
    x0 = rect.left() + (rect.width() - cell * 7) / 2
    y0 = rect.top() + (rect.height() - cell * 7) / 2
    for y, row in enumerate(rows):
        for x, bit in enumerate(row):
            if bit == "#":
                painter.fillRect(QRectF(x0 + x * cell, y0 + y * cell, cell + 0.2, cell + 0.2), color)


def draw_brackets(painter: QPainter, rect: QRectF, color: QColor, length: float = 10.0, width: float = 2.0) -> None:
    """Viewfinder corner brackets, the NASA-punk selection mark."""
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.SquareCap)
    painter.setPen(pen)
    left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
    for x, y, dx, dy in ((left, top, 1, 1), (right, top, -1, 1), (left, bottom, 1, -1), (right, bottom, -1, -1)):
        painter.drawLine(QLineF(x, y, x + dx * length, y))
        painter.drawLine(QLineF(x, y, x, y + dy * length))


class GlyphPlate(QWidget):
    """A square instrument plate carrying one pixel glyph."""

    def __init__(self, glyph: str, level: str = "", parent=None, *, size: int = 44):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._glyph = glyph
        self._level = level or glyph

    def paintEvent(self, _event):
        palette = theme()
        color = level_color(self._level, palette)
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor(palette.bg_input))
        painter.setPen(QPen(color, 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        painter.fillRect(0, 0, rect.width(), 3, color)
        inset = max(4, round(rect.width() * 0.24))
        draw_glyph(painter, rect.adjusted(inset, inset + 2, -inset, -inset + 2), self._glyph, color)
        painter.end()


# ── Formatting ───────────────────────────────────────────────────────

def fmt_bytes(value: float) -> str:
    amount = float(max(0.0, value))
    for unit in ("B", "KB", "MB", "GB"):
        if amount < 1024.0 or unit == "GB":
            return f"{int(amount)} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} GB"


def fmt_rate(bytes_per_second: float) -> str:
    return f"{fmt_bytes(bytes_per_second)}/s" if bytes_per_second > 1 else ""


def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def fmt_age(timestamp: float | None, now: float | None = None) -> str:
    if not timestamp:
        return tr("never")
    delta = max(0, int((now or time.time()) - timestamp))
    if delta < 45:
        return tr("just now")
    if delta < 3600:
        return tr("{count} min ago", count=max(1, round(delta / 60)))
    if delta < 86400:
        return tr("{count} h ago", count=round(delta / 3600))
    return tr("{count} d ago", count=round(delta / 86400))


def fmt_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ── Backdrop ─────────────────────────────────────────────────────────

_TILES: dict[tuple[bool, str], QPixmap] = {}


def backdrop_tile(palette: ThemePalette, dark: bool) -> QPixmap:
    key = (dark, palette.bg_base)
    tile = _TILES.get(key)
    if tile is not None:
        return tile
    size = 96
    tile = QPixmap(size, size)
    tile.fill(QColor(palette.bg_base))
    painter = QPainter(tile)
    grid = QColor(palette.grid)
    grid.setAlpha(5 if dark else 7)
    for y in range(0, size, 3):
        painter.fillRect(0, y, size, 1, grid)
    grid.setAlpha(10 if dark else 14)
    painter.fillRect(0, 0, size, 1, grid)
    painter.fillRect(0, 0, 1, size, grid)
    grid.setAlpha(6 if dark else 9)
    painter.fillRect(0, 48, size, 1, grid)
    painter.fillRect(48, 0, 1, size, grid)
    grid.setAlpha(46 if dark else 60)
    painter.fillRect(0, 0, 2, 2, grid)
    rng = random.Random(0x1F)
    for _ in range(340):
        grid.setAlpha(rng.randint(4, 13) if dark else rng.randint(5, 16))
        painter.fillRect(rng.randrange(size), rng.randrange(size), 1, 1, grid)
    painter.end()
    _TILES[key] = tile
    return tile


class Backdrop(QWidget):
    """Window background: grid, scanlines and grain from one cached tile."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def paintEvent(self, event):
        palette, dark = theme(), is_dark()
        painter = QPainter(self)
        painter.drawTiledPixmap(event.rect(), backdrop_tile(palette, dark), event.rect().topLeft())
        painter.end()


# ── Decorative strips ────────────────────────────────────────────────

class HazardStripe(QWidget):
    """Diagonal hazard stripes. While ``active`` they scroll like a conveyor."""

    def __init__(self, height: int = 6, parent=None, *, width: int | None = None, color: str = ""):
        super().__init__(parent)
        if width is not None:
            self.setFixedSize(width, height)
        else:
            self.setFixedHeight(height)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._offset = 0
        self._color = color
        self._timer = QTimer(self)
        self._timer.setInterval(45)
        self._timer.timeout.connect(self._advance)
        self._active = False

    def set_color(self, color: str) -> None:
        self._color = color
        self.update()

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self._sync()

    def _sync(self):
        if self._active and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()

    def showEvent(self, event):
        self._sync()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _advance(self):
        self._offset = (self._offset + 1) % max(8, self.height() * 2)
        self.update()

    def paintEvent(self, _event):
        palette = theme()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(palette.bg_deep))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._color or palette.accent_fill))
        h = self.height()
        step = max(8, h * 2)
        for x in range(-step * 2 + self._offset, self.width() + step, step):
            painter.drawPolygon(QPolygonF([
                QPointF(x, h), QPointF(x + h, 0), QPointF(x + h + step / 2, 0), QPointF(x + step / 2, h),
            ]))
        painter.end()


class TickRule(QWidget):
    """A measured rule: minor ticks every 8 px, majors every 64 px."""

    def __init__(self, parent=None, *, accent_length: int = 72):
        super().__init__(parent)
        self._accent_length = accent_length
        self.setFixedHeight(9)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, _event):
        palette = theme()
        painter = QPainter(self)
        base = self.height() - 1
        line = qcolor(palette.text_muted, 110)
        painter.fillRect(0, base, self.width(), 1, line)
        for x in range(0, self.width(), 8):
            major = x % 64 == 0
            painter.fillRect(x, base - (5 if major else 2), 1, 5 if major else 2, line)
        painter.fillRect(0, base - 1, self._accent_length, 2, QColor(palette.accent_fill))
        painter.end()


class StatusLed(QWidget):
    """Square indicator lamp. ``busy`` blinks while visible."""

    def __init__(self, kind: str = "idle", parent=None, *, size: int = 8):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._kind = kind
        self._lit = True
        self._timer = QTimer(self)
        self._timer.setInterval(520)
        self._timer.timeout.connect(self._blink)

    def kind(self) -> str:
        return self._kind

    def set_kind(self, kind: str) -> None:
        if kind == self._kind:
            return
        self._kind = kind
        self._lit = True
        self._sync_timer()
        self.update()

    def _sync_timer(self):
        if self._kind == "busy" and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()

    def showEvent(self, event):
        self._sync_timer()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _blink(self):
        self._lit = not self._lit
        self.update()

    def paintEvent(self, _event):
        palette = theme()
        colors = {
            "ok": palette.signal, "warn": palette.warn, "error": palette.danger,
            "busy": palette.accent, "active": palette.accent, "idle": palette.text_muted,
        }
        color = QColor(colors.get(self._kind, palette.text_muted))
        if not self._lit or self._kind == "idle":
            color.setAlpha(90)
        painter = QPainter(self)
        painter.fillRect(self.rect(), color)
        if self._lit and self._kind != "idle":
            painter.fillRect(1, 1, 2, 2, qcolor("#ffffff", 120))
        painter.end()


class SegmentMeter(QWidget):
    """Cassette-deck level meter used as a progress bar.

    ``set_fraction(None)`` shows an indeterminate sweeping block.
    """

    SEGMENT = 5
    GAP = 2

    def __init__(self, parent=None, *, height: int = 8):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._fraction: float | None = 0.0
        self._phase = 0
        self._tone = "accent"
        self._timer = QTimer(self)
        self._timer.setInterval(70)
        self._timer.timeout.connect(self._advance)

    def fraction(self) -> float | None:
        return self._fraction

    def set_tone(self, tone: str) -> None:
        if tone != self._tone:
            self._tone = tone
            self.update()

    def set_fraction(self, fraction: float | None) -> None:
        fraction = None if fraction is None else max(0.0, min(1.0, float(fraction)))
        if fraction == self._fraction:
            return
        self._fraction = fraction
        self._sync_timer()
        self.update()

    def _sync_timer(self):
        if self._fraction is None and self.isVisible():
            self._timer.start()
        else:
            self._timer.stop()

    def showEvent(self, event):
        self._sync_timer()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _advance(self):
        self._phase += 1
        self.update()

    def paintEvent(self, _event):
        palette = theme()
        painter = QPainter(self)
        pitch = self.SEGMENT + self.GAP
        count = max(1, (self.width() + self.GAP) // pitch)
        off = qcolor(palette.text_muted, 38)
        tone = {"accent": palette.accent_fill, "ok": palette.signal, "warn": palette.warn,
                "error": palette.danger}.get(self._tone, palette.accent_fill)
        on, hot = QColor(tone), QColor(palette.accent if self._tone == "accent" else tone)
        if self._fraction is None:
            head = self._phase % (count + 6)
            lit = set(range(head - 6, head))
        else:
            lit = set(range(0, round(self._fraction * count)))
        last = max(lit) if lit else -1
        for index in range(count):
            color = (hot if index == last else on) if index in lit else off
            painter.fillRect(index * pitch, 0, self.SEGMENT, self.height(), color)
        painter.end()


def chip(text: str, kind: str = "", parent=None) -> QLabel:
    label = QLabel(text, parent)
    label.setObjectName("chip")
    if kind:
        label.setProperty("kind", kind)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    return label


def set_kind(widget: QWidget, kind: str) -> None:
    """Change a QSS ``kind`` property and repolish only when it changed."""
    if widget.property("kind") == kind:
        return
    widget.setProperty("kind", kind)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


class Monogram(QWidget):
    """Project index tile: two letters on a tape-label plate."""

    def __init__(self, name: str, parent=None, *, size: int = 40):
        super().__init__(parent)
        self.setFixedSize(size, size)
        words = [part for part in name.replace("_", " ").replace("-", " ").split() if part]
        letters = "".join(word[0] for word in words[:2]) or name[:2] or "?"
        if len(letters) == 1 and len(name) > 1:
            letters = name[:2]
        self._letters = letters.upper()
        self._variant = zlib.crc32(name.encode("utf-8")) % 4
        self._dim = False

    def set_dim(self, dim: bool) -> None:
        self._dim = dim
        self.update()

    def paintEvent(self, _event):
        palette = theme()
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor(palette.chip))
        painter.setPen(QPen(QColor(palette.border_hover), 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        accent = QColor(palette.text_muted if self._dim else palette.accent_fill)
        notch = (rect.width() // 4) * (self._variant + 1) // 2
        painter.fillRect(0, 0, max(6, notch), 3, accent)
        painter.fillRect(rect.width() - 4, rect.height() - 4, 2, 2, qcolor(palette.text_muted, 160))
        painter.setPen(QColor(palette.disabled_text if self._dim else palette.text_primary))
        painter.setFont(caps_font(max(11, rect.height() // 3), tracking=0.5))
        painter.drawText(rect.adjusted(0, 2, 0, 0), Qt.AlignmentFlag.AlignCenter, self._letters)
        painter.end()


class PageHeader(QWidget):
    """Swiss page plate: §-numbered kicker, display title, measured rule."""

    def __init__(self, index: str, kicker: str, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.kicker_label = mono_label(f"§{index}  /  {kicker}", "pageKicker", spacing=1.8)
        layout.addWidget(self.kicker_label)
        row = QHBoxLayout()
        row.setSpacing(10)
        text = QVBoxLayout()
        text.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("pageTitle")
        text.addWidget(self.title_label)
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("pageSubtitle")
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setVisible(bool(subtitle))
        text.addWidget(self.subtitle_label)
        row.addLayout(text, 1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        row.addLayout(self.actions)
        row.setAlignment(self.actions, Qt.AlignmentFlag.AlignBottom)
        layout.addLayout(row)
        layout.addSpacing(8)
        layout.addWidget(TickRule())


# ── Dropdown ─────────────────────────────────────────────────────────

META_ROLE = int(Qt.ItemDataRole.UserRole) + 41
TAG_ROLE = int(Qt.ItemDataRole.UserRole) + 42


class _ComboDelegate(QStyledItemDelegate):
    """Rows for ForgeComboBox popups: tag chip, label, right-aligned meta."""

    ROW_HEIGHT = 34

    def __init__(self, combo: QComboBox):
        super().__init__(combo)
        self._combo = combo

    def sizeHint(self, option, index):
        return QSize(max(option.rect.width(), 120), self.ROW_HEIGHT)

    def paint(self, painter, option, index):
        palette = theme()
        rect = option.rect
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        current = index.row() == self._combo.currentIndex()
        enabled = bool(index.flags() & Qt.ItemFlag.ItemIsEnabled)
        painter.save()
        painter.fillRect(rect, QColor(palette.nav_active if (hovered or selected) and enabled else palette.bg_surface))
        if (hovered or selected) and enabled:
            painter.fillRect(rect.left(), rect.top(), 2, rect.height(), QColor(palette.accent_fill))
        x = rect.left() + 14
        if current:
            painter.fillRect(x, rect.center().y() - 3, 6, 6, QColor(palette.accent))
        x += 16
        tag = index.data(TAG_ROLE)
        if tag:
            font = caps_font(9, tracking=0.8)
            painter.setFont(font)
            width = QFontMetrics(font).horizontalAdvance(str(tag)) + 10
            chip_rect = QRect(x, rect.center().y() - 8, width, 16)
            painter.fillRect(chip_rect, QColor(palette.label_paper))
            painter.setPen(QColor(palette.label_ink))
            painter.drawText(chip_rect, Qt.AlignmentFlag.AlignCenter, str(tag))
            x += width + 8
        meta = index.data(META_ROLE) or ""
        meta_width = 0
        if meta:
            meta_font = ui_font(11, weight=QFont.Weight.Medium, tracking=0.4)
            meta_width = QFontMetrics(meta_font).horizontalAdvance(str(meta)) + 14
            painter.setFont(meta_font)
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(QRect(rect.right() - meta_width - 6, rect.top(), meta_width, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, str(meta))
        painter.setFont(ui_font(13, weight=QFont.Weight.Bold if current else QFont.Weight.Medium))
        painter.setPen(QColor(palette.text_primary if enabled else palette.disabled_text))
        text_rect = QRect(x, rect.top(), rect.right() - x - meta_width - 10, rect.height())
        text = QFontMetrics(painter.font()).elidedText(str(index.data(Qt.ItemDataRole.DisplayRole) or ""),
                                                       Qt.TextElideMode.ElideRight, text_rect.width())
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)
        painter.restore()


class ForgeComboBox(QComboBox):
    """Dropdown with a painted chevron, styled popup rows and optional meta text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(_ComboDelegate(self))
        self.setMaxVisibleItems(10)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(34)
        self.view().setMouseTracking(True)
        self.view().setTextElideMode(Qt.TextElideMode.ElideRight)
        popup = self.view().window()
        popup.setWindowFlags(popup.windowFlags() | Qt.WindowType.FramelessWindowHint
                             | Qt.WindowType.NoDropShadowWindowHint)
        self._open = 0.0
        self._anim = QPropertyAnimation(self, b"openProgress", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    @Property(float)
    def openProgress(self):
        return self._open

    @openProgress.setter
    def openProgress(self, value):
        self._open = float(value)
        self.update()

    def addItem(self, text, userData=None, *, meta: str = "", tag: str = ""):  # noqa: N803 - Qt naming
        super().addItem(text, userData)
        index = self.count() - 1
        if meta:
            self.setItemData(index, meta, META_ROLE)
        if tag:
            self.setItemData(index, tag, TAG_ROLE)

    def _animate(self, target: float):
        self._anim.stop()
        self._anim.setStartValue(self._open)
        self._anim.setEndValue(target)
        self._anim.start()

    def showPopup(self):
        self._animate(1.0)
        super().showPopup()
        popup = self.view().window()
        popup.setMinimumWidth(self.width())

    def hidePopup(self):
        self._animate(0.0)
        super().hidePopup()

    def paintEvent(self, event):
        super().paintEvent(event)
        palette = theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        index = self.currentIndex()
        meta = self.itemData(index, META_ROLE) if index >= 0 else None
        tag = self.itemData(index, TAG_ROLE) if index >= 0 else None
        right = self.width() - 30
        if meta or tag:
            label = " · ".join(str(part) for part in (tag, meta) if part)
            font = ui_font(11, weight=QFont.Weight.Medium, tracking=0.4)
            metrics = QFontMetrics(font)
            text_width = QFontMetrics(self.font()).horizontalAdvance(self.currentText()) + 12
            available = right - 10 - text_width
            if available > 40:
                painter.setFont(font)
                painter.setPen(QColor(palette.text_muted))
                elided = metrics.elidedText(label, Qt.TextElideMode.ElideLeft, available)
                painter.drawText(QRect(right - available - 8, 0, available, self.height()),
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, elided)
        color = QColor(palette.accent if (self.hasFocus() or self._open > 0.01) else palette.text_secondary)
        if not self.isEnabled():
            color = QColor(palette.disabled_text)
        pen = QPen(color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.SquareCap)
        painter.setPen(pen)
        cx, cy = self.width() - 15, self.height() / 2
        flip = 1 - 2 * self._open
        painter.drawPolyline(QPolygonF([
            QPointF(cx - 4.5, cy - 2 * flip), QPointF(cx, cy + 2.5 * flip), QPointF(cx + 4.5, cy - 2 * flip),
        ]))
        painter.end()


# ── Toasts ───────────────────────────────────────────────────────────

class Toast(QFrame):
    """Transient status plate with an optional action and a draining timer bar."""

    closed = Signal()

    def __init__(self, text: str, kind: str = "ok", parent=None, *, action: tuple[str, object] | None = None,
                 timeout_ms: int = 4600):
        super().__init__(parent)
        self.setObjectName("toast")
        self._kind = kind
        self._timeout = timeout_ms
        self._elapsed = 0.0
        self._last = time.monotonic()
        self.setFixedWidth(380)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 11, 8, 13)
        layout.setSpacing(12)
        glyph = {"ok": "ok", "warn": "warning", "error": "critical", "info": "info"}.get(kind, "info")
        layout.addWidget(GlyphPlate(glyph, {"warn": "warning", "error": "critical"}.get(kind, kind), size=26))
        label = QLabel(text)
        label.setObjectName("toastText")
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        if action is not None:
            action_label, callback = action
            button = QPushButton(action_label)
            button.setObjectName("ghostBtn")
            button.setFixedHeight(28)
            button.clicked.connect(lambda: (callback(), self.dismiss()))
            layout.addWidget(button)
        close = QPushButton("×")
        close.setObjectName("iconBtn")
        close.setFixedSize(26, 26)
        close.setToolTip(tr("Dismiss"))
        close.clicked.connect(self.dismiss)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        self._tick = QTimer(self)
        self._tick.setInterval(60)
        self._tick.timeout.connect(self._step)
        self._hover = False
        self._dismissing = False
        self.setMouseTracking(True)

    def start(self):
        self._elapsed = 0.0
        self._last = time.monotonic()
        self._tick.start()

    def enterEvent(self, event):
        self._hover = True
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        super().leaveEvent(event)

    def _step(self):
        now = time.monotonic()
        if not self._hover:
            # Hovering pauses the countdown so the action stays reachable.
            self._elapsed += now - self._last
        self._last = now
        if self._elapsed * 1000 >= self._timeout:
            self.dismiss()
        self.update()

    def dismiss(self):
        if self._dismissing:
            return
        self._dismissing = True
        self._tick.stop()
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(160)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.finished.connect(self._finish)
        animation.start()
        self._fade = animation

    def _finish(self):
        self.hide()
        self.closed.emit()
        self.deleteLater()

    def paintEvent(self, event):
        palette = theme()
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect, QColor(palette.bg_surface))
        painter.setPen(QPen(QColor(palette.border_hover), 1))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        color = level_color({"warn": "warning", "error": "critical"}.get(self._kind, self._kind), palette)
        painter.fillRect(0, 0, 3, rect.height(), color)
        remaining = 1.0 - min(1.0, self._elapsed * 1000 / max(1, self._timeout))
        painter.fillRect(3, rect.height() - 2, round((rect.width() - 3) * remaining), 2, qcolor(palette.accent_fill, 180))
        painter.end()
        super().paintEvent(event)


class ToastHost(QWidget):
    """Stacks toasts in the bottom-left corner of its parent."""

    MARGIN = 20
    BOTTOM = 52

    def __init__(self, parent: QWidget, *, left_inset: int = 0):
        super().__init__(parent)
        self._left_inset = left_inset
        self._toasts: list[Toast] = []
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()

    def show_toast(self, text: str, kind: str = "ok", *, action=None, timeout_ms: int = 4600) -> Toast:
        toast = Toast(text, kind, self.parentWidget(), action=action, timeout_ms=timeout_ms)
        toast.closed.connect(lambda item=toast: self._remove(item))
        self._toasts.append(toast)
        toast.adjustSize()
        toast.show()
        toast.raise_()
        self.relayout(animate_last=True)
        toast.start()
        while len(self._toasts) > 4:
            self._toasts[0].dismiss()
            self._toasts.pop(0)
        return toast

    def _remove(self, toast: Toast):
        if toast in self._toasts:
            self._toasts.remove(toast)
        self.relayout()

    def relayout(self, *, animate_last: bool = False):
        parent = self.parentWidget()
        y = parent.height() - self.BOTTOM
        for index, toast in enumerate(reversed(self._toasts)):
            toast.adjustSize()
            y -= toast.height()
            target = QPoint(self._left_inset + self.MARGIN, y)
            if animate_last and index == 0:
                toast.move(target + QPoint(0, 18))
                animation = QPropertyAnimation(toast, b"pos", toast)
                animation.setDuration(220)
                animation.setEasingCurve(QEasingCurve.Type.OutCubic)
                animation.setEndValue(target)
                animation.start()
                toast._slide = animation
            else:
                toast.move(target)
            toast.raise_()
            y -= 8


def toast(widget: QWidget | None, text: str, kind: str = "ok", *, action=None, timeout_ms: int = 4600):
    """Show a toast on the Hub window owning ``widget``; a no-op elsewhere."""
    window = getattr(widget, "window", None)
    if not callable(window):
        return None
    host = getattr(window(), "toast_host", None)
    if host is None:
        return None
    return host.show_toast(text, kind, action=action, timeout_ms=timeout_ms)


# ── Instruments ──────────────────────────────────────────────────────

class MissionClock(QLabel):
    """Mission-elapsed time since the Hub started (MET T+hh:mm:ss)."""

    def __init__(self, started: float, parent=None):
        super().__init__(parent)
        self._started = started
        self.setObjectName("telemetryLine")
        track(self, 0.8)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._tick()

    def _tick(self):
        self.setText("MET  T+" + fmt_clock(time.monotonic() - self._started))

    def showEvent(self, event):
        self._timer.start()
        self._tick()
        super().showEvent(event)

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)


class IgnitionLogo(QWidget):
    """The Hub icon on an instrument plate: an orbit ring wakes on hover."""

    ignited = Signal()

    def __init__(self, pixmap: QPixmap, parent=None, *, size: int = 40):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pixmap = pixmap
        self._hover = 0.0
        self._angle = 0
        self._clicks: list[float] = []
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance)
        self._anim = QPropertyAnimation(self, b"hoverProgress", self)
        self._anim.setDuration(240)

    @Property(float)
    def hoverProgress(self):
        return self._hover

    @hoverProgress.setter
    def hoverProgress(self, value):
        self._hover = float(value)
        if self._hover <= 0.001:
            self._timer.stop()
        self.update()

    def enterEvent(self, event):
        self._timer.start()
        self._anim.stop()
        self._anim.setEndValue(1.0)
        self._anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._anim.stop()
        self._anim.setEndValue(0.0)
        self._anim.start()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        now = time.monotonic()
        self._clicks = [stamp for stamp in self._clicks if now - stamp < 1.6] + [now]
        if len(self._clicks) >= 5:
            self._clicks.clear()
            self.ignited.emit()
        super().mousePressEvent(event)

    def _advance(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, _event):
        palette = theme()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        if not self._pixmap.isNull():
            painter.drawPixmap(rect.toRect(), self._pixmap)
        if self._hover > 0.01:
            ring = qcolor(palette.accent, round(200 * self._hover))
            painter.setPen(QPen(ring, 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            outer = QRectF(self.rect()).adjusted(1, 1, -1, -1)
            painter.drawArc(outer, self._angle * 16, 240 * 16)
            painter.setBrush(ring)
            painter.setPen(Qt.PenStyle.NoPen)
            radius = outer.width() / 2
            rad = math.radians(self._angle + 240)
            center = outer.center()
            painter.drawRect(QRectF(center.x() + radius * math.cos(rad) - 1.5,
                                    center.y() - radius * math.sin(rad) - 1.5, 3, 3))
        painter.end()


__all__ = [
    "Backdrop", "ForgeComboBox", "GlyphPlate", "HazardStripe", "IgnitionLogo", "META_ROLE",
    "MissionClock", "Monogram", "PageHeader", "SegmentMeter", "StatusLed", "TAG_ROLE",
    "TickRule", "Toast", "ToastHost", "caps_font", "caps_label", "chip", "display_font",
    "draw_brackets", "draw_glyph", "fmt_age", "fmt_bytes", "fmt_clock", "fmt_eta", "fmt_rate",
    "is_dark", "level_color", "mix", "mono_font", "mono_label", "qcolor", "repolish",
    "set_kind", "theme", "toast", "track", "ui_font",
]
