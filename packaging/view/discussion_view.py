"""Live community page for Infernux Hub."""

from __future__ import annotations

import json

import time

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPainter
import threading
import urllib.request
import weakref

import shiboken6
from itertools import count

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import community_feed
from community_feed import COMMUNITY_ORIGIN, HotTopic, parse_hot_topics
from i18n import tr
from style import StyleManager
from view.hover_widgets import AnimatedSurfaceFrame
from view.forge import HazardStripe, PageHeader, StatusLed, fmt_age, mono_label, qcolor, theme


# The last successful feed survives page rebuilds (e.g. a language switch).
_FEED_CACHE: dict[str, object] = {"topics": None, "at": 0.0}


class _FeedBridge(QObject):
    """Main-thread mailbox for feed workers; it outlives every page.

    Requests use urllib (the same TLS stack as the rest of Hub) on a daemon
    thread instead of QtNetwork, whose TLS backend plugins are not shipped by
    every PySide6 distribution. A page that is closed or rebuilt simply stops
    listening; a late result is then dropped.
    """

    finished = Signal(int, object, str)


_BRIDGE: _FeedBridge | None = None
_REQUEST_IDS = count(1)


def _bridge() -> _FeedBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = _FeedBridge(QApplication.instance())
    return _BRIDGE


def _fetch_feed(request_id: int, url: str, timeout: float, bridge: _FeedBridge) -> None:
    topics, error = None, ""
    try:
        request = urllib.request.Request(
            url, headers={"Accept": "application/json", "User-Agent": "InfernuxHub-Community"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            topics = parse_hot_topics(json.loads(response.read(2 * 1024 * 1024).decode("utf-8")))
    except Exception as exc:  # urllib, http.client, decoding and contract errors
        error = str(exc) or type(exc).__name__
    try:
        bridge.finished.emit(request_id, topics, error)
    except RuntimeError:
        pass  # Application is shutting down.


class DiscussionGlyph(QWidget):
    """Pixel speech-bubble mark drawn natively, keeping the page asset-free."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)

    def paintEvent(self, _event):
        palette = theme()
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        accent = QColor(palette.accent_fill)
        painter.fillRect(4, 6, 8, 52, accent)
        painter.fillRect(4, 6, 64, 8, accent)
        painter.fillRect(60, 6, 8, 44, accent)
        painter.fillRect(20, 42, 48, 8, accent)
        painter.fillRect(12, 50, 8, 16, accent)
        painter.fillRect(20, 24, 8, 8, qcolor(palette.text_primary))
        painter.fillRect(34, 24, 8, 8, qcolor(palette.text_primary))
        painter.fillRect(48, 24, 6, 8, qcolor(palette.text_muted))
        painter.end()


class _TopicRow(AnimatedSurfaceFrame):
    def __init__(self, rank: int, topic: HotTopic, parent=None):
        super().__init__("communityTopic", parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._url = topic.url
        self.setToolTip(topic.url)
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 10, 12, 10)
        row.setSpacing(14)
        rank_label = QLabel(f"{rank:02d}")
        rank_label.setObjectName("topicRank")
        rank_label.setFixedWidth(30)
        row.addWidget(rank_label)
        copy = QVBoxLayout()
        copy.setSpacing(3)
        title = QLabel(topic.title)
        title.setObjectName("communityTopicTitle")
        title.setWordWrap(True)
        copy.addWidget(title)
        stats = QLabel(
            tr(
                "{replies} replies · {views} views · {likes} likes",
                replies=topic.replies,
                views=topic.views,
                likes=topic.likes,
            )
        )
        stats.setObjectName("communityTopicStats")
        copy.addWidget(stats)
        row.addLayout(copy, 1)
        button = QPushButton(tr("Open"))
        button.setObjectName("ghostBtn")
        button.setFixedHeight(30)
        button.clicked.connect(self._open)
        row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._open()

    def _open(self):
        QDesktopServices.openUrl(QUrl(self._url))


class DiscussionView(QWidget):
    """Official community entry plus the live weekly top-topic feed.

    The feed loads when the page is first shown and refreshes every ten
    minutes while visible; current topics stay on screen while reloading.
    """

    FORUM_URL = COMMUNITY_ORIGIN + "/"
    REQUEST_TIMEOUT_MS = 15000
    AUTO_REFRESH_MS = 10 * 60 * 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reply: int | None = None
        self._closing = False
        view_ref = weakref.ref(self)

        def deliver(request_id, topics, error):
            view = view_ref()
            if view is not None and shiboken6.isValid(view):
                view._request_finished(request_id, topics, error)

        self._deliver = deliver
        # Queued: workers emit from their own thread; widgets update on the GUI thread.
        _bridge().finished.connect(deliver, Qt.ConnectionType.QueuedConnection)
        self._request_error = ""
        self._deadline = QTimer(self)
        self._deadline.setSingleShot(True)
        self._deadline.timeout.connect(self._request_expired)
        self._auto_refresh = QTimer(self)
        self._auto_refresh.setInterval(self.AUTO_REFRESH_MS)
        self._auto_refresh.timeout.connect(self.refresh)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 28, 36, 24)
        layout.setSpacing(18)
        layout.addWidget(PageHeader(
            "03", tr("COMMUNITY"), tr("Community"),
            tr("The official Infernux community for support, ideas, and project sharing."),
        ))

        hero = AnimatedSurfaceFrame("discussionHero")
        hero.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        hero_outer = QVBoxLayout(hero)
        hero_outer.setContentsMargins(0, 0, 0, 0)
        hero_outer.setSpacing(0)
        hero_outer.addWidget(HazardStripe(5))
        hero_layout = QHBoxLayout()
        hero_layout.setContentsMargins(28, 24, 28, 24)
        hero_layout.setSpacing(26)
        hero_layout.addWidget(DiscussionGlyph(), 0, Qt.AlignmentFlag.AlignTop)
        copy = QVBoxLayout()
        copy.setSpacing(8)
        eyebrow = mono_label(tr("INFERNUX COMMUNITY"), "discussionEyebrow", spacing=2.0)
        copy.addWidget(eyebrow)
        heading = QLabel(tr("Join the Infernux community."))
        heading.setObjectName("discussionHeading")
        copy.addWidget(heading)
        description = QLabel(
            tr(
                "Ask questions, report bugs, discuss engine workflows, and share projects with other Infernux users."
            )
        )
        description.setObjectName("discussionDescription")
        description.setWordWrap(True)
        copy.addWidget(description)
        footer = QHBoxLayout()
        footer.setSpacing(14)
        enter = QPushButton(tr("Open Community"))
        enter.setObjectName("primaryBtn")
        enter.setCursor(Qt.CursorShape.PointingHandCursor)
        enter.setFixedHeight(36)
        enter.clicked.connect(self._open_forum)
        footer.addWidget(enter)
        address = mono_label("infernux-engine.discourse.group", "discussionAddress", spacing=0.6)
        footer.addWidget(address)
        footer.addStretch()
        copy.addSpacing(6)
        copy.addLayout(footer)
        hero_layout.addLayout(copy, 1)
        hero_outer.addLayout(hero_layout)
        layout.addWidget(hero)

        feed_header = QHBoxLayout()
        feed_header.setSpacing(10)
        feed_title = QLabel(tr("Popular this week"))
        feed_title.setObjectName("communityFeedTitle")
        feed_header.addWidget(feed_title)
        self._feed_led = StatusLed("idle")
        feed_header.addSpacing(6)
        feed_header.addWidget(self._feed_led, 0, Qt.AlignmentFlag.AlignVCenter)
        self._feed_state = mono_label("", "monoMeta", spacing=0.8)
        feed_header.addWidget(self._feed_state)
        feed_header.addStretch()
        self._refresh = QPushButton(tr("Refresh"))
        self._refresh.setObjectName("ghostBtn")
        self._refresh.setFixedHeight(30)
        self._refresh.clicked.connect(self.refresh)
        feed_header.addWidget(self._refresh)
        layout.addLayout(feed_header)

        self._feed = QVBoxLayout()
        self._feed.setSpacing(4)
        layout.addLayout(self._feed)
        layout.addStretch()
        cached = _FEED_CACHE["topics"]
        if cached is not None:
            self._show_topics(cached)
            self._set_state("ok")
        else:
            self._set_feed_message(tr("Loading community topics..."), "loading")
            self._set_state("idle")

    def _set_state(self, kind: str, text: str = "") -> None:
        self._feed_led.set_kind(kind)
        if not text:
            stamp = float(_FEED_CACHE["at"] or 0.0)
            text = tr("UPDATED {age}", age=fmt_age(stamp).upper()) if stamp else ""
        self._feed_state.setText(text)

    def showEvent(self, event):
        super().showEvent(event)
        self._auto_refresh.start()
        stamp = float(_FEED_CACHE["at"] or 0.0)
        if time.time() - stamp > self.AUTO_REFRESH_MS / 1000:
            QTimer.singleShot(0, self.refresh)

    def hideEvent(self, event):
        self._auto_refresh.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        if self._closing or self._reply is not None:
            return
        if _FEED_CACHE["topics"] is None:
            self._clear_feed()
            self._set_feed_message(tr("Loading community topics..."), "loading")
        self._set_state("busy", tr("UPDATING"))
        self._refresh.setEnabled(False)
        self._request_error = ""
        self._reply = next(_REQUEST_IDS)
        threading.Thread(
            target=_fetch_feed,
            args=(self._reply, community_feed.HOT_TOPICS_URL, self.REQUEST_TIMEOUT_MS / 1000, _bridge()),
            name="community-feed",
            daemon=True,
        ).start()
        # An overall deadline also covers a response that trickles indefinitely.
        self._deadline.start(self.REQUEST_TIMEOUT_MS)

    def _request_expired(self) -> None:
        if self._reply is not None:
            # The worker cannot be interrupted; its late result is ignored.
            self._reply = None
            self._show_error(tr("Community request timed out."))
            self._refresh.setEnabled(True)

    def _request_finished(self, request_id: int, topics, error: str) -> None:
        if self._closing or request_id != self._reply:
            return
        self._reply = None
        self._deadline.stop()
        try:
            if error or topics is None:
                self._show_error(error or tr("Community request failed."))
            else:
                _FEED_CACHE["topics"], _FEED_CACHE["at"] = topics, time.time()
                self._show_topics(topics)
                self._set_state("ok")
        finally:
            self._refresh.setEnabled(True)

    def shutdown(self) -> None:
        self._closing = True
        self._deadline.stop()
        self._auto_refresh.stop()
        self._reply = None
        try:
            _bridge().finished.disconnect(self._deliver)
        except (RuntimeError, TypeError):
            pass

    def _clear_feed(self) -> None:
        while self._feed.count():
            item = self._feed.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()

    def _set_feed_message(self, text: str, kind: str) -> None:
        label = QLabel(text)
        label.setObjectName("communityFeedStatus")
        label.setProperty("kind", kind)
        label.setWordWrap(True)
        self._feed.addWidget(label)

    def _show_topics(self, topics: list[HotTopic]) -> None:
        self._clear_feed()
        if not topics:
            self._set_feed_message(tr("No public topics this week."), "empty")
            return
        for rank, topic in enumerate(topics, start=1):
            self._feed.addWidget(_TopicRow(rank, topic))

    def _show_error(self, message: str) -> None:
        cached = _FEED_CACHE["topics"]
        self._clear_feed()
        self._set_feed_message(
            tr("Community topics could not be loaded: {message}", message=message),
            "error",
        )
        if cached:
            for rank, topic in enumerate(cached, start=1):
                self._feed.addWidget(_TopicRow(rank, topic))
        self._set_state("error", tr("OFFLINE"))

    @classmethod
    def _open_forum(cls):
        QDesktopServices.openUrl(QUrl(cls.FORUM_URL))


__all__ = ["DiscussionView"]
