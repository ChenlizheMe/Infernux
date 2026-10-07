"""Live community page for Infernux Hub."""

from __future__ import annotations

import json

from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPainter
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
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


class DiscussionGlyph(QWidget):
    """A small native-drawn mark, keeping the community page asset-free."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        painter.setPen(Qt.PenStyle.NoPen)
        application = QApplication.instance()
        palette = StyleManager.palette(
            bool(getattr(application, "is_dark_theme", True))
        )
        painter.setBrush(QColor(palette.accent))
        painter.drawRect(4, 6, 8, 60)
        painter.drawRect(20, 6, 48, 8)
        painter.drawRect(20, 32, 34, 8)
        painter.drawRect(20, 58, 48, 8)
        painter.drawRect(60, 18, 8, 36)
        painter.end()


class DiscussionView(QWidget):
    """Official community entry plus the live weekly top-topic feed."""

    FORUM_URL = COMMUNITY_ORIGIN + "/"
    REQUEST_TIMEOUT_MS = 15000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._network = QNetworkAccessManager(self)
        self._reply: QNetworkReply | None = None
        self._closing = False
        self._request_error = ""
        self._deadline = QTimer(self)
        self._deadline.setSingleShot(True)
        self._deadline.timeout.connect(self._request_expired)
        QApplication.instance().aboutToQuit.connect(self.shutdown)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 30, 32, 30)
        layout.setSpacing(18)

        title = QLabel(tr("Community"))
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            tr("The official Infernux community for support, ideas, and project sharing.")
        )
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(subtitle)

        hero = AnimatedSurfaceFrame("discussionHero")
        hero.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(30, 30, 30, 30)
        hero_layout.setSpacing(26)
        hero_layout.addWidget(DiscussionGlyph(), 0, Qt.AlignmentFlag.AlignTop)

        copy = QVBoxLayout()
        copy.setSpacing(8)
        eyebrow = QLabel(tr("INFERNUX COMMUNITY"))
        eyebrow.setObjectName("discussionEyebrow")
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
        address = QLabel("infernux-engine.discourse.group")
        address.setObjectName("discussionAddress")
        copy.addWidget(address)
        copy.addSpacing(8)
        enter = QPushButton(tr("Open Community"))
        enter.setObjectName("primaryBtn")
        enter.setCursor(Qt.CursorShape.PointingHandCursor)
        enter.setFixedHeight(38)
        enter.clicked.connect(self._open_forum)
        copy.addWidget(enter, 0, Qt.AlignmentFlag.AlignLeft)
        hero_layout.addLayout(copy, 1)
        layout.addWidget(hero)

        feed_header = QHBoxLayout()
        feed_title = QLabel(tr("Popular this week"))
        feed_title.setObjectName("communityFeedTitle")
        feed_header.addWidget(feed_title)
        feed_header.addStretch()
        self._refresh = QPushButton(tr("Refresh"))
        self._refresh.setFixedHeight(34)
        self._refresh.clicked.connect(self.refresh)
        feed_header.addWidget(self._refresh)
        layout.addLayout(feed_header)

        self._feed = QVBoxLayout()
        self._feed.setSpacing(8)
        layout.addLayout(self._feed)
        layout.addStretch()
        self._set_feed_message(
            tr("Select Refresh to load public community topics."), "empty"
        )

    def refresh(self) -> None:
        if self._closing or self._reply is not None:
            return
        self._clear_feed()
        self._set_feed_message(tr("Loading community topics..."), "loading")
        self._refresh.setEnabled(False)
        self._request_error = ""
        request = QNetworkRequest(QUrl(community_feed.HOT_TOPICS_URL))
        request.setRawHeader(b"Accept", b"application/json")
        request.setRawHeader(b"User-Agent", b"InfernuxHub-Community")
        request.setTransferTimeout(self.REQUEST_TIMEOUT_MS)
        self._reply = self._network.get(request)
        self._reply.finished.connect(self._request_finished)
        # An overall deadline also covers a response that trickles indefinitely.
        self._deadline.start(self.REQUEST_TIMEOUT_MS)

    def _request_expired(self) -> None:
        if self._reply is not None:
            self._request_error = tr("Community request timed out.")
            self._reply.abort()

    def _request_finished(self) -> None:
        reply, self._reply = self._reply, None
        if reply is None:
            return
        self._deadline.stop()
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self._show_error(self._request_error or reply.errorString())
            else:
                self._show_topics(parse_hot_topics(json.loads(bytes(reply.readAll()).decode("utf-8"))))
        except (UnicodeDecodeError, ValueError) as exc:
            self._show_error(str(exc))
        finally:
            reply.deleteLater()
            self._refresh.setEnabled(True)

    def shutdown(self) -> None:
        self._closing = True
        self._deadline.stop()
        reply, self._reply = self._reply, None
        if reply is not None:
            reply.abort()
            reply.deleteLater()

    def _clear_feed(self) -> None:
        while self._feed.count():
            item = self._feed.takeAt(0)
            widget = item.widget()
            if widget is not None:
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
        for topic in topics:
            card = QFrame()
            card.setObjectName("communityTopic")
            row = QHBoxLayout(card)
            row.setContentsMargins(16, 12, 12, 12)
            row.setSpacing(14)
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
            button.setFixedSize(76, 34)
            button.clicked.connect(
                lambda _checked=False, url=topic.url: QDesktopServices.openUrl(QUrl(url))
            )
            row.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
            self._feed.addWidget(card)

    def _show_error(self, message: str) -> None:
        self._clear_feed()
        self._set_feed_message(
            tr("Community topics could not be loaded: {message}", message=message),
            "error",
        )

    @classmethod
    def _open_forum(cls):
        QDesktopServices.openUrl(QUrl(cls.FORUM_URL))


__all__ = ["DiscussionView"]
