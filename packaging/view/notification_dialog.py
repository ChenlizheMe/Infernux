"""Presentation layer for Hub's version-scoped notification queue."""

from __future__ import annotations

import threading
import urllib.request
from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, Signal

from hub_notifications import HUB_NOTIFICATIONS_URL, HubNotificationQueue
from hub_updater import current_hub_version
from i18n import tr
from view import dialogs


class _NoticeBridge(QObject):
    # urllib runs on a daemon thread (QtNetwork TLS plugins are not always
    # shipped); the payload returns to the GUI thread through this signal.
    loaded = Signal(object)


class HubNotificationController:
    def __init__(
        self,
        main_window,
        database,
        *,
        open_installs: Callable[[], None],
        queue: HubNotificationQueue | None = None,
    ) -> None:
        self._main_window = main_window
        self._open_installs = open_installs
        self._queue = queue or HubNotificationQueue(database)
        self._bridge = _NoticeBridge(main_window)
        self._bridge.loaded.connect(self._notification_download_finished, Qt.ConnectionType.QueuedConnection)
        self._loading = False

    def show_pending(self) -> None:
        if self._queue.source_path is not None:
            self._present(self._queue.pending(current_hub_version()))
            return
        if self._loading:
            return
        self._loading = True
        bridge = self._bridge

        def fetch():
            payload = None
            try:
                request = urllib.request.Request(
                    HUB_NOTIFICATIONS_URL,
                    headers={"Accept": "application/json", "User-Agent": "InfernuxHub-Notifications"},
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    payload = response.read(1024 * 1024)
            except Exception:
                payload = None
            try:
                bridge.loaded.emit(payload)
            except RuntimeError:
                pass  # Hub closed while the request was in flight.

        threading.Thread(target=fetch, name="hub-notifications", daemon=True).start()

    def _notification_download_finished(self, payload) -> None:
        self._loading = False
        if not payload:
            return
        try:
            notifications = self._queue.pending_bytes(current_hub_version(), bytes(payload))
        except (UnicodeDecodeError, ValueError):
            return
        self._present(notifications)

    def _present(self, notifications) -> None:
        for notification in notifications:
            # Record delivery before opening the modal dialog so a crash or forced
            # shutdown cannot turn a show-once notice into a startup loop.
            self._queue.mark_seen(notification)
            level = {"warning": "warning", "critical": "critical"}.get(notification.level, "info")
            dialog = dialogs.HubDialog(
                self._main_window, level=level, title=notification.title,
                text=notification.message, kicker=tr("RELEASE NOTICE"),
            )
            close = dialog.addButton(tr("Close"), "reject")
            action_button = None
            if notification.action and notification.action_label:
                action_button = dialog.addButton(notification.action_label, "accept")
            dialog.setDefaultButton(action_button or close)
            dialog.exec()
            if (
                action_button is not None
                and dialog.clickedButton() is action_button
                and notification.action == "open_installs"
            ):
                self._open_installs()


__all__ = ["HubNotificationController"]
