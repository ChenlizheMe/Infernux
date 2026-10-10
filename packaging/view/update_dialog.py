"""Update approval and shared-queue staging for Infernux Hub."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PySide6.QtCore import QObject, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication

from hub_updater import HubUpdateStatus, check_for_update, launch_external_updater, stage_update
from i18n import tr
from view import dialogs


def _write_update_trace(payload: dict) -> None:
    """Write an opt-in packaged-update diagnostic for release verification."""
    destination = os.environ.get("INFERNUX_HUB_UPDATE_TRACE", "").strip()
    if not destination:
        return
    try:
        path = Path(destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


class _CheckWorker(QThread):
    checked = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            result = check_for_update()
            update = result.update
            _write_update_trace(
                {
                    "status": result.status.value,
                    "current_version": result.current_version,
                    "target_version": result.latest_version,
                    "asset_name": update.asset_name if update is not None else "",
                    "detail": result.detail,
                }
            )
            self.checked.emit(result)
        except Exception as exc:
            _write_update_trace({"status": "failed", "error": str(exc)})
            self.failed.emit(str(exc))


class UpdateController(QObject):
    """Own worker lifetime and present updates without interrupting the user.

    Startup and periodic checks are silent: an available update lights the
    sidebar's update pill and posts a toast. A manual check, or a click on the
    pill, opens the update plate directly.
    """

    check_finished = Signal()
    update_available = Signal(object)
    SKIP_SETTING = "skipped_hub_update"

    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        self.queue = main_window.install_queue
        self._update_job = None
        self.pending_update = None
        self.queue.idle.connect(self._apply_staged_update)
        self.thread = None
        self._silent_check = True
        self._completion_pending = False
        QApplication.instance().aboutToQuit.connect(self._wait_for_check)

    def _wait_for_check(self):
        if self.thread is not None:
            self.thread.wait()

    def check(self, *, silent: bool = True):
        if self.thread and self.thread.isRunning():
            # A manual click joins the startup request and must receive its result.
            self._silent_check = self._silent_check and silent
            return
        self._silent_check = silent
        self._completion_pending = True
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = _CheckWorker(self)
        # Connect to QObject-bound slots, not lambdas.  Lambdas have no Qt
        # receiver affinity and therefore run in the worker thread, which
        # caused dialogs to be created for the main window across threads.
        self.thread.checked.connect(self._checked)
        self.thread.failed.connect(self._check_failed)
        self.thread.start()

    def _setting(self, key: str, default: str = "") -> str:
        database = getattr(self.main_window, "db", None)
        return database.get_setting(key, default) if database is not None else default

    def _set_setting(self, key: str, value: str) -> None:
        database = getattr(self.main_window, "db", None)
        if database is not None:
            database.set_setting(key, value)

    def _checked(self, result):
        if result.status is HubUpdateStatus.UP_TO_DATE:
            self.pending_update = None
            if not self._silent_check:
                dialogs.information(
                    self.main_window, tr("Hub Update"), tr("Infernux Hub is up to date."),
                    level="ok", kicker=tr("HUB UPDATE"),
                )
            self._finish_check()
            return
        if result.status is HubUpdateStatus.NETWORK_UNAVAILABLE:
            if not self._silent_check:
                dialogs.warning(
                    self.main_window,
                    tr("Update Check Unavailable"),
                    tr("The Hub update catalog could not be reached."),
                    detail=result.detail, kicker=tr("HUB UPDATE"),
                )
            self._finish_check()
            return
        if result.status is HubUpdateStatus.CATALOG_INVALID:
            if not self._silent_check:
                dialogs.warning(
                    self.main_window,
                    tr("Update Catalog Invalid"),
                    tr("The Hub update catalog is invalid."),
                    detail=result.detail, kicker=tr("HUB UPDATE"),
                )
            self._finish_check()
            return
        if result.status is HubUpdateStatus.UNSUPPORTED_CURRENT_VERSION:
            answer = dialogs.question(
                self.main_window,
                tr("Full Hub Install Required"),
                tr(
                    "This Hub is too old for the current in-app update path. "
                    "Open the {version} installer download now?",
                    version=result.latest_version,
                ),
                dialogs.Yes | dialogs.No, dialogs.Yes,
                labels={dialogs.Yes: tr("Open download"), dialogs.No: tr("Later")},
                kicker=tr("HUB UPDATE"),
            )
            if answer == dialogs.Yes:
                QDesktopServices.openUrl(QUrl(result.installer_url))
            self._finish_check()
            return
        update = result.update
        if result.status is not HubUpdateStatus.UPDATE_AVAILABLE or update is None:
            raise RuntimeError(f"Unhandled Hub update status: {result.status}")
        self.pending_update = result
        if self._silent_check:
            if self._setting(self.SKIP_SETTING) != update.target_version:
                self.update_available.emit(result)
            self._finish_check()
            return
        self.prompt_update()
        self._finish_check()

    def prompt_update(self) -> bool:
        """Ask once for the pending update; queue the download on consent."""
        result = self.pending_update
        if result is None or result.update is None:
            return False
        update = result.update
        if self._update_job is not None and self._update_job.active:
            return True
        size = getattr(update, "size", 0)
        detail_lines = [
            f"{tr('Current')}: {result.current_version}",
            f"{tr('Target')}: {update.target_version}",
        ]
        if size:
            from view.forge import fmt_bytes
            detail_lines.append(f"{tr('Download')}: {fmt_bytes(size)}")
        if getattr(update, "platform", ""):
            detail_lines.append(f"{tr('Platform')}: {update.platform}")
        release_url = getattr(update, "release_url", "")
        answer = dialogs.question(
            self.main_window,
            tr("Infernux Hub {version} is ready", version=update.target_version),
            tr(
                "Hub downloads the update in the background, then closes, installs it and restarts. "
                "Projects, engines and settings are kept."
            ),
            dialogs.Yes | dialogs.No | dialogs.StandardButton.Ignore, dialogs.Yes,
            labels={
                dialogs.Yes: tr("Update and restart"),
                dialogs.No: tr("Later"),
                dialogs.StandardButton.Ignore: tr("Skip this version"),
            },
            detail="\n".join(detail_lines),
            kicker=tr("HUB UPDATE  ·  {current} TO {target}", current=result.current_version,
                      target=update.target_version),
            link=(tr("Release notes"), release_url) if release_url else None,
        )
        if answer == dialogs.StandardButton.Ignore:
            self._set_setting(self.SKIP_SETTING, update.target_version)
            self.pending_update = None
            self.update_available.emit(None)
            return False
        if answer != dialogs.Yes:
            return False
        self._update_job = self.queue.submit(
            f"hub-update:{update.target_version}",
            tr("Hub update {version}", version=update.target_version),
            lambda report: str(stage_update(
                update, lambda done, total: report(tr("Downloading"), done, total),
            )),
        )
        return True

    def _apply_staged_update(self):
        job = self._update_job
        if job is None or job.state != "succeeded":
            return
        self._update_job = None
        try:
            application = QApplication.instance()
            launch_external_updater(
                job.result, is_dark=bool(getattr(application, "is_dark_theme", True)),
            )
        except Exception as exc:
            job.state, job.error = "failed", str(exc)
            self.queue.changed.emit()
            return
        self.main_window.hide()
        self.main_window.app.quit()

    def _check_failed(self, message: str):
        if not self._silent_check:
            dialogs.warning(self.main_window, tr("Update Check Failed"), message, kicker=tr("HUB UPDATE"))
        self._finish_check()

    def _finish_check(self) -> None:
        if not self._completion_pending:
            return
        self._completion_pending = False
        self.check_finished.emit()


__all__ = ["UpdateController"]
