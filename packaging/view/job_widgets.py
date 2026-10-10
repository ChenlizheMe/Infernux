"""Inline transfer readouts bound to the shared InstallQueue."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from i18n import tr
from view.forge import SegmentMeter, StatusLed, fmt_bytes, fmt_eta, fmt_rate, mono_label, set_kind


STATE_TEXT = {
    "queued": "Queued",
    "running": "Installing",
    "cancelling": "Cancelling",
    "succeeded": "Completed",
    "failed": "Failed",
    "cancelled": "Cancelled",
}


def job_state_text(job) -> str:
    if job.state == "running" and job.total:
        return tr("Downloading")
    return tr(STATE_TEXT.get(job.state, job.state))


def job_detail_text(job) -> str:
    """Bytes, rate and ETA for transfers; the worker's message for other phases."""
    if job.state == "failed":
        return job.error
    if job.state in {"running", "cancelling"} and job.total:
        parts = [f"{fmt_bytes(job.completed)} / {fmt_bytes(job.total)}"]
        rate = fmt_rate(job.bytes_per_second)
        if rate:
            parts.append(rate)
        eta = fmt_eta(job.eta_seconds)
        if eta and job.state == "running":
            parts.append(tr("ETA {time}", time=eta))
        return "  ·  ".join(parts)
    if job.state in {"running", "cancelling"}:
        return tr(job.message) if job.message else ""
    if job.state == "queued":
        return tr("Waiting for the current installation")
    return ""


def job_led_kind(job) -> str:
    return {
        "queued": "idle", "running": "busy", "cancelling": "warn",
        "succeeded": "ok", "failed": "error", "cancelled": "idle",
    }.get(job.state, "idle")


def job_fraction(job) -> float | None:
    if job.state in {"running", "cancelling"}:
        return job.fraction
    return 1.0 if job.state == "succeeded" else 0.0


class JobStrip(QWidget):
    """Progress, rate, ETA and Cancel/Retry for the newest job with ``key``.

    Hidden when the key has no job, or once that job succeeded.
    """

    retried = Signal(object)

    def __init__(self, queue, key: str, parent=None, *, show_success: bool = False):
        super().__init__(parent)
        self._queue = queue
        self._key = key
        self._show_success = show_success
        self._job = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        self._led = StatusLed("idle")
        row.addWidget(self._led, 0, Qt.AlignmentFlag.AlignVCenter)
        self._state = mono_label("", "monoValue", spacing=1.0)
        row.addWidget(self._state)
        self._detail = mono_label("", "monoMeta", spacing=0.4)
        self._detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(self._detail, 1)
        self._cancel = QPushButton(tr("Cancel"))
        self._cancel.setObjectName("ghostBtn")
        self._cancel.setFixedHeight(26)
        self._cancel.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel)
        self._retry = QPushButton(tr("Retry"))
        self._retry.setObjectName("ghostBtn")
        self._retry.setFixedHeight(26)
        self._retry.clicked.connect(self._on_retry)
        row.addWidget(self._retry)
        layout.addLayout(row)
        self._meter = SegmentMeter(height=6)
        layout.addWidget(self._meter)
        queue.changed.connect(self.refresh)
        self.refresh()

    @property
    def job(self):
        return self._job

    def refresh(self):
        job = self._queue.job_for(self._key) if hasattr(self._queue, "job_for") else None
        self._job = job
        visible = job is not None and (job.state != "succeeded" or self._show_success) and job.state != "cancelled"
        self.setVisible(visible)
        if not visible:
            return
        self._led.set_kind(job_led_kind(job))
        self._state.setText(job_state_text(job).upper())
        detail = job_detail_text(job)
        if self._detail.text() != detail:
            self._detail.setText(detail)
            self._detail.setToolTip(detail)
        self._meter.set_fraction(job_fraction(job))
        self._meter.set_tone("error" if job.state == "failed" else "warn" if job.state == "cancelling" else "accent")
        self._meter.setVisible(job.state != "failed")
        cancellable = job.state == "queued" or (job.state == "running" and job.cancellable)
        self._cancel.setVisible(cancellable)
        self._cancel.setEnabled(job.state != "cancelling")
        self._retry.setVisible(job.state == "failed")

    def _on_cancel(self):
        if self._job is not None:
            self._queue.cancel(self._job)

    def _on_retry(self):
        if self._job is not None:
            job = self._queue.retry(self._job)
            if job is not None:
                self.retried.emit(job)


__all__ = ["JobStrip", "job_detail_text", "job_fraction", "job_led_kind", "job_state_text"]
