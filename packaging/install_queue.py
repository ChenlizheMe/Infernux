"""One application-owned installation queue; workers never own Hub windows."""
from __future__ import annotations

from collections import deque
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from PySide6.QtCore import QObject, QThread, QTimer, Signal


Progress = Callable[[str, int, int], None]


class InstallCancelled(Exception):
    """Raised inside a worker at its next progress checkpoint after Cancel."""


@dataclass
class InstallJob:
    key: str
    title: str
    operation: Callable[[Progress], object] = field(repr=False)
    state: str = "queued"
    message: str = ""
    completed: int = 0
    total: int = 0
    result: object = None
    error: str = ""
    # Only byte transfers are cancellable: they stop at a chunk boundary and
    # leave no published state. Runtime publication must run to completion.
    cancellable: bool = False
    bytes_per_second: float = 0.0
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    _sample: tuple[float, int] | None = field(default=None, repr=False, compare=False)

    @property
    def active(self) -> bool:
        return self.state in {"queued", "running", "cancelling"}

    @property
    def fraction(self) -> float | None:
        return min(1.0, self.completed / self.total) if self.total else None

    @property
    def eta_seconds(self) -> float | None:
        if not self.total or self.bytes_per_second <= 1.0:
            return None
        return max(0.0, (self.total - self.completed) / self.bytes_per_second)


class ProgressReporter:
    """Progress callback handed to an operation; doubles as its cancel token."""

    def __init__(self, job: InstallJob, emit: Callable[[str, object, object], None]):
        self._job = job
        self._emit = emit

    def should_cancel(self) -> bool:
        return self._job.cancel_event.is_set()

    def __call__(self, message: str, completed: int = 0, total: int = 0) -> None:
        if self.should_cancel():
            raise InstallCancelled(self._job.title)
        self._emit(message, completed, total)


class _InstallThread(QThread):
    # Android downloads exceed Qt's signed 32-bit int signal payload.
    progress = Signal(str, object, object)

    def __init__(self, job: InstallJob, parent: QObject):
        super().__init__(parent)
        self.job = job
        self.result = None
        self.error = ""
        self.cancelled = False

    def run(self):
        try:
            logging.getLogger(__name__).info("Installation started: %s", self.job.key)
            self.result = self.job.operation(ProgressReporter(self.job, self.progress.emit))
            logging.getLogger(__name__).info("Installation completed: %s", self.job.key)
        except Exception as exc:
            if self.job.cancel_event.is_set() and _is_cancellation(exc):
                logging.getLogger(__name__).info("Installation cancelled: %s", self.job.key)
                self.cancelled = True
                return
            logging.getLogger(__name__).exception("Installation failed: %s", self.job.key)
            self.error = f"{type(exc).__name__}: {exc}"


def _is_cancellation(exc: BaseException) -> bool:
    # Managers may translate the reporter's exception into their own type.
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, InstallCancelled) or type(exc).__name__ == "DownloadCancelled":
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class InstallQueue(QObject):
    changed = Signal()
    job_finished = Signal(object)
    idle = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs: list[InstallJob] = []
        self._pending: deque[InstallJob] = deque()
        self._thread: _InstallThread | None = None
        # Downloads report every 64 KiB; repaint listeners at most ~20 times a second.
        self._progress_timer = QTimer(self)
        self._progress_timer.setSingleShot(True)
        self._progress_timer.setInterval(50)
        self._progress_timer.timeout.connect(self.changed)

    @property
    def busy(self) -> bool:
        return self._thread is not None or bool(self._pending)

    def submit(
        self, key: str, title: str, operation: Callable[[Progress], object], *, cancellable: bool = False,
    ) -> InstallJob:
        for job in self.jobs:
            if job.key == key and job.active:
                return job
        job = InstallJob(key, title, operation, cancellable=cancellable)
        self.jobs.append(job)
        self._pending.append(job)
        self.changed.emit()
        QTimer.singleShot(0, self._start_next)
        return job

    def cancel(self, job: InstallJob) -> bool:
        """Cancel a queued job now, or ask a running cancellable job to stop."""
        if job.state == "queued":
            self._pending.remove(job)
            job.state = "cancelled"
            self.changed.emit()
            return True
        if job.state == "running" and job.cancellable:
            job.cancel_event.set()
            job.state = "cancelling"
            self.changed.emit()
            return True
        return False

    def cancel_queued(self, job: InstallJob) -> None:
        if job.state == "queued":
            self.cancel(job)

    def retry(self, job: InstallJob) -> InstallJob | None:
        if job.active:
            return None
        self.jobs.remove(job)
        return self.submit(job.key, job.title, job.operation, cancellable=job.cancellable)

    def clear_finished(self) -> None:
        self.jobs[:] = [job for job in self.jobs if job.active]
        self.changed.emit()

    def is_pending(self, key: str) -> bool:
        return any(job.key == key and job.active for job in self.jobs)

    def job_for(self, key: str) -> InstallJob | None:
        return next((job for job in reversed(self.jobs) if job.key == key), None)

    def _start_next(self):
        if self._thread is not None:
            return
        if not self._pending:
            self.idle.emit()
            return
        job = self._pending.popleft()
        job.state = "running"
        self._thread = _InstallThread(job, self)
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.start()
        self.changed.emit()

    def _on_progress(self, message: str, completed: int, total: int):
        if self._thread is None:
            return
        job = self._thread.job
        now = time.monotonic()
        if total and job._sample is not None and completed >= job._sample[1]:
            elapsed = now - job._sample[0]
            if elapsed >= 0.25:
                rate = (completed - job._sample[1]) / elapsed
                # Exponential smoothing keeps the readout steady on bursty links.
                job.bytes_per_second = rate if job.bytes_per_second <= 0 else 0.7 * job.bytes_per_second + 0.3 * rate
                job._sample = (now, completed)
        elif total:
            job._sample = (now, completed)
        else:
            job._sample, job.bytes_per_second = None, 0.0
        message_changed = message != job.message
        job.message, job.completed, job.total = message, completed, total
        if message_changed:
            self._progress_timer.stop()
            self.changed.emit()
        elif not self._progress_timer.isActive():
            self._progress_timer.start()

    def _on_finished(self):
        thread = self._thread
        thread.wait()
        self._thread = None
        job = thread.job
        job.result, job.error = thread.result, thread.error
        job.bytes_per_second = 0.0
        if thread.cancelled:
            job.state = "cancelled"
        else:
            job.state = "failed" if job.error else "succeeded"
        thread.deleteLater()
        self.changed.emit()
        self.job_finished.emit(job)
        QTimer.singleShot(0, self._start_next)
