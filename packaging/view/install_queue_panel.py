"""Transfer deck: compact non-modal download/install activity for every page."""
from PySide6.QtCore import Qt, Signal, QTimer, QEvent
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget, QSizePolicy

from i18n import tr
from view.forge import SegmentMeter, StatusLed, fmt_rate, mono_label, track
from view.installs_view import _configure_install_scroll_area
from view.job_widgets import job_detail_text, job_fraction, job_led_kind, job_state_text


class InstallQueuePanel(QFrame):
    layout_changed = Signal()

    def __init__(self, queue, parent):
        super().__init__(parent)
        self.setObjectName("installQueuePanel")
        self._queue = queue
        self._expanded = False
        self._rows = {}
        self._leave_timer = QTimer(self)
        self._leave_timer.setSingleShot(True)
        self._leave_timer.setInterval(150)
        self._leave_timer.timeout.connect(self._collapse)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 5)
        layout.setSpacing(4)
        self._details = QFrame(parent)
        self._details.setObjectName("installQueuePopup")
        self._details.installEventFilter(self)
        details_layout = QVBoxLayout(self._details)
        details_layout.setContentsMargins(10, 8, 10, 8)
        header = QHBoxLayout()
        header.addWidget(mono_label(tr("TRANSFER DECK"), "pageKicker", spacing=1.6))
        header.addWidget(QLabel(tr("Downloads and installs")), 1)
        clear = QPushButton(tr("Clear finished"))
        clear.setObjectName("ghostBtn")
        clear.setMinimumHeight(26)
        clear.clicked.connect(queue.clear_finished)
        header.addWidget(clear)
        details_layout.addLayout(header)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        content = QWidget()
        _configure_install_scroll_area(self._scroll, content)
        self._items = QVBoxLayout(content)
        self._items.setContentsMargins(0, 0, 0, 0)
        self._items.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(content)
        details_layout.addWidget(self._scroll)
        self._bar = QWidget()
        self._bar.setFixedHeight(28)
        bar_layout = QVBoxLayout(self._bar)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        bar_layout.setSpacing(4)
        summary_row = QHBoxLayout()
        summary_row.setSpacing(8)
        self._led = StatusLed("idle")
        summary_row.addWidget(self._led, 0, Qt.AlignmentFlag.AlignVCenter)
        self._summary = QLabel()
        self._summary.setObjectName("queueSummary")
        track(self._summary, 0.6)
        self._summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        summary_row.addWidget(self._summary, 1)
        bar_layout.addLayout(summary_row)
        self._progress = SegmentMeter(height=3)
        bar_layout.addWidget(self._progress)
        layout.addWidget(self._bar)
        queue.changed.connect(self.refresh)
        self.refresh()

    def enterEvent(self, event):
        self._leave_timer.stop()
        self._expanded = True
        self.refresh()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._leave_timer.start()
        super().leaveEvent(event)

    def _collapse(self):
        cursor = QCursor.pos()
        if self.rect().contains(self.mapFromGlobal(cursor)) or (
            self._details.isVisible() and self._details.rect().contains(self._details.mapFromGlobal(cursor))
        ):
            return
        self._expanded = False
        self.refresh()

    def eventFilter(self, watched, event):
        if watched is self._details:
            if event.type() == QEvent.Type.Enter:
                self._leave_timer.stop()
            elif event.type() == QEvent.Type.Leave:
                self._leave_timer.start()
        return super().eventFilter(watched, event)

    def position_details(self):
        self._details.setFixedWidth(self.width())
        self._details.move(self.x(), max(0, self.y() - self._details.height() - 6))
        self._details.raise_()

    def hideEvent(self, event):
        self._expanded = False
        self._details.hide()
        super().hideEvent(event)

    def _build_row(self, job):
        row = QFrame()
        row.setObjectName("queueRow")
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(2, 8, 2, 10)
        row_layout.setSpacing(5)
        line = QHBoxLayout()
        line.setSpacing(8)
        led = StatusLed("idle")
        line.addWidget(led, 0, Qt.AlignmentFlag.AlignVCenter)
        title = QLabel(job.title)
        title.setObjectName("queueTitle")
        title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        line.addWidget(title, 1)
        state = mono_label("", "monoValue", spacing=1.0)
        line.addWidget(state)
        cancel = QPushButton(tr("Cancel"))
        cancel.setObjectName("ghostBtn")
        cancel.setFixedHeight(24)
        cancel.clicked.connect(lambda _checked=False, item=job: self._queue.cancel(item))
        line.addWidget(cancel)
        retry = QPushButton(tr("Retry"))
        retry.setObjectName("ghostBtn")
        retry.setFixedHeight(24)
        retry.clicked.connect(lambda _checked=False, item=job: self._queue.retry(item))
        line.addWidget(retry)
        row_layout.addLayout(line)
        status = QLabel()
        status.setWordWrap(True)
        status.setTextFormat(Qt.TextFormat.PlainText)
        status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status.setObjectName("cardPath")
        row_layout.addWidget(status)
        progress = SegmentMeter(height=5)
        row_layout.addWidget(progress)
        self._items.addWidget(row)
        return row, status, progress, cancel, retry, led, state

    def refresh(self):
        jobs = self._queue.jobs
        current = {id(job) for job in jobs}
        for key in tuple(self._rows):
            if key not in current:
                row = self._rows.pop(key)[0]
                self._items.removeWidget(row)
                row.deleteLater()
        for job in jobs:
            key = id(job)
            if key not in self._rows:
                self._rows[key] = self._build_row(job)
            _row, status, progress, cancel, retry, led, state = self._rows[key]
            detail = job_detail_text(job)
            if status.text() != detail:
                status.setText(detail)
            status.setVisible(bool(detail))
            state.setText(job_state_text(job).upper())
            led.set_kind(job_led_kind(job))
            cancel.setVisible(job.state == "queued" or (job.state == "running" and job.cancellable))
            retry.setVisible(job.state == "failed")
            progress.setVisible(job.state in {"running", "cancelling"})
            progress.set_fraction(job_fraction(job))
            progress.set_tone("warn" if job.state == "cancelling" else "accent")
        active = next((job for job in jobs if job.state in {"running", "cancelling"}), None)
        queued = sum(job.state == "queued" for job in jobs)
        failed = sum(job.state == "failed" for job in jobs)
        summary = active.title if active else tr("Downloads and installs")
        if active and active.total:
            summary += f" · {int(active.completed * 100 / active.total)}%"
            rate = fmt_rate(active.bytes_per_second)
            if rate:
                summary += f" · {rate}"
        if queued and active:
            summary += f" · +{queued}"
        elif queued:
            summary += " · " + tr("{count} queued", count=queued)
        elif not active:
            summary = tr("{count} failed", count=failed) if failed else tr("Installations complete")
        self._summary.setText(summary)
        self._summary.setToolTip(summary)
        self._led.set_kind("busy" if active else "error" if failed else "ok")
        self._progress.setVisible(active is not None)
        self._progress.set_fraction(active.fraction if active else 0.0)
        self._details.setFixedHeight(48 + min(300, len(jobs) * 92))
        self._details.setVisible(self._expanded and bool(jobs))
        self.setFixedHeight(40)
        self.position_details()
        self.setVisible(bool(jobs))
        self.layout_changed.emit()
