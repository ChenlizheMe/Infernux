"""Modal Hub migration progress; disk copying never blocks the GUI thread."""

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

from i18n import tr
from shared_storage_migration import migrate_legacy_storage
from view.forge import HazardStripe, SegmentMeter, mono_label


class _MigrationThread(QThread):
    progress = Signal(str)

    def __init__(self, plan, project_roots, parent=None):
        super().__init__(parent)
        self.plan = plan
        self.project_roots = project_roots
        self.result = ()
        self.error = ""

    def run(self):
        try:
            self.result = migrate_legacy_storage(
                self.plan, self.project_roots, progress=self.progress.emit
            )
        except Exception as exc:
            self.error = str(exc)


class StorageMigrationDialog(QDialog):
    def __init__(self, plan, project_roots, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Migrate Legacy Resources"))
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._stripe = HazardStripe(5)
        self._stripe.set_active(True)
        layout.addWidget(self._stripe)
        body = QVBoxLayout()
        body.setContentsMargins(24, 18, 24, 22)
        body.setSpacing(10)
        body.addWidget(mono_label(tr("STORAGE  ·  TRANSFER IN PROGRESS"), "pageKicker", spacing=1.6))
        title = QLabel(tr("Migrate Legacy Resources"))
        title.setObjectName("dialogTitle")
        body.addWidget(title)
        self.status = QLabel(tr("Moving resources. Keep Hub open until this finishes."))
        self.status.setObjectName("dialogBody")
        self.status.setWordWrap(True)
        body.addWidget(self.status)
        meter = SegmentMeter(height=6)
        meter.set_fraction(None)
        body.addWidget(meter)
        layout.addLayout(body)
        self.worker = _MigrationThread(plan, project_roots, self)
        self.worker.progress.connect(self.status.setText)
        self.worker.finished.connect(self.accept)

    def exec(self):
        self.worker.start()
        return super().exec()

    def reject(self):
        if not self.worker.isRunning():
            super().reject()

    def closeEvent(self, event):
        if self.worker.isRunning():
            event.ignore()
        else:
            super().closeEvent(event)
