"""Unity-style splash screen shown while the engine is loading."""

import os
import subprocess
import sys
import tempfile
import uuid

from PySide6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QGraphicsDropShadowEffect, QProgressBar, QMessageBox
from PySide6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QFont, QPainter, QColor, QPen, QBrush


class EngineSplashScreen(QWidget):
    """Borderless square overlay shown while the engine process starts up.

    Displays the engine icon, name, and an animated loading indicator.
    Fades in on show, fades out when the detached engine process signals
    readiness via a small ready-file, then closes.
    """

    _SPLASH_SIZE = 420
    _FADE_IN_MS = 150
    _FADE_OUT_MS = 200

    def __init__(self, icon_path: str, project_name: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(self._SPLASH_SIZE, self._SPLASH_SIZE)

        self._process: subprocess.Popen | None = None
        self._ready_file: str = ""
        self._angle = 0  # spinner angle
        self._closing = False

        # Center on screen
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(
                geo.x() + (geo.width() - self._SPLASH_SIZE) // 2,
                geo.y() + (geo.height() - self._SPLASH_SIZE) // 2,
            )

        # ── Layout ──
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 50, 40, 40)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignCenter)

        # Icon
        self._icon_label = QLabel(self)
        self._icon_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(icon_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(QSize(96, 96), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._icon_label.setPixmap(pixmap)
        layout.addWidget(self._icon_label)

        # Title
        title = QLabel("Infernux Engine", self)
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setStyleSheet("color: #e0e0e0; background: transparent;")
        layout.addWidget(title)

        # Project name
        proj = QLabel(project_name, self)
        proj.setAlignment(Qt.AlignCenter)
        proj.setFont(QFont("Segoe UI", 11))
        proj.setStyleSheet("color: #888888; background: transparent;")
        layout.addWidget(proj)

        # Spacer
        layout.addSpacing(20)

        # Status label
        self._status = QLabel("Initializing engine…", self)
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setFont(QFont("Segoe UI", 10))
        self._status.setStyleSheet("color: #666666; background: transparent;")
        layout.addWidget(self._status)
        # Progress bar
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setStyleSheet(
            "QProgressBar { background: #333333; border: none; border-radius: 2px; }"
            "QProgressBar::chunk { background: #e0e0e0; border-radius: 2px; }"
        )
        layout.addWidget(self._progress_bar)
        # Spinner animation timer
        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(self._tick_spinner)
        self._spin_timer.start(30)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_launch_state)

    # ── Show with fade-in ──

    def show(self):
        self.setWindowOpacity(0.0)
        super().show()
        self._fade_in_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_in_anim.setDuration(self._FADE_IN_MS)
        self._fade_in_anim.setStartValue(0.0)
        self._fade_in_anim.setEndValue(1.0)
        self._fade_in_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_in_anim.start()

    # ── Painting ──

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Dark rounded-rect background
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(25, 25, 25, 245)))
        p.drawRoundedRect(self.rect(), 18, 18)

        # Subtle border
        p.setPen(QPen(QColor(60, 60, 60, 180), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 18, 18)

        # Spinner arc at the bottom
        spinner_size = 28
        sx = (self.width() - spinner_size) // 2
        sy = self.height() - 52
        pen = QPen(QColor(180, 180, 180), 3)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(sx, sy, spinner_size, spinner_size, self._angle * 16, 270 * 16)

        p.end()

    def _tick_spinner(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    # ── Fade-out and close ──

    def _fade_out_and_close(self):
        if self._closing:
            return
        self._closing = True
        self._spin_timer.stop()
        self._poll_timer.stop()

        self._fade_out_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_out_anim.setDuration(self._FADE_OUT_MS)
        self._fade_out_anim.setStartValue(self.windowOpacity())
        self._fade_out_anim.setEndValue(0.0)
        self._fade_out_anim.setEasingCurve(QEasingCurve.InCubic)
        self._fade_out_anim.finished.connect(self._finish_close)
        self._fade_out_anim.start()

    # ── Process management ──

    def launch(self, python_exe: str, script: str, project_path: str,
               *, detached: bool = True):
        """Start the engine process and poll a ready-file for progress.

        Args:
            detached: If *True* (packaged mode), the engine runs as a fully
                detached process with no console.  If *False* (dev mode),
                stdout/stderr are inherited so logs remain visible.
        """
        self._ready_file = os.path.join(
            tempfile.gettempdir(), f"infengine_ready_{uuid.uuid4().hex}.flag"
        )
        env = os.environ.copy()
        env["_INFENGINE_READY_FILE"] = self._ready_file

        popen_kwargs: dict = {"cwd": project_path, "env": env}

        # Always capture stderr so we can report the real error when
        # the engine crashes.  In detached mode, suppress the console
        # window; in dev mode, let stdout pass through for logging.
        popen_kwargs["stdin"] = subprocess.DEVNULL
        popen_kwargs["stderr"] = subprocess.PIPE
        if detached:
            popen_kwargs["stdout"] = subprocess.DEVNULL
            if sys.platform == "win32":
                flags = 0
                flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                popen_kwargs["creationflags"] = flags
            else:
                popen_kwargs["start_new_session"] = True

        try:
            self._process = subprocess.Popen(
                [python_exe, "-u", "-c", script, project_path],
                **popen_kwargs,
            )
        except OSError:
            self._status.setText("Launch failed")
            QTimer.singleShot(250, self._fade_out_and_close)
            return
        self._poll_timer.start(100)

    def _poll_launch_state(self):
        if self._ready_file and os.path.isfile(self._ready_file):
            try:
                with open(self._ready_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except OSError:
                return

            # Race condition: writer may have truncated but not yet written.
            if not content:
                return

            if content.startswith("ENGINE_LOADED"):
                self._status.setText("Ready")
                self._progress_bar.setValue(100)
                self._fade_out_and_close()
                return

            if content.startswith("LOADING:"):
                try:
                    _, fraction, message = content.split(":", 2)
                    self._status.setText(message)
                    current, total = fraction.split("/")
                    self._progress_bar.setValue(
                        int(int(current) * 100 / int(total))
                    )
                except (ValueError, ZeroDivisionError):
                    pass
            return

        if self._process is not None and self._process.poll() is not None:
            # Engine exited before signalling readiness — show the error.
            stderr_text = ""
            try:
                raw = self._process.stderr.read()
                if raw:
                    stderr_text = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                pass

            self._status.setText("Launch failed")
            self._fade_out_and_close()

            if stderr_text:
                QMessageBox.critical(
                    None,
                    "Engine Launch Failed",
                    f"The engine process exited with an error:\n\n"
                    f"{stderr_text[:2000]}",
                )
            else:
                QMessageBox.critical(
                    None,
                    "Engine Launch Failed",
                    f"The engine process exited unexpectedly "
                    f"(code {self._process.returncode}).\n\n"
                    "Check the project's Logs/ folder for details.",
                )

    def close(self):
        self._fade_out_and_close()

    def _finish_close(self):
        if self._ready_file:
            try:
                os.remove(self._ready_file)
            except OSError:
                pass
            self._ready_file = ""
        self.hide()
        super().close()
        self.deleteLater()
