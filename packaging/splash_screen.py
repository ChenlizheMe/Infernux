"""Unity-style splash screen shown while the engine is loading."""

import contextlib
import logging
import os
import subprocess
import sys
import tempfile
import threading
import uuid

from PySide6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QGraphicsDropShadowEffect, QProgressBar, QMessageBox
from PySide6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QFont, QPainter, QColor, QPen, QBrush

from hub_utils import get_project_lock_path, remove_project_lock, write_project_lock


_WIN_CRASH_CODES = {
    -1073741819: "Access violation (0xC0000005)",
    -1073740791: "Stack buffer overrun (0xC0000409)",
    -1073741571: "Stack overflow (0xC00000FD)",
    -1073741676: "Integer divide by zero (0xC0000094)",
    -1073741685: "Illegal instruction (0xC000001D)",
}

# Unix signal-based crash codes (negative signal number)
_UNIX_CRASH_CODES = {
    -6: "SIGABRT — Aborted",
    -11: "SIGSEGV — Segmentation fault",
    -4: "SIGILL — Illegal instruction",
    -8: "SIGFPE — Floating point exception",
    -5: "SIGTRAP — Trace/breakpoint trap",
    -9: "SIGKILL — Killed",
    -10: "SIGBUS — Bus error",
}


@contextlib.contextmanager
def _suppress_windows_error_dialogs():
    """Temporarily suppress Windows crash dialog boxes for child processes."""
    if sys.platform != "win32":
        yield
        return

    try:
        import ctypes

        SEM_FAILCRITICALERRORS = 0x0001
        SEM_NOGPFAULTERRORBOX = 0x0002
        previous_mode = ctypes.windll.kernel32.SetErrorMode(
            SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
        )
    except Exception:
        yield
        return

    try:
        yield
    finally:
        try:
            ctypes.windll.kernel32.SetErrorMode(previous_mode)
        except Exception as _exc:
            logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
            pass


def _format_exit_code(returncode: int) -> str:
    """Return a user-facing explanation for process exit codes."""
    if returncode in _WIN_CRASH_CODES:
        return f"{_WIN_CRASH_CODES[returncode]}\nRaw exit code: {returncode}"
    if returncode in _UNIX_CRASH_CODES:
        return f"{_UNIX_CRASH_CODES[returncode]}\nRaw exit code: {returncode}"
    return f"Raw exit code: {returncode}"


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
        self._project_path: str = ""
        self._lock_token: str = ""
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
               *, detached: bool = True, extra_env: dict[str, str] | None = None):
        """Start the engine process and poll a ready-file for progress.

        Args:
            detached: If *True* (packaged mode), the engine runs as a fully
                detached process with no console.  If *False* (dev mode),
                stdout/stderr are inherited so logs remain visible.
        """
        self._ready_file = os.path.join(
            tempfile.gettempdir(), f"infernux_ready_{uuid.uuid4().hex}.flag"
        )
        self._project_path = project_path
        self._lock_token = uuid.uuid4().hex
        env = os.environ.copy()
        env["_INFERNUX_READY_FILE"] = self._ready_file
        env["_INFERNUX_PROJECT_LOCK_PATH"] = get_project_lock_path(project_path)
        env["_INFERNUX_PROJECT_LOCK_TOKEN"] = self._lock_token
        if extra_env:
            env.update(extra_env)
        write_project_lock(project_path, os.getpid(), self._lock_token, "editor", "launching")

        popen_kwargs: dict = {"cwd": project_path, "env": env}

        if detached:
            # Engine has its own Console panel — never inherit stdout/stderr
            # to the launcher terminal.  stderr is piped so we can display
            # crash messages; a background thread drains it continuously to
            # prevent the OS pipe-buffer from filling up and deadlocking the
            # engine process.
            popen_kwargs["stdin"] = subprocess.DEVNULL
            popen_kwargs["stdout"] = subprocess.DEVNULL
            popen_kwargs["stderr"] = subprocess.PIPE
            if sys.platform == "win32":
                flags = 0
                flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
                flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                popen_kwargs["creationflags"] = flags
            else:
                popen_kwargs["start_new_session"] = True
        else:
            # Dev mode — spawn a visible console so developers can read
            # stdout/stderr directly.  stderr is still piped for the
            # crash-message dialog.
            popen_kwargs["stdin"] = subprocess.DEVNULL
            popen_kwargs["stdout"] = None          # inherit → visible console
            popen_kwargs["stderr"] = subprocess.PIPE
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE

        try:
            with _suppress_windows_error_dialogs():
                self._process = subprocess.Popen(
                    [python_exe, "-u", "-c", script, project_path],
                    **popen_kwargs,
                )
        except OSError:
            remove_project_lock(project_path, self._lock_token)
            self._status.setText("Launch failed")
            QTimer.singleShot(250, self._fade_out_and_close)
            return

        # Drain stderr in a background thread so the pipe buffer never
        # fills up (which would block the engine's sys.stderr.write).
        self._stderr_chunks: list[bytes] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True
        )
        self._stderr_thread.start()

        self._poll_timer.start(100)

    def _drain_stderr(self):
        """Read stderr until EOF so the pipe buffer never stalls the engine."""
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        try:
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                self._stderr_chunks.append(chunk)
        except (ValueError, OSError) as _exc:
            logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
            pass
        finally:
            try:
                proc.stderr.close()
            except Exception as _exc:
                logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
                pass

    def _poll_launch_state(self):
        if self._ready_file and os.path.isfile(self._ready_file):
            try:
                with open(self._ready_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except OSError as _exc:
                logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
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
                except (ValueError, ZeroDivisionError) as _exc:
                    logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
                    pass
            return

        if self._process is not None and self._process.poll() is not None:
            # Engine exited before signalling readiness — show the error.
            # Collect stderr from the drain thread's buffer.
            if hasattr(self, '_stderr_thread'):
                self._stderr_thread.join(timeout=2.0)
            stderr_text = b"".join(
                getattr(self, '_stderr_chunks', [])
            ).decode("utf-8", errors="replace").strip()

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
                    f"before the editor finished loading.\n\n"
                    f"{_format_exit_code(self._process.returncode)}\n\n"
                    "Check the project's Logs/ folder for details.",
                )

    def close(self):
        self._fade_out_and_close()

    def _finish_close(self):
        if self._process is not None and self._process.poll() is not None and self._project_path:
            remove_project_lock(self._project_path, self._lock_token or None)
        if self._ready_file:
            try:
                os.remove(self._ready_file)
            except OSError as _exc:
                logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
                pass
            self._ready_file = ""
        self.hide()
        super().close()
        self.deleteLater()
