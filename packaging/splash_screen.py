"""Unity-style splash screen shown while the engine is loading."""

import contextlib
import math
import random
import zlib
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from PySide6.QtWidgets import QWidget, QApplication, QLabel, QProgressBar
from PySide6.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, QUrl, QPointF, QRectF
from PySide6.QtGui import QPixmap, QFont, QFontMetrics, QPainter, QColor, QPen, QDesktopServices, QPolygonF

from hub_utils import get_project_lock_path, merge_child_env_utf8, remove_project_lock, write_project_lock
from i18n import tr
from style import FONT_FAMILIES, StyleManager
from view import dialogs
from view.forge import mix, qcolor
from python_execution import EditorPythonRuntime, python_executable_path


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


class _StatusLine(QLabel):
    """Status label that also feeds the splash's telemetry log."""

    def __init__(self, text: str, owner: "EngineSplashScreen"):
        super().__init__(text, owner)
        self._owner = owner

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        if text and text != self.text():
            self._owner._log_status(text)
        super().setText(text)


class EngineSplashScreen(QWidget):
    """Editor launch sequence shown while the engine process starts up.

    A wide instrument plate: a 3N ignition network on the left lights up as
    loading advances; the right column names the project and walks through
    the launch stages with a telemetry tail. It fades in on show and fades
    out when the detached engine process signals readiness via a ready-file.
    """

    _SPLASH_W = 760
    _SPLASH_H = 420
    _SPLASH_SIZE = _SPLASH_W
    _FADE_IN_MS = 150
    _FADE_OUT_MS = 200
    _STARTUP_TIMEOUT_SECONDS = 90
    _STAGES = (
        ("PREPARE RUNTIME", 0, 10),
        ("START PROCESS", 10, 15),
        ("LOAD ENGINE", 15, 99),
        ("EDITOR READY", 99, 100),
    )
    # Network layout of the ignition panel: nodes per layer.
    _LAYERS = (4, 6, 6, 3)

    def __init__(self, icon_path: str, project_name: str, parent=None, *, detail: str = ""):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setFixedSize(self._SPLASH_W, self._SPLASH_H)

        self._process: subprocess.Popen | None = None
        self._ready_file: str = ""
        self._project_path: str = ""
        self._lock_token: str = ""
        self._owns_launch_reservation = False
        self._angle = 0  # animation phase
        self._closing = False
        self._terminal_handled = False
        self._launch_started_at = 0.0
        self._launch_args = None
        self._project_name = project_name
        self._detail = detail
        self._log: list[str] = []
        self._failed = False
        app = QApplication.instance()
        self._palette = StyleManager.palette(bool(getattr(app, "is_dark_theme", True)))
        self._icon = QPixmap(icon_path) if icon_path else QPixmap()
        self._shown_at = time.monotonic()
        rng = random.Random(zlib.crc32(project_name.encode("utf-8")))
        self._stars = [(rng.random(), rng.random(), rng.choice((1, 1, 1, 2))) for _ in range(70)]

        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(
                geo.x() + (geo.width() - self._SPLASH_W) // 2,
                geo.y() + (geo.height() - self._SPLASH_H) // 2,
            )

        # Text widgets the launch logic updates; everything else is painted.
        self._status = _StatusLine(tr("Initializing engine..."), self)
        self._status.setObjectName("monoValue")
        self._status.setGeometry(336, 318, 380, 18)
        status_font = QFont()
        status_font.setFamilies(list(FONT_FAMILIES))
        status_font.setPixelSize(12)
        status_font.setWeight(QFont.Weight.Bold)
        status_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
        self._status.setFont(status_font)
        self._status.setStyleSheet(f"color: {self._palette.text_mono}; background: transparent;")
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        self._progress_bar.valueChanged.connect(lambda _value: self.update())

        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(self._tick_spinner)
        self._spin_timer.start(33)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_launch_state)

    def _log_status(self, text: str) -> None:
        if not self._log or self._log[-1] != text:
            self._log.append(text)
            del self._log[:-4]
        self.update()

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

    def _font(self, size: int, *, bold: bool = True, tracking: float = 0.0) -> QFont:
        font = QFont()
        font.setFamilies(list(FONT_FAMILIES))
        font.setPixelSize(size)
        font.setWeight(QFont.Weight.Bold if bold else QFont.Weight.Medium)
        if tracking:
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, tracking)
        return font

    def _node_positions(self, panel: QRectF) -> list[list[QPointF]]:
        layers = []
        count = len(self._LAYERS)
        for column, nodes in enumerate(self._LAYERS):
            x = panel.left() + 40 + column * (panel.width() - 80) / (count - 1)
            span = panel.height() - 150
            top = panel.top() + 74 + (span - (nodes - 1) * span / 5) / 2
            layers.append([QPointF(x, top + row * span / 5) for row in range(nodes)])
        return layers

    def paintEvent(self, event):
        palette = self._palette
        p = QPainter(self)
        rect = self.rect()
        p.fillRect(rect, QColor(palette.bg_base))
        progress = self._progress_bar.value() / 100.0
        accent = QColor(palette.danger if self._failed else palette.accent)
        fill = QColor(palette.danger if self._failed else palette.accent_fill)
        muted = QColor(palette.text_muted)

        # ── Ignition panel ──
        panel = QRectF(0, 0, 300, rect.height())
        p.fillRect(panel, QColor(palette.bg_deep))
        for x, y, size in self._stars:
            twinkle = 60 + int(50 * (0.5 + 0.5 * math.sin(self._angle * 0.05 + x * 40)))
            p.fillRect(QRectF(panel.left() + x * panel.width(), panel.top() + y * panel.height(), size, size),
                       qcolor(palette.grid, twinkle // 3))
        layers = self._node_positions(panel)
        lit_layers = progress * (len(layers) + 0.6)
        p.setRenderHint(QPainter.Antialiasing, True)
        for column in range(len(layers) - 1):
            live = lit_layers > column + 1
            for a_index, a in enumerate(layers[column]):
                for b_index, b in enumerate(layers[column + 1]):
                    if (a_index + b_index) % 2 and len(layers[column]) > 3:
                        continue
                    edge = qcolor(palette.accent_fill if live else palette.text_muted, 120 if live else 34)
                    p.setPen(QPen(edge, 1))
                    p.drawLine(a, b)
                    if live:
                        phase = ((self._angle * 0.018) + (a_index * 0.37) + (b_index * 0.21)) % 1.0
                        point = a + (b - a) * phase
                        p.fillRect(QRectF(point.x() - 1.5, point.y() - 1.5, 3, 3), accent)
        p.setRenderHint(QPainter.Antialiasing, False)
        for column, nodes in enumerate(layers):
            lit = lit_layers > column
            for index, node in enumerate(nodes):
                size = 9
                box = QRectF(node.x() - size / 2, node.y() - size / 2, size, size)
                p.fillRect(box, QColor(palette.bg_deep))
                if lit:
                    pulse = 0.55 + 0.45 * math.sin(self._angle * 0.12 + index + column * 1.7)
                    p.fillRect(box, mix(fill, accent, pulse))
                else:
                    p.setPen(QPen(qcolor(palette.text_muted, 120), 1))
                    p.drawRect(box)
        p.setFont(self._font(10, tracking=1.6))
        p.setPen(accent)
        p.drawText(QRectF(26, 26, 250, 16), Qt.AlignLeft | Qt.AlignVCenter, "3N  ·  IGNITION")
        p.setPen(muted)
        p.setFont(self._font(10, bold=False, tracking=0.8))
        p.drawText(QRectF(26, rect.height() - 50, 250, 16), Qt.AlignLeft | Qt.AlignVCenter,
                   "NEURAL NETWORK-NATIVE ENGINE")
        if not self._icon.isNull():
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            p.drawPixmap(QRectF(panel.right() - 50, 18, 30, 30).toRect(), self._icon)
        p.fillRect(QRectF(panel.right() - 1, 0, 1, rect.height()), QColor(palette.border))

        # ── Hazard band on the right column ──
        band = 6
        p.fillRect(QRectF(300, 0, rect.width() - 300, band), QColor(palette.bg_deep))
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        offset = (self._angle // 2) % (band * 2) if progress < 1.0 and not self._failed else 0
        for x in range(300 - band * 2 + offset, rect.width() + band, band * 2):
            p.drawPolygon(QPolygonF([QPointF(x, band), QPointF(x + band, 0),
                                     QPointF(x + band * 2, 0), QPointF(x + band, band)]))
        p.setRenderHint(QPainter.Antialiasing, False)

        left = 336
        p.setFont(self._font(10, tracking=1.8))
        p.setPen(accent)
        p.drawText(QRectF(left, 30, 260, 16), Qt.AlignLeft | Qt.AlignVCenter,
                   "FAULT  ·  LAUNCH ABORTED" if self._failed else "EDITOR LAUNCH SEQUENCE")
        elapsed = int(time.monotonic() - self._shown_at)
        p.setPen(muted)
        p.drawText(QRectF(rect.width() - 160, 30, 128, 16), Qt.AlignRight | Qt.AlignVCenter,
                   f"T+{elapsed // 60:02d}:{elapsed % 60:02d}")
        p.setPen(QColor(palette.text_primary))
        title_font = self._font(28)
        p.setFont(title_font)
        title = QFontMetrics(title_font).elidedText(self._project_name, Qt.ElideRight, rect.width() - left - 32)
        p.drawText(QRectF(left, 52, rect.width() - left - 32, 40), Qt.AlignLeft | Qt.AlignVCenter, title)
        if self._detail:
            p.setFont(self._font(11, bold=False, tracking=0.4))
            p.setPen(muted)
            detail = QFontMetrics(p.font()).elidedText(self._detail, Qt.ElideMiddle, rect.width() - left - 32)
            p.drawText(QRectF(left, 92, rect.width() - left - 32, 18), Qt.AlignLeft | Qt.AlignVCenter, detail)

        # Stage checklist.
        y = 132
        value = self._progress_bar.value()
        for number, (label, start, end) in enumerate(self._STAGES, start=1):
            done = value >= end or (label == "EDITOR READY" and value >= 100)
            active = start <= value < end and not done
            lamp = QRectF(left, y + 4, 8, 8)
            if self._failed and active:
                p.fillRect(lamp, QColor(palette.danger))
            elif done:
                p.fillRect(lamp, QColor(palette.signal))
            elif active:
                blink = (self._angle // 16) % 2 == 0
                p.fillRect(lamp, accent if blink else qcolor(palette.accent, 90))
            else:
                p.setPen(QPen(qcolor(palette.text_muted, 120), 1))
                p.drawRect(lamp)
            p.setFont(self._font(10, bold=False, tracking=0.6))
            p.setPen(muted)
            p.drawText(QRectF(left + 18, y, 26, 16), Qt.AlignLeft | Qt.AlignVCenter, f"{number:02d}")
            p.setFont(self._font(12, bold=active or done, tracking=1.0))
            p.setPen(QColor(palette.text_primary if (active or done) else palette.text_muted))
            p.drawText(QRectF(left + 46, y, 220, 16), Qt.AlignLeft | Qt.AlignVCenter, tr(label))
            p.setFont(self._font(10, tracking=1.0))
            state = "FAULT" if self._failed and active else "DONE" if done else "ACTIVE" if active else "--"
            p.setPen(QColor(palette.signal) if done else accent if active else muted)
            p.drawText(QRectF(rect.width() - 152, y, 120, 16), Qt.AlignRight | Qt.AlignVCenter, tr(state))
            y += 30

        # Telemetry tail: the previous status lines fade upward.
        p.setFont(self._font(10, bold=False, tracking=0.3))
        history = self._log[:-1][-2:]
        for index, line in enumerate(history):
            alpha = 90 + 60 * index
            p.setPen(qcolor(palette.text_muted, alpha))
            text = QFontMetrics(p.font()).elidedText("› " + line, Qt.ElideRight, rect.width() - left - 40)
            p.drawText(QRectF(left, 278 + index * 18, rect.width() - left - 40, 16),
                       Qt.AlignLeft | Qt.AlignVCenter, text)

        # Segmented meter.
        meter_y, meter_h = rect.height() - 58, 8
        right = rect.width() - 32
        pitch, segment = 7, 5
        count = int((right - left) // pitch)
        lit = round(value / 100 * count)
        head = (self._angle // 3) % (count + 8)
        off = qcolor(palette.text_muted, 40)
        for index in range(count):
            if index < lit:
                color = fill
            elif head - 5 <= index < head and value < 100 and not self._failed:
                color = qcolor(palette.accent, 110 + 25 * (index - head + 5))
            else:
                color = off
            p.fillRect(QRectF(left + index * pitch, meter_y, segment, meter_h), color)
        p.setFont(self._font(10, tracking=1.0))
        p.setPen(muted)
        p.drawText(QRectF(left, meter_y + 14, 200, 16), Qt.AlignLeft | Qt.AlignVCenter, f"{value:03d}%")
        p.drawText(QRectF(right - 220, meter_y + 14, 220, 16), Qt.AlignRight | Qt.AlignVCenter,
                   "INFERNUX  ·  " + ("ESC TO HIDE" if not self._failed else "SEE DETAILS"))
        p.setPen(QPen(QColor(palette.border_hover), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(rect.adjusted(0, 0, -1, -1))
        p.end()

    def keyPressEvent(self, event):
        # Escape hides the plate; launch monitoring continues in the background.
        if event.key() == Qt.Key_Escape and not self._terminal_handled:
            self.hide()
            return
        super().keyPressEvent(event)

    def _tick_spinner(self):
        self._angle = (self._angle + 1) % 100000
        self.update()

    def set_preparation_status(self, message: str, progress: int = 5):
        """Update the visible pre-launch phase from the Hub worker."""
        self._status.setText(tr(message))
        self._progress_bar.setValue(max(0, min(int(progress), 10)))

    def show_preparation_failure(self, detail: str):
        """Report a failure that happened before an engine process existed."""
        if self._terminal_handled:
            return
        self._terminal_handled = True
        self._failed = True
        self._status.setText(tr("Launch failed"))
        summary, _, rest = detail.partition("\n")
        dialogs.critical(self, tr("Engine Launch Failed"), summary,
                         detail=detail if rest.strip() else "", kicker=tr("EDITOR LAUNCH"))
        self._fade_out_and_close()

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
               *, detached: bool = True, extra_env: dict[str, str] | None = None,
               lock_token: str | None = None, runtime: EditorPythonRuntime | None = None):
        """Start the engine without blocking the UI and monitor readiness."""
        self._launch_args = (python_exe, script, project_path, detached, extra_env, runtime)
        self._terminal_handled = False
        self._closing = False
        self._poll_timer.stop()
        self._cleanup_ready_file()
        self._ready_file = os.path.join(
            tempfile.gettempdir(), f"infernux_ready_{uuid.uuid4().hex}.flag"
        )
        self._project_path = project_path
        self._lock_token = lock_token or uuid.uuid4().hex
        self._owns_launch_reservation = False
        self._process = None
        self._stderr_chunks = []
        self._status.setText(tr("Checking project..."))
        self._progress_bar.setValue(5)
        env = os.environ.copy()
        env["_INFERNUX_READY_FILE"] = self._ready_file
        if extra_env:
            env.update(extra_env)
        if runtime is not None:
            env.update(runtime.environment)
        env["_INFERNUX_PROJECT_LOCK_PATH"] = get_project_lock_path(project_path)
        env["_INFERNUX_PROJECT_LOCK_TOKEN"] = self._lock_token
        env = merge_child_env_utf8(env)

        popen_kwargs: dict = {"cwd": runtime.working_directory if runtime else project_path, "env": env}
        executable = runtime.executable if runtime else python_executable_path(python_exe)
        if runtime is not None:
            script = runtime.bootstrap(script, project_path)

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

        self._status.setText(tr("Starting engine process..."))
        self._progress_bar.setValue(10)
        try:
            write_project_lock(project_path, os.getpid(), self._lock_token, "editor", "preparing")
            self._owns_launch_reservation = True
            with _suppress_windows_error_dialogs():
                self._process = subprocess.Popen(
                    [executable, "-u", "-c", script, project_path],
                    executable=executable,
                    **popen_kwargs,
                )
        except (OSError, RuntimeError, ValueError) as exc:
            if self._owns_launch_reservation:
                remove_project_lock(project_path, self._lock_token)
            self._status.setText(tr("Launch failed"))
            detail = f"The engine process could not be started.\n\n{exc}"
            QTimer.singleShot(
                0,
                lambda: self._show_failure(
                    tr("Engine Launch Failed"),
                    detail,
                ),
            )
            return

        # Own the pipe and child immediately, including a failed handoff.
        # Bind this launch's objects before an explicit Retry can replace them.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._process, self._stderr_chunks),
            daemon=True,
        )
        self._stderr_thread.start()

        # The lock must refer to the engine process, not the still-running Hub.
        try:
            write_project_lock(
                project_path, self._process.pid, self._lock_token, "editor", "launching",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self._stop_process()
            detail = f"The project launch reservation could not be transferred.\n\n{exc}"
            QTimer.singleShot(0, lambda: self._show_failure(tr("Engine Launch Failed"), detail))
            return

        self._launch_started_at = time.monotonic()
        self._status.setText(tr("Waiting for the editor..."))
        self._progress_bar.setValue(15)
        self._poll_timer.start(150)

    def _drain_stderr(self, proc, chunks):
        """Drain the engine pipe, then reap it even after the splash has closed."""
        if proc is None or proc.stderr is None:
            return
        try:
            with proc.stderr:
                while True:
                    chunk = proc.stderr.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
        finally:
            proc.wait()

    def _poll_launch_state(self):
        if self._terminal_handled:
            return
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
                self._status.setText(tr("Ready"))
                self._progress_bar.setValue(100)
                self._fade_out_and_close()
                return

            if content.startswith("ERROR:"):
                self._show_failure(
                    tr("Engine Launch Failed"),
                    content.partition(":")[2].strip() or "The editor reported a startup error.",
                )
                return

            if content.startswith("LOADING:"):
                try:
                    _, fraction, message = content.split(":", 2)
                    self._status.setText(tr(message))
                    current, total = fraction.split("/")
                    self._progress_bar.setValue(
                        int(int(current) * 100 / int(total))
                    )
                except (ValueError, ZeroDivisionError) as _exc:
                    logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
                    pass

        if self._process is not None and self._process.poll() is not None:
            stderr_text = b"".join(
                getattr(self, '_stderr_chunks', [])
            ).decode("utf-8", errors="replace").strip()
            if stderr_text:
                detail = (
                    "The engine process exited with an error:\n\n"
                    f"{stderr_text[:4000]}"
                )
            else:
                detail = (
                    "The engine process exited unexpectedly "
                    f"before the editor finished loading.\n\n"
                    f"{_format_exit_code(self._process.returncode)}\n\n"
                    "Check the project's Logs/ folder for details."
                )
            self._show_failure(tr("Engine Launch Failed"), detail)
            return

        elapsed = time.monotonic() - self._launch_started_at
        if elapsed >= self._STARTUP_TIMEOUT_SECONDS:
            self._show_timeout()

    def _show_timeout(self):
        if self._terminal_handled:
            return
        self._terminal_handled = True
        self._poll_timer.stop()
        box = dialogs.HubDialog(
            self, level="warning", title=tr("Engine Launch Timed Out"),
            text=tr("The editor did not become ready within {seconds} seconds.",
                    seconds=self._STARTUP_TIMEOUT_SECONDS),
            kicker=tr("EDITOR LAUNCH"),
        )
        stop = box.addButton(tr("Stop"), "reject")
        open_logs = box.addButton(tr("Open Logs"), "action")
        keep_waiting = box.addButton(tr("Keep Waiting"), "action")
        retry = box.addButton(tr("Retry"), "accept")
        box.setDefaultButton(keep_waiting)
        box.exec()
        clicked = box.clickedButton()
        if clicked is retry:
            self._retry_launch()
        elif clicked is stop:
            self._stop_process()
            self._fade_out_and_close()
        else:
            if clicked is open_logs:
                self._open_logs()
            self._terminal_handled = False
            self._launch_started_at = time.monotonic()
            self._poll_timer.start(150)

    def _show_failure(self, title: str, detail: str):
        if self._terminal_handled:
            return
        self._terminal_handled = True
        self._failed = True
        self._poll_timer.stop()
        self._status.setText(tr("Launch failed"))
        if self._owns_launch_reservation:
            remove_project_lock(self._project_path, self._lock_token)
        box = dialogs.HubDialog(
            self, level="critical", title=title, text=detail.split("\n", 1)[0], detail=detail,
            kicker=tr("EDITOR LAUNCH"),
        )
        close = box.addButton(tr("Stop"), "reject")
        open_logs = box.addButton(tr("Open Logs"), "action")
        retry = box.addButton(tr("Retry"), "accept")
        box.setDefaultButton(retry)
        box.exec()
        clicked = box.clickedButton()
        if clicked is retry:
            self._failed = False
            self._retry_launch()
            return
        if clicked is open_logs:
            self._open_logs()
        elif clicked is close:
            pass
        self._fade_out_and_close()

    def _retry_launch(self):
        args = self._launch_args
        self._stop_process()
        if args is None:
            self._fade_out_and_close()
            return
        deadline = time.monotonic() + 5.0

        def launch_after_exit():
            if self._process is not None and self._process.poll() is None:
                if time.monotonic() >= deadline:
                    self._terminal_handled = False
                    self._show_failure(tr("Engine Launch Failed"), "The previous editor process has not stopped.")
                    return
                QTimer.singleShot(50, launch_after_exit)
                return
            if self._owns_launch_reservation:
                remove_project_lock(self._project_path, self._lock_token)
            python_exe, script, project_path, detached, extra_env, runtime = args
            self.launch(python_exe, script, project_path, detached=detached, extra_env=extra_env, runtime=runtime)

        launch_after_exit()

    def _stop_process(self):
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
        if self._project_path and self._owns_launch_reservation:
            remove_project_lock(self._project_path, self._lock_token)

    def _open_logs(self):
        if not self._project_path:
            return
        logs_dir = os.path.join(self._project_path, "Logs")
        os.makedirs(logs_dir, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(logs_dir))

    def _cleanup_ready_file(self):
        if not self._ready_file:
            return
        try:
            os.remove(self._ready_file)
        except OSError:
            pass
        self._ready_file = ""

    def close(self):
        self._fade_out_and_close()

    def _finish_close(self):
        if (self._process is not None and self._process.poll() is not None
                and self._project_path and self._owns_launch_reservation):
            remove_project_lock(self._project_path, self._lock_token)
        self._cleanup_ready_file()
        self.hide()
        super().close()
        self.deleteLater()
