"""Independent desktop progress presentation for synchronous Editor work.

The helper is a presentation-only child process. Its lifecycle is a real
boundary: a missing executable or a helper that exits early must degrade to
"no progress window", never abort Enter Play Mode or raise into the frame.
"""

import os
from pathlib import Path
import subprocess
import sys


class BusyProgress:
    def __init__(self):
        from infernux.lib import native_dir

        executable = Path(native_dir) / (
            "InfernuxEditorProgress.exe" if sys.platform == "win32"
            else "InfernuxEditorProgress"
        )
        self._process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def _disable(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass

    def update(self, progress: float, message: str) -> None:
        process = self._process
        if process is None or process.stdin is None:
            return
        try:
            process.stdin.write(f"{progress}\t{message}\n")
            process.stdin.flush()
        except OSError:
            # The helper exited (crash, user closed it, permission change).
            # Progress reporting is best-effort from this point on.
            self._disable()

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
