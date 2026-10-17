"""Stream package/source downloads and publish the completed file atomically."""

from __future__ import annotations

import os
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Mapping

from infernux.version import ENGINE_VERSION


def _size_text(received: int, total: int) -> str:
    def size(value: int) -> str:
        amount = float(value)
        for unit in ("B", "KiB", "MiB"):
            if amount < 1024.0:
                return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
            amount /= 1024.0
        return f"{amount:.1f} GiB"

    return f"{size(received)} / {size(total)}" if total else size(received)


def download_file(
    url: str,
    destination: str | Path,
    *,
    progress: Callable[[str, float, str], None] | None = None,
    phase: str = "download_package",
    start: float = 0.08,
    end: float = 0.30,
    headers: Mapping[str, str] | None = None,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"infernux/{ENGINE_VERSION}", **(headers or {})},
    )
    partial = Path(f"{destination}.{uuid.uuid4().hex}.part")
    try:
        with urllib.request.urlopen(request, timeout=30) as response, partial.open("wb") as stream:
            total = int(response.headers.get("Content-Length", "0") or 0)
            received = 0
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if progress is not None:
                    fraction = min(1.0, received / total) if total else 0.0
                    progress(phase, start + (end - start) * fraction, _size_text(received, total))
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
