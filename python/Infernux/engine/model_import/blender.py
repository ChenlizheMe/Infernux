"""Single Blender source-conversion backend for the model importer.

The caller supplies the managed toolchain executable and a project Library
destination. GLB is an importer intermediate, never the published Player asset.
No PATH probing, downloads, Assimp .blend fallback, or source-file rewrites.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile

from Infernux.engine.path_utils import resolved_path


class BlenderImportError(RuntimeError):
    pass


def convert_blend(source: str | Path, destination: str | Path, *,
                  blender_executable: str | Path, timeout: float = 120.0) -> Path:
    """Convert with the supported Blender 5.2 toolchain, then publish atomically.

    Failure leaves the previous derived file intact and is reported to the
    importer; retaining that file does not turn a failed import into success.
    """
    source = Path(resolved_path(source))
    executable = Path(resolved_path(blender_executable))
    destination = Path(resolved_path(destination))
    if source.suffix.lower() != ".blend" or not source.is_file():
        raise ValueError("Blender source must be a .blend file")
    if not executable.is_file():
        raise ValueError("Blender toolchain executable must be a file")
    if destination.suffix.lower() != ".glb":
        raise ValueError("Blender intermediate must use the .glb extension")
    if timeout <= 0:
        raise ValueError("Blender import timeout must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).with_name("_blender_export.py")
    with tempfile.TemporaryDirectory(prefix=".blend-import-", dir=destination.parent) as staging:
        output = Path(staging) / "model.glb"
        command = [str(executable), "--background", "--factory-startup", "--disable-autoexec",
                   "--python-exit-code", "1", "--python", str(helper), "--", str(source), str(output)]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except subprocess.TimeoutExpired as exc:
            raise BlenderImportError(f"Blender import timed out after {timeout:g}s: {source}") from exc
        if result.returncode:
            detail = (result.stdout + "\n" + result.stderr).strip()[-8000:]
            raise BlenderImportError(f"Blender import failed ({result.returncode}): {source}\n{detail}")
        if not output.is_file():
            raise BlenderImportError(f"Blender produced no model intermediate: {source}")
        os.replace(output, destination)
    return destination
