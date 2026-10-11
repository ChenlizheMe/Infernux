"""Read-only discovery of optional local test tools; never install or download."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def _editor_blender():
    from infernux.engine.model_import.toolchain import get_blender_executable
    return get_blender_executable()


def find_executable(name: str) -> str | None:
    """Find a test prerequisite in this Python environment or a normal install."""
    suffix = ".exe" if sys.platform == "win32" else ""
    binary = name + suffix
    prefix = Path(sys.prefix)
    candidates = [prefix / binary, prefix / "Scripts" / binary,
                  prefix / "Library/bin" / binary, prefix / "bin" / binary]
    on_path = shutil.which(name)
    if on_path:
        candidates.append(Path(on_path))
    if sys.platform == "win32":
        program_files = Path(os.environ.get("ProgramFiles") or "C:/Program Files")
        relative = {"cmake": "CMake/bin/cmake.exe", "git": "Git/cmd/git.exe",
                    "pwsh": "PowerShell/7/pwsh.exe", "bash": "Git/bin/bash.exe"}.get(name)
        if relative:
            candidates.append(program_files / relative)
        if name == "powershell":
            candidates.append(Path(os.environ.get("SystemRoot") or "C:/Windows") /
                              "System32/WindowsPowerShell/v1.0/powershell.exe")
    else:
        candidates.extend(Path(root) / binary for root in ("/usr/local/bin", "/usr/bin", "/opt/homebrew/bin"))
    for candidate in candidates:
        executable = shutil.which(str(candidate))
        if executable:
            return str(Path(executable).resolve())
    return None


def blender_candidates():
    # Share the Editor's explicit preference, Hub tool and .blend association.

    configured = _editor_blender()
    if configured:
        yield Path(configured)
    on_path = find_executable("blender")
    if on_path:
        yield Path(on_path)

    if sys.platform == "win32":
        roots = {Path(os.environ.get(name) or default) / "Blender Foundation"
                 for name, default in (("ProgramW6432", "C:/Program Files"),
                                       ("ProgramFiles", "C:/Program Files"),
                                       ("ProgramFiles(x86)", "C:/Program Files (x86)"))}
        for root in sorted(roots):
            yield from sorted(root.glob("Blender */blender.exe"), reverse=True)
    elif sys.platform == "darwin":
        for root in (Path("/Applications"), Path.home() / "Applications"):
            yield from sorted(root.glob("Blender*.app/Contents/MacOS/Blender"), reverse=True)
    else:
        yield Path("/usr/local/bin/blender")
        yield Path("/usr/bin/blender")
        yield from sorted(Path("/opt").glob("blender*/blender"), reverse=True)

    # Source Hub installations are discoverable even when tests aren't launched
    # through Hub. The configured shared root is a product setting, not a test flag.
    shared = Path(os.environ.get("INFERNUX_SHARED_DATA_ROOT") or
                  Path(__file__).resolve().parents[1] / "packaging/InfernuxHubData/Shared")
    executable = "blender.exe" if sys.platform == "win32" else "blender"
    yield from sorted((shared / "AuthoringTools/Blender").glob(f"*/*/{executable}"), reverse=True)


def find_blender(candidates=None) -> str:
    """Find the 5.2 series required by our exporter, checking each binary once."""
    seen, rejected = set(), []
    for candidate in blender_candidates() if candidates is None else candidates:
        path = Path(candidate).expanduser().resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            result = subprocess.run(
                [str(path), "--version"], stdin=subprocess.DEVNULL,
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            version = re.search(r"^Blender (\d+)\.(\d+)(?:\.\d+)?\b", result.stdout, re.MULTILINE)
            if result.returncode == 0 and version and version.groups() == ("5", "2"):
                return str(path)
            detail = result.stdout.strip().splitlines()[:1] or result.stderr.strip().splitlines()[:1]
            rejected.append(f"{path}: exit {result.returncode}, {' '.join(detail) or 'no version output'}")
        except (OSError, subprocess.SubprocessError) as error:
            rejected.append(f"{path}: {error}")
    detail = "; ".join(rejected) or "no local Blender executable found"
    raise FileNotFoundError(
        "Blender 5.2 is required by the model exporter. Install it in a standard location, "
        "put Blender on PATH, or select it in Editor preferences. " + detail
    )
