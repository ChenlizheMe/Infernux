"""Discover the native Blender associated with .blend; never execute a file handler."""
from __future__ import annotations

import configparser
import ctypes
import logging
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


def _windows_executable() -> str:
    from ctypes import wintypes

    query = ctypes.WinDLL("shlwapi").AssocQueryStringW
    query.argtypes = [wintypes.DWORD, ctypes.c_int, wintypes.LPCWSTR,
                      wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    query.restype = ctypes.c_long
    size = wintypes.DWORD()
    # ASSOCSTR_EXECUTABLE honours the user's default application, including UserChoice.
    if query(0, 2, ".blend", "open", None, ctypes.byref(size)) != 1 or not size.value:
        return ""
    buffer = ctypes.create_unicode_buffer(size.value)
    return buffer.value if query(0, 2, ".blend", "open", buffer, ctypes.byref(size)) == 0 else ""


def _desktop_executable(path: Path) -> str:
    document = configparser.ConfigParser(interpolation=None, strict=False)
    document.read(path, encoding="utf-8")
    entry = document["Desktop Entry"]
    if entry.get("Type") != "Application" or entry.getboolean("Hidden", fallback=False):
        return ""
    # Decode desktop-entry string escapes before Exec quoting. Do not invoke a shell
    # or turn arbitrary launcher arguments into Blender command-line arguments.
    command = entry.get("Exec", "")
    for escaped, value in ((r"\s", " "), (r"\n", "\n"), (r"\t", "\t"), (r"\r", "\r")):
        command = command.replace(escaped, value)
    command = command.replace(r"\\", "\\")
    args = shlex.split(command)
    if not args:
        return ""
    executable = args[0].replace("%%", "%")
    if any(arg not in {"%f", "%F", "%u", "%U"} for arg in args[1:]):
        logging.getLogger(__name__).warning(
            "Blender association requires a launcher or options (%s); select a native Blender executable in Preferences", path)
        return ""
    return shutil.which(executable) or ""


def _linux_executable() -> str:
    query = shutil.which("xdg-mime")
    if not query:
        return ""
    result = subprocess.run([query, "query", "default", "application/x-blender"],
                            capture_output=True, text=True, timeout=5, check=True)
    desktop_id = result.stdout.strip()
    if not desktop_id.endswith(".desktop") or "/" in desktop_id or "\\" in desktop_id:
        return ""
    roots = [Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")]
    roots.extend(Path(part) for part in (os.environ.get("XDG_DATA_DIRS") or
                                        "/usr/local/share:/usr/share").split(":") if part)
    for root in roots:
        applications = root / "applications"
        direct = applications / desktop_id
        if direct.is_file():
            return _desktop_executable(direct)
        # Desktop IDs flatten subdirectory separators to '-'. Respect XDG precedence.
        for candidate in sorted(applications.glob("**/*.desktop")):
            if candidate.relative_to(applications).as_posix().replace("/", "-") == desktop_id:
                return _desktop_executable(candidate)
    return ""


def find_associated_blender() -> str:
    try:
        executable = (_windows_executable() if sys.platform == "win32" else
                      _linux_executable() if sys.platform.startswith("linux") else "")
        if executable:
            path = Path(executable)
            # A .blend handler may also be an asset browser. Only use native Blender.
            if path.name.casefold() in {"blender", "blender.exe"} and path.is_file():
                return str(path.resolve())
    except (OSError, ValueError, KeyError, configparser.Error, subprocess.SubprocessError):
        logging.getLogger(__name__).warning("Could not resolve the .blend default application", exc_info=True)
    return ""
