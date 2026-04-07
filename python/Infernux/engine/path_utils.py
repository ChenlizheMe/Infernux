"""
path_utils — helpers for cross-language path safety.

The C++ engine core uses ``std::string`` (narrow) for file-system paths.
On Windows, if a path contains characters outside the active ANSI code-page
(e.g. Chinese / Japanese usernames), the narrow-string conversion raises
"No mapping for the Unicode character exists in the target multi-byte code
page".

For **standalone builds** the Nuitka EXE ships with an application manifest
that sets ``<activeCodePage>UTF-8</activeCodePage>`` — this is the primary
fix and requires no runtime workaround.

For the **editor** (no manifest), ``safe_path()`` tries two fall-backs:
  1. Windows 8.3 short-path (ASCII-only, needs NTFS 8.3 generation enabled)
  2. Original path unchanged (works if the process codepage can represent it)
"""

import os
import sys

from Infernux.debug import Debug


def safe_path(path: str) -> str:
    """Return an ASCII-safe version of *path* for consumption by C++.

    On non-Windows, or when the path is already ASCII-clean, returns *path*
    unchanged.  On Windows, uses ``GetShortPathNameW`` to obtain the 8.3
    short-name form.  Falls back to the original path if conversion fails.
    """
    if sys.platform != "win32" or not path:
        return path
    # Fast path: skip if already ASCII-safe
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 32768)
        if n and n < 32768:
            short = buf.value
            # Only use if genuinely different (8.3 may be disabled)
            if short != path:
                return short
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    return path
