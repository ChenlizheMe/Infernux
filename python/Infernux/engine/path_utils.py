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


def _process_codepage_is_utf8() -> bool:
    """Return True when the process ANSI code page is UTF-8 (65001).

    This is the case for standalone builds that embed a UTF-8 manifest
    (``<activeCodePage>UTF-8</activeCodePage>``).  When the ACP is
    UTF-8, narrow-string Windows APIs accept UTF-8 directly, so *no*
    path conversion is needed — Chinese / Japanese characters work as-is.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        return ctypes.windll.kernel32.GetACP() == 65001
    except Exception:
        return False


_UTF8_ACP: bool = _process_codepage_is_utf8()


def safe_path(path: str) -> str:
    """Return a version of *path* safe for the C++ engine's narrow-string APIs.

    * If the process code page is UTF-8 (standalone builds with manifest),
      returns *path* unchanged — no conversion needed.
    * Otherwise, on Windows, tries the 8.3 short-path fallback.
    * Falls back to the original path if all else fails.
    """
    if sys.platform != "win32" or not path:
        return path
    # Standalone build with UTF-8 manifest → pass through unchanged
    if _UTF8_ACP:
        return path
    # Fast path: skip if already ASCII-safe
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
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
