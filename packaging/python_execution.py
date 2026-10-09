"""Windows loader paths for Hub-owned Python runtimes.

Extended paths belong at the process/import boundary, never in project assets.
The private runtime's startup hook applies them to native extension loaders;
a long-path-aware executable alone is insufficient. Python search paths and
package metadata retain their ordinary spelling (including pip's RECORD paths).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def python_executable_path(path: str) -> str:
    if sys.platform != "win32":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return absolute
    return "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute


# Executed by site before third-party .pth startup hooks. Keep PathFinder's
# normal search/loader protocol, changing only the Windows DLL loader path.
# No machine path is recorded, so a copied/moved runtime uses its new location.
_WINDOWS_PATHS_BOOTSTRAP = r'''
import os
import sys
from importlib.machinery import ExtensionFileLoader, PathFinder


def extended(path):
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return absolute
    return "\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute


class WindowsPathFinder(PathFinder):
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        spec = super().find_spec(fullname, path, target)
        if spec is not None and isinstance(spec.loader, ExtensionFileLoader):
            spec.origin = extended(spec.origin)
            spec.loader.path = spec.origin
        return spec


def install():
    if sys.platform != "win32":
        return
    for index, finder in enumerate(sys.meta_path):
        if finder is PathFinder:
            sys.meta_path[index] = WindowsPathFinder
            break
    sys.executable = extended(sys.executable)
'''.lstrip()

_WINDOWS_PATHS_PTH = (
    "# Infernux private runtime: retain Win32 extended paths for DLL imports.\n"
    "import _infernux_runtime_paths; _infernux_runtime_paths.install()\n"
)


def prepare_private_runtime_paths(root: str | os.PathLike[str]) -> None:
    """Install deterministic startup support into a Hub-owned runtime only."""
    if sys.platform != "win32":
        return
    site = Path(root) / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    for name, content in (("_infernux_runtime_paths.py", _WINDOWS_PATHS_BOOTSTRAP),
                          ("000_infernux_windows_paths.pth", _WINDOWS_PATHS_PTH)):
        target = site / name
        if not target.is_file() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8", newline="\n")
