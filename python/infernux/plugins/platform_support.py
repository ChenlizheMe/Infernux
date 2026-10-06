"""Project-independent platform support prerequisites owned by Infernux Hub."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Mapping

from infernux.core.file_read_cache import FileReadCache
from infernux.engine.path_utils import lexical_path, resolved_path


ANDROID_PLUGIN_REFERENCE = "infernux/platform-android"
ANDROID_SUPPORT_SCHEMA = "infernux.android_support"
ANDROID_SUPPORT_KIND = "infernux-android-support"
ANDROID_SUPPORT_VERSION = "0.1.0"
ANDROID_SUPPORT_MANIFEST = "infernux-android-support.json"
ANDROID_SUPPORT_REQUIRED_MESSAGE = (
    "Install Android compatibility from Infernux Hub before importing the "
    "Android platform plugin."
)
_manifest_reads = FileReadCache(capacity=8)


@cache
def _host_id() -> str:
    """The host architecture is immutable for the lifetime of this process.

    Do not use platform.machine here: on CPython/Windows its first call also
    queries OS details through WMI, blocking the first visible plugin panel.
    """
    if os.name == "nt":
        # IsWow64Process2 and this platform-kit host check require Windows 10
        # 1709+. This kit probe does not advertise support on older hosts.
        if sys.getwindowsversion() < (10, 0, 16299):
            return ""
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        query = kernel32.IsWow64Process2
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.USHORT), ctypes.POINTER(wintypes.USHORT)]
        query.restype = wintypes.BOOL
        process_machine, native_machine = wintypes.USHORT(), wintypes.USHORT()
        # -1 is the current-process pseudo handle; it is not owned or closed.
        if not query(wintypes.HANDLE(-1), ctypes.byref(process_machine), ctypes.byref(native_machine)):
            raise ctypes.WinError(ctypes.get_last_error())
        # Inspect the host, not a possibly emulated process architecture.
        return "windows-x64" if native_machine.value == 0x8664 else ""
    if os.name == "posix":
        host = os.uname()
        if host.sysname == "Linux" and host.machine.casefold() in {"amd64", "x86_64"}:
            return "linux-x64"
    return ""


def android_support_root(
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    location = _android_support_location(environ)
    return Path(resolved_path(location)) if location is not None else None


def _android_support_location(environ: Mapping[str, str] | None) -> Path | None:
    values = os.environ if environ is None else environ
    explicit = str(values.get("INFERNUX_ANDROID_SUPPORT_ROOT", "") or "").strip()
    if explicit:
        return Path(lexical_path(os.path.expandvars(os.path.expanduser(explicit))))
    data_root = str(values.get("INFERNUX_SHARED_DATA_ROOT", "") or "").strip()
    if not data_root:
        data_root = str(values.get("INFERNUX_DATA_ROOT", "") or "").strip()
    if data_root:
        base = Path(lexical_path(os.path.expandvars(os.path.expanduser(data_root))))
    elif os.name == "nt":
        local_app_data = str(values.get("LOCALAPPDATA", "") or "").strip()
        if not local_app_data:
            return None
        base = Path(lexical_path(local_app_data)) / "InfernuxHub"
    else:
        xdg_data = str(values.get("XDG_DATA_HOME", "") or "").strip()
        base = (
            Path(lexical_path(Path(xdg_data).expanduser())) / "InfernuxHub"
            if xdg_data
            else Path.home() / ".local/share/InfernuxHub"
        )
    host = _host_id()
    if not host:
        return None
    return base / "PlatformKits/android" / ANDROID_SUPPORT_VERSION / host


def _relative(value: object) -> Path:
    text = str(value or "").replace("\\", "/").strip("/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError("Android compatibility manifest contains an unsafe path")
    return Path(*path.parts)


@dataclass(frozen=True, slots=True)
class _SupportLayout:
    sdk: Path
    jdk: Path
    gradle: Path
    arm64: Path
    x64: Path
    directories: tuple[Path, ...]
    files: tuple[Path, ...]

    def available(self) -> bool:
        # The prepared layout is immutable; presence is always current, even
        # inside a panel's read scope. Stop at the first missing prerequisite.
        return all(path.is_dir() for path in self.directories) and all(
            path.is_file() for path in self.files
        )


def _support_layout(root: Path) -> _SupportLayout | None:
    manifest = root / ANDROID_SUPPORT_MANIFEST
    host = _host_id()
    if not host:
        return None

    def prepare(observed):
        observed.watch(manifest)
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if type(document) is not dict:
            return None
        paths = document.get("paths")
        if (
            document.get("$schema") != ANDROID_SUPPORT_SCHEMA
            or document.get("kind") != ANDROID_SUPPORT_KIND
            or document.get("version") != ANDROID_SUPPORT_VERSION
            or document.get("host") != host
            or type(paths) is not dict
            or set(paths) != {"sdk", "jdk", "gradle", "python"}
        ):
            return None
        python_paths = paths["python"]
        if type(python_paths) is not dict or set(python_paths) != {"arm64-v8a", "x86_64"}:
            return None
        sdk = root / _relative(paths["sdk"])
        jdk = root / _relative(paths["jdk"])
        gradle = root / _relative(paths["gradle"])
        arm64 = root / _relative(python_paths["arm64-v8a"])
        x64 = root / _relative(python_paths["x86_64"])
        executable = ".exe" if os.name == "nt" else ""
        gradle_name = "gradle.bat" if os.name == "nt" else "gradle"
        return _SupportLayout(
            sdk, jdk, gradle, arm64, x64,
            directories=(sdk / "platforms/android-36", sdk / "build-tools/36.0.0"),
            files=(
                sdk / "ndk/29.0.14206865/build/cmake/android.toolchain.cmake",
                sdk / "platform-tools" / f"adb{executable}",
                jdk / "bin" / f"java{executable}",
                gradle / "bin" / gradle_name,
                arm64 / "infernux-android-python.json",
                x64 / "infernux-android-python.json",
            ),
        )

    # Commands also use this query, so never reuse a presentation snapshot.
    # Cache schema validation and path construction, not installation readiness.
    try:
        return _manifest_reads.get((str(manifest), host), prepare, current=True)
    except (OSError, ValueError):
        return None


def android_support_available(
    environ: Mapping[str, str] | None = None,
) -> bool:
    root = _android_support_location(environ)
    if root is None:
        return False
    layout = _support_layout(root)
    return layout is not None and layout.available()


def android_support_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    root = _android_support_location(environ)
    if root is None:
        return {}
    layout = _support_layout(root)
    if layout is None or not layout.available():
        return {}
    # Readiness and environment must come from the same validated manifest.
    # A second raw read here could publish paths from an unvalidated revision.
    location = root
    root = Path(resolved_path(location))

    def environment_path(path: Path) -> str:
        return str(root / path.relative_to(location))

    return {
        "INFERNUX_ANDROID_SUPPORT_ROOT": str(root),
        "ANDROID_SDK_ROOT": environment_path(layout.sdk),
        "ANDROID_HOME": environment_path(layout.sdk),
        "JAVA_HOME": environment_path(layout.jdk),
        "INFERNUX_GRADLE_HOME": environment_path(layout.gradle),
        "INFERNUX_ANDROID_PYTHON_PREFIX_ARM64": environment_path(layout.arm64),
        "INFERNUX_ANDROID_PYTHON_PREFIX_X86_64": environment_path(layout.x64),
    }


def activate_android_support_environment() -> bool:
    environment = android_support_environment()
    if not environment:
        return False
    os.environ.update(environment)
    return True


def plugin_install_block_reason(
    reference: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    if str(reference or "").strip().casefold() != ANDROID_PLUGIN_REFERENCE:
        return ""
    return "" if android_support_available(environ) else ANDROID_SUPPORT_REQUIRED_MESSAGE


def require_plugin_support(
    reference: str,
    environ: Mapping[str, str] | None = None,
) -> None:
    reason = plugin_install_block_reason(reference, environ)
    if reason:
        raise RuntimeError(reason)


__all__ = [
    "ANDROID_PLUGIN_REFERENCE",
    "ANDROID_SUPPORT_REQUIRED_MESSAGE",
    "android_support_available",
    "android_support_environment",
    "android_support_root",
    "activate_android_support_environment",
    "plugin_install_block_reason",
    "require_plugin_support",
]
