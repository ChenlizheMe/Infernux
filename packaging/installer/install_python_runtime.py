from __future__ import annotations

import argparse
import csv
import ctypes
import json
import logging
import os
import sys
import subprocess
from pathlib import Path


_PACKAGING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGING_DIR not in sys.path:
    sys.path.insert(0, _PACKAGING_DIR)

from embed_runtime_manager import PythonRuntimeManager
from hub_utils import get_hub_shared_data_dir
from python_runtime_catalog import DEFAULT_PYTHON_RUNTIME
from installer_safety import (
    WINDOWS_ACCESS_LAYOUT, install_marker_path, is_marked_install_dir, write_install_marker,
)
from publication_directory import clear_windows_readonly_tree, restore_windows_inherited_access_tree
from python_execution import windows_native_path


def _show_message_box(title: str, message: str, icon: int = 0x40) -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.MessageBoxW(None, message, title, icon)
    except Exception as _exc:
        logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
        pass


def prepare_shared_storage(app_dir: str) -> str:
    """Grant the installing account access to Shared, never to Hub program files."""
    shared = Path(get_hub_shared_data_dir(str(Path(app_dir).resolve())))
    if shared.resolve() != shared:
        raise RuntimeError("Installer shared storage must not redirect to another directory")
    shared.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        system = Path(os.environ["SystemRoot"]) / "System32"
        options = dict(
            check=True, capture_output=True, text=True, encoding="utf-8",
            errors="replace", creationflags=0x08000000,
        )
        identity = subprocess.run(
            [str(system / "whoami.exe"), "/user", "/fo", "csv", "/nh"], **options,
        )
        sid = next(csv.reader(identity.stdout.splitlines()))[1]
        subprocess.run(
            [str(system / "icacls.exe"), windows_native_path(str(shared)), "/grant", f"*{sid}:(OI)(CI)M"],
            **options,
        )
        restore_windows_inherited_access_tree(shared)
        clear_windows_readonly_tree(shared)
    return str(shared)


def repair_legacy_windows_permissions(app_dir: str) -> bool:
    """Repair legacy staging ACLs once after an elevated in-place update.

    The old updater restarts the new Hub elevated. Use that existing privilege
    to restore inherited application access and the user's Shared write access;
    subsequent ordinary launches do not rewrite ACLs or request elevation.
    """
    if sys.platform != "win32" or not is_marked_install_dir(app_dir):
        return False
    marker = Path(install_marker_path(app_dir))
    if json.loads(marker.read_text(encoding="utf-8")).get("access_layout") == WINDOWS_ACCESS_LAYOUT:
        return False
    if not ctypes.windll.shell32.IsUserAnAdmin():
        return False
    restore_windows_inherited_access_tree(Path(app_dir).resolve())
    prepare_shared_storage(app_dir)
    clear_windows_readonly_tree(Path(app_dir))
    write_install_marker(app_dir)
    return True


def install_runtime_for_app(app_dir: str, progress_callback=None) -> str:
    bundle_runtime_dir = os.path.join(app_dir, "InfernuxHubData", "runtime")
    shared = prepare_shared_storage(app_dir)
    manager = PythonRuntimeManager(
        runtime_dir=os.path.join(shared, "Runtimes"),
        bundle_runtime_dir=bundle_runtime_dir,
        default_version=DEFAULT_PYTHON_RUNTIME,
    )
    return manager.ensure_runtime(
        version=DEFAULT_PYTHON_RUNTIME,
        on_status=progress_callback,
        allow_frozen_repair=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir")
    args = parser.parse_args()

    if not args.app_dir:
        _show_message_box(
            "Infernux Runtime Installer",
            "This program is an internal installer helper for Infernux Hub.\n\n"
            "Please run InfernuxHubInstaller.exe instead of launching this file directly.",
            0x30,
        )
        return 1

    install_runtime_for_app(args.app_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _show_message_box(
            "Infernux Runtime Installer Error",
            str(exc),
            0x10,
        )
        try:
            sys.stderr.write(str(exc) + "\n")
        except Exception as _exc:
            logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
            pass
        raise SystemExit(1)
