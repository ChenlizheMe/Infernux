"""Editor-machine model authoring tools, never project or Player dependencies."""
import os
from pathlib import Path

from infernux.engine.preferences_store import PreferencesStore
from infernux.engine.path_utils import resolved_path
from ._association import find_associated_blender


def get_blender_executable() -> str:
    configured = PreferencesStore().get("blender_executable", "")
    if configured:
        return configured
    return _default_blender_executable()


def _default_blender_executable() -> str:
    managed = os.environ.get("INFERNUX_BLENDER_EXECUTABLE", "").strip()
    if managed and Path(managed).is_file():
        return resolved_path(managed)
    return find_associated_blender()


def export_script() -> str:
    return resolved_path(Path(__file__).with_name("_blender_export.py"))


def configure_database(database) -> None:
    executable = get_blender_executable()
    if executable:
        database.configure_blender_import(executable, export_script())


def set_blender_executable(value: str) -> None:
    from infernux.core.assets import AssetManager

    executable = resolved_path(value) if value else ""
    if executable and not Path(executable).is_file():
        raise ValueError("Select an existing Blender 5.2 executable")
    database = AssetManager.require_asset_database()
    effective = executable or _default_blender_executable()
    database.configure_blender_import(effective, export_script() if effective else "")
    PreferencesStore().set("blender_executable", executable)
