"""Editor-machine model authoring tools, never project or Player dependencies."""
from pathlib import Path

from Infernux.engine.preferences_store import PreferencesStore
from Infernux.engine.path_utils import resolved_path


def get_blender_executable() -> str:
    return PreferencesStore().get("blender_executable", "")


def export_script() -> str:
    return str(Path(__file__).with_name("_blender_export.py").resolve())


def configure_database(database) -> None:
    executable = get_blender_executable()
    if executable:
        database.configure_blender_import(executable, export_script())


def set_blender_executable(value: str) -> None:
    from Infernux.core.assets import AssetManager

    executable = resolved_path(value) if value else ""
    if executable and not Path(executable).is_file():
        raise ValueError("Select an existing Blender 5.2 executable")
    database = AssetManager.require_asset_database()
    database.configure_blender_import(executable, export_script() if executable else "")
    PreferencesStore().set("blender_executable", executable)
