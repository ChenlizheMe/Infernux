"""Editor-only authoring API, available as ``inx.editor``.

Tools use the same command and shortcut services as the built-in Editor.
Register contributions from an editor preload, not from gameplay callbacks.
"""

from .engine.interaction.documents import DocumentActionResult, DocumentActionStatus

from .engine.interaction.commands import (
    CommandContext, CommandResult, CommandSource, CommandStatus,
    EditorCommand, EditorCommandRegistry,
)
from .engine.interaction.shortcuts import (
    KeyChord, ShortcutBinding, ShortcutModifier, ShortcutPhase,
    ShortcutRouter, ShortcutScope,
)

__all__ = (
    "CommandContext", "CommandResult", "CommandSource", "CommandStatus",
    "EditorCommand", "EditorCommandRegistry", "KeyChord", "ShortcutBinding",
    "ShortcutModifier", "ShortcutPhase", "ShortcutRouter", "ShortcutScope",
    "DocumentActionResult", "DocumentActionStatus", "edit_scene",
    "create_game_object", "create_prefab", "instantiate_prefab", "apply_prefab",
    "revert_prefab", "save_scene", "open_scene", "new_scene", "undo", "redo",
    "create_data_asset", "create_folder", "get_build_scenes", "set_build_scenes",
    "save_project_settings",
    "revert_property_override",
)


def _authoring_core():
    from .engine.interaction.session import EditorInteractionCore
    from .engine.play_mode import PlayModeManager

    core = EditorInteractionCore.instance()
    if core is None:
        raise RuntimeError("Editor authoring requires an active Editor session")
    play = PlayModeManager.instance()
    if play is not None and play.is_playing:
        raise RuntimeError("Editor authoring is unavailable in Play Mode")
    return core


def _asset_path(path):
    import os
    from .engine.project_context import get_project_root
    from .engine.path_utils import resolved_path

    return resolved_path(os.path.join(get_project_root(), os.fspath(path)))


def edit_scene(description: str):
    """Group journal-aware operations into one Undo entry; not arbitrary writes."""
    return _authoring_core().scene_objects.user_action(description)


def create_game_object(name: str = "GameObject", *, kind: str = "empty", parent=None, configure=None):
    """Create through Hierarchy; initialize inside configure for one Undo record."""
    _authoring_core()
    from .engine.hierarchy_creation_service import HierarchyCreationService
    from .lib import SceneManager

    result = HierarchyCreationService.instance().create(
        kind, name=name, parent_id=parent.id if parent is not None else 0,
        configure_created=configure,
    )
    return SceneManager.instance().find_runtime_object_by_id(result["id"])


def create_prefab(game_object, directory="Assets") -> str:
    """Create a uniquely named asset in directory and link its source hierarchy."""
    return _authoring_core().prefabs.create_from_object(game_object.id, _asset_path(directory))


def create_data_asset(value, path) -> str:
    """Create a typed asset through Project Undo; leave the input value unchanged."""
    import os
    from .core.data_asset import DataAsset, DATA_ASSET_EXTENSION
    from .engine.ui import project_file_ops

    core = _authoring_core()
    if not isinstance(value, DataAsset):
        raise TypeError("create_data_asset requires a DataAsset value")
    target = _asset_path(path)
    if os.path.splitext(target)[1].casefold() != DATA_ASSET_EXTENSION:
        raise ValueError("DataAsset paths require the .inxdata extension")
    parent, name = os.path.split(target)
    return core.project_assets.create_with_path(
        parent,
        lambda: project_file_ops.create_data_asset(
            parent, name, type(value).__serialized_type_id__,
            core.project_assets.asset_database, value=value,
        ),
        description="Create Data Asset",
    )


def create_folder(path) -> str:
    """Create one project folder through Project Undo; its parent must exist."""
    import os
    from .engine.ui import project_file_ops

    core = _authoring_core()
    parent, name = os.path.split(_asset_path(path))
    if not os.path.isdir(parent):
        raise FileNotFoundError(parent)
    return core.project_assets.create_with_path(
        parent, lambda: project_file_ops.create_folder(parent, name),
        description="Create Folder",
    )


def _project_settings():
    from .engine.interaction.project_settings import ensure_project_settings_document
    core = _authoring_core()
    return ensure_project_settings_document(core.project_assets.project_root)


def get_build_scenes() -> list[str]:
    """Return a copy of the ordered project-relative build scene list."""
    return _project_settings().section("build")["scenes"]


def set_build_scenes(paths) -> bool:
    """Edit the shared Build Settings document as one undoable operation."""
    import os
    from .engine.path_utils import is_path_within, relative_path

    core = _authoring_core()
    if not isinstance(paths, (list, tuple)):
        raise TypeError("Build scenes must be a list or tuple of paths")
    root = core.project_assets.project_root
    scenes = []
    for path in paths:
        target = _asset_path(path)
        if not is_path_within(target, os.path.join(root, "Assets"), allow_root=False) or not target.lower().endswith(".scene"):
            raise ValueError("Build scenes must be .scene assets beneath Assets")
        if not os.path.isfile(target):
            raise FileNotFoundError(target)
        scenes.append(relative_path(target, root).replace("\\", "/"))
    controller = _project_settings()
    settings = controller.section("build")
    settings["scenes"] = scenes
    return controller.apply_section("build", settings, edit_key="build.scenes", description="Set Build Scenes")


def save_project_settings() -> DocumentActionResult:
    """Save through the normal document ticket; PENDING is not completion."""
    core = _authoring_core()
    return core.documents.request_save(_project_settings().document_id)


def instantiate_prefab(path, *, parent=None):
    """Return the newly placed GameObject, using the same Undo path as drag/drop."""
    result = _authoring_core().scene_objects.instantiate_prefab_object(
        _asset_path(path), parent_id=parent.id if parent is not None else 0,
    )
    if result is None:
        raise RuntimeError(f"Prefab instantiation failed: {path}")
    return result


def apply_prefab(game_object) -> bool:
    return _authoring_core().prefabs.apply(game_object.id)


def revert_prefab(game_object) -> bool:
    return _authoring_core().prefabs.revert(game_object.id)


def revert_property_override(component, field_name: str) -> bool:
    """Revert one declared component field, retaining unrelated instance edits."""
    return _authoring_core().prefabs.revert_property(component, field_name)


def save_scene(path=None) -> DocumentActionResult:
    """Save the active Scene/Prefab through SaveTicket; an explicit path is Save As.

    Inspect the returned status: PENDING means a dialog or async save is still
    outstanding, not a completed disk write. Paths are project-relative.
    """
    core = _authoring_core()
    from .engine.scene_manager import SceneFileManager

    document_id = SceneFileManager.instance().document_id
    if path is not None:
        return core.documents.request_save_to_resource(document_id, _asset_path(path))
    return core.documents.request_save(document_id)


def open_scene(path) -> bool:
    """Request a deferred scene change using the normal unsaved-changes dialog."""
    _authoring_core()
    from .engine.scene_manager import SceneFileManager
    return SceneFileManager.instance().open_scene(_asset_path(path))


def new_scene():
    """Request a new scene; unsaved-change confirmation and frame deferral apply."""
    _authoring_core()
    from .engine.scene_manager import SceneFileManager
    return SceneFileManager.instance().new_scene()


def undo(*, defer: bool = True):
    _authoring_core()
    from .engine.undo import UndoManager
    return UndoManager.instance().undo(defer=defer)


def redo(*, defer: bool = True):
    _authoring_core()
    from .engine.undo import UndoManager
    return UndoManager.instance().redo(defer=defer)
