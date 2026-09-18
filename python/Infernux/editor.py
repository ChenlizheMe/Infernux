"""Editor-only authoring API, available as ``inx.editor``.

Tools use the same command and shortcut services as the built-in Editor.
Register contributions from an editor preload, not from gameplay callbacks.
"""

from .engine.interaction.documents import DocumentActionResult, DocumentActionStatus
from .engine.prefab_overrides import PropertyModification

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
    "load_prefab_contents", "save_as_prefab_asset", "unload_prefab_contents",
    "revert_prefab", "save_scene", "open_scene", "new_scene", "undo", "redo",
    "create_data_asset", "create_folder", "get_build_scenes", "set_build_scenes",
    "save_project_settings",
    "revert_property_override",
    "load_data_asset", "set_data_asset_fields", "save_data_asset",
    "add_component",
    "PropertyModification", "get_property_modifications", "is_property_override",
    "defer",
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


def defer(callback, *, description: str = "Editor task") -> bool:
    """Run a batch at the next editor owner safe point, outside GUI drawing.

    Returns False while another editor task is active. The callback is not
    executed inline or retried, and owns its individual authoring/Undo calls.
    """
    _authoring_core()
    if not callable(callback):
        raise TypeError("Editor deferred callback must be callable")
    from .engine.deferred_task import DeferredTaskRunner

    return DeferredTaskRunner.instance().submit(description, [(description, 0.5, callback)])


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


def load_prefab_contents(path):
    """Load into an isolated Scene; release the returned root in ``finally``.

    Like scene loading, run batch authoring at an owner safe point (the
    public editor.defer entry), not inside a GUI draw callback. Resolving an
    unloaded/missing script may need to publish a runtime type descriptor.
    Loaded contents do not run Awake/Start/Update or join rendering/physics.
    Serialization callbacks still run. Mutations of this temporary tree are
    not individual Undo commands; saving the asset is a Project Undo entry.
    """
    return _authoring_core().prefabs.load_contents(_asset_path(path))


def save_as_prefab_asset(game_object, path) -> str:
    """Save contents or a new hierarchy through Project history; return the path.

    To overwrite an existing asset, edit its load_prefab_contents root. Scene
    instances are not linked by this operation; use create_prefab for that.
    """
    return _authoring_core().prefabs.save_contents(game_object, _asset_path(path))


def unload_prefab_contents(root) -> None:
    """Discard the isolated Scene. Unsaved edits are deliberately not written."""
    _authoring_core().prefabs.unload_contents(root)


def add_component(game_object, type_name: str, *, configure=None):
    """Attach a registered component through Inspector's undoable command."""
    _authoring_core()
    from .engine.interaction.components import ComponentCommandService

    return ComponentCommandService.require().add(
        game_object, type_name, initializer=configure,
    )


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


def load_data_asset(path):
    """Read the live Editor document, including edits not yet flushed to disk."""
    from .core.data_asset import DataAsset
    from .engine.interaction import DocumentKey, DocumentKind, DocumentRegistry

    core = _authoring_core()
    target = core.project_assets._project_path(_asset_path(path))
    guid = str(core.project_assets.asset_database.get_guid_from_path(target) or "")
    if not guid:
        raise ValueError(f"DataAsset is not registered: {target}")
    document = DocumentRegistry.instance().get_by_key(DocumentKey.asset(DocumentKind.DATA_ASSET, guid))
    if document is not None:
        return document.controller.resource
    return _data_asset_document(DataAsset.load(target)).resource


def _data_asset_document(asset):
    from .core.data_asset import DataAsset
    from .engine.interaction import (
        DocumentKey, DocumentKind, DocumentRegistry, ensure_editable_resource_document,
    )

    core = _authoring_core()
    if not isinstance(asset, DataAsset) or not asset.is_persistent:
        raise ValueError("DataAsset editing requires a persistent DataAsset")
    core.project_assets._project_path(asset.file_path)
    document = DocumentRegistry.instance().get_by_key(
        DocumentKey.asset(DocumentKind.DATA_ASSET, asset.guid),
    )
    # A loaded handle may precede an Inspector edit or script refresh. The
    # shared document owns the current value, not that caller's older handle.
    resource = document.controller.resource if document is not None else asset
    return ensure_editable_resource_document(
        category="data_asset", document_kind=DocumentKind.DATA_ASSET,
        file_path=asset.file_path, resource=resource, guid=asset.guid,
        view_id="editor.authoring",
    )


def set_data_asset_fields(asset, **values) -> bool:
    """Edit declared fields together through the Inspector's document and Undo."""
    from .components.fields import get_field_schema

    controller = _data_asset_document(asset)
    draft = controller.resource.instantiate()
    for name, value in values.items():
        schema = get_field_schema(type(draft), name)
        if schema.read_only:
            raise ValueError(f"Property is read-only: {schema.property_path}")
        setattr(draft, name, value)
    after = draft.serialize_document()
    if after == controller.capture_document():
        return False
    if not controller.apply_document(
        after, view_id="editor.authoring",
        edit_key="data_asset:" + ",".join(sorted(values)),
        description="Set Data Asset Fields",
    ):
        raise RuntimeError("DataAsset field edit was rejected")
    return True


def save_data_asset(asset) -> DocumentActionResult:
    """Flush a DataAsset through its SaveTicket; PENDING is not completion."""
    controller = _data_asset_document(asset)
    return _authoring_core().documents.request_save(controller.document_id)


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


def get_property_modifications(game_object) -> tuple[PropertyModification, ...]:
    """Read detached field differences for its nearest Prefab instance subtree.

    Includes default root placement differences; structural additions/removals
    are not property modifications. An ordinary scene object returns ().
    """
    return _authoring_core().prefabs.property_modifications(game_object.id)


def is_property_override(component, field_name: str) -> bool:
    """Query a declared field by its Python name without modifying history."""
    return _authoring_core().prefabs.is_property_override(component, field_name)


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
