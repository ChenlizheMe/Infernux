"""Current script loading, component discovery and source diagnostics.

Editor source changes are published exclusively by ReloadCoordinator.
Player/source attachment uses the same component identities without live-body patching.
"""
from __future__ import annotations
import importlib
import importlib.util
import inspect
import os
import sys
import types
from typing import Type, List, Optional
from infernux.engine.path_utils import path_key, resolved_path
from infernux.engine.project_context import get_script_module_name, resolve_script_path, temporary_script_import_paths
from .component import InxComponent

_script_errors: dict[str, str] = {}
_script_error_revision = 0



class ScriptLoadError(Exception):
    """Raised when a script cannot be loaded or doesn't contain valid components."""

    def __init__(self, message: str, *, source_file: str = "", source_line: int = 0):
        super().__init__(message)
        self.source_file = source_file
        self.source_line = source_line


def _component_classes_from_module(
    module: types.ModuleType,
    module_name: str,
) -> list[Type[InxComponent]]:
    """Extract component declarations in source execution order.

    ``inspect.getmembers`` sorts names alphabetically.  That is the wrong
    ordering for a script asset: the first component declaration is the
    deterministic attachment target when the authored class name changes.
    Module dictionaries retain execution order, so use that order directly.
    """
    components = []
    for _name, obj in vars(module).items():
        if (
            inspect.isclass(obj)
            and obj is not InxComponent
            and issubclass(obj, InxComponent)
            and obj.__module__ == module_name
        ):
            components.append(obj)
    return components


def retire_script_module(file_path: str) -> object | None:
    """Remove the source's canonical module after a delete or move."""
    module_name = get_script_module_name(file_path)
    if not module_name:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        return None
    module_file = resolved_path(getattr(module, "__file__", "") or "")
    if module_file and path_key(module_file) != path_key(file_path):
        # A move can keep the same canonical module name. The candidate at
        # the destination must survive; only the old module may be retired.
        return None
    return sys.modules.pop(module_name, None)


def _set_script_error_key(key: str, message: str | None) -> None:
    """Update one normalized error entry and advance the change revision."""
    global _script_error_revision
    previous = _script_errors.get(key)
    if message is None:
        if key not in _script_errors:
            return
        _script_errors.pop(key, None)
    else:
        if previous == message:
            return
        _script_errors[key] = message
    _script_error_revision += 1


def _normalize_script_path(file_path: str) -> str:
    """Return a stable absolute key for script-error bookkeeping."""
    return path_key(file_path)


def _unique_module_name_for_path(file_path: str) -> str:
    """Build a fallback module name for scripts without a valid import path."""
    import hashlib

    normalized_path = path_key(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    path_hash = hashlib.md5(normalized_path.encode()).hexdigest()[:8]
    return f"infernux_script_{module_name}_{path_hash}"


def _record_script_error(file_path: str, exc: Exception) -> None:
    """Record that *file_path* failed to load with *exc*."""
    import traceback
    norm = _normalize_script_path(file_path)
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    _set_script_error_key(norm, tb_str)
    # Also log to Console so the user sees it
    try:
        from infernux.debug import Debug
        Debug.log_error(tb_str, source_file=file_path, source_line=0)
    except ImportError:
        import sys
        print(tb_str, file=sys.stderr)


def set_script_error(file_path: str, message: str) -> None:
    """Record an error message for a script (no exception object needed)."""
    _set_script_error_key(_normalize_script_path(file_path), message)


def _clear_script_error(file_path: str) -> None:
    """Clear any previously recorded error for *file_path*."""
    _set_script_error_key(_normalize_script_path(file_path), None)


def clear_deleted_script_errors(path: str) -> list[str]:
    """Forget tracked script errors for a deleted script path.

    Accepts either a single script file or a directory path. Directory cleanup is
    useful for editor-side recursive deletes where every nested broken script
    should stop blocking Play Mode immediately instead of waiting for a restart.
    """
    if not path:
        return []

    normalized = _normalize_script_path(path)
    removed: list[str] = []

    if os.path.isdir(path):
        prefix = normalized.rstrip("\\/") + os.sep
        for key in list(_script_errors.keys()):
            if key == normalized or key.startswith(prefix):
                _set_script_error_key(key, None)
                removed.append(key)
        return removed

    if normalized in _script_errors:
        _set_script_error_key(normalized, None)
        removed.append(normalized)
    return removed


def get_script_errors() -> dict[str, str]:
    """Return a snapshot of all currently broken scripts {path: traceback}."""
    return dict(_script_errors)


def has_script_errors() -> bool:
    """Return True if any loaded script has unresolved errors."""
    return bool(_script_errors)


def get_script_error_revision() -> int:
    """Return a monotonic revision changed only when diagnostics change."""
    return _script_error_revision


def get_script_error_by_path(file_path: str) -> Optional[str]:
    """Return the error string for *file_path*, or ``None`` if it loaded OK."""
    return _script_errors.get(_normalize_script_path(file_path))


def load_component_from_file(file_path: str) -> Type[InxComponent]:
    """
    Load the first InxComponent subclass from a Python file.
    
    Args:
        file_path: Absolute path to the .py file
        
    Returns:
        The first InxComponent subclass found in the file
        
    Raises:
        ScriptLoadError: If file doesn't exist, can't be imported, or contains no components
    """
    components = load_all_components_from_file(file_path)
    if not components:
        raise ScriptLoadError(f"No InxComponent subclasses found in {file_path}")
    if len(components) > 1:
        names = ", ".join(cls.__name__ for cls in components)
        raise ScriptLoadError(
            f"Script '{file_path}' defines multiple InxComponent classes ({names}). "
            "Dragging or attaching by script file requires exactly one component class."
        )
    return components[0]


def load_component_class_from_file(file_path: str, type_name: str = "") -> Optional[Type[InxComponent]]:
    """Load a specific component class from a Python file.

    When ``type_name`` is provided, prefer an exact class-name match. If the
    authored name is missing but the file still defines exactly one
    ``InxComponent`` subclass, return that class so a pure class rename
    (same script GUID / one-component-per-file) keeps scene references alive.
    """
    components = load_all_components_from_file(file_path)
    if not components:
        return None

    if type_name:
        for component_class in components:
            if component_class.__name__ == type_name:
                return component_class

    # A script asset is attached by file/GUID, not by a fragile class-name
    # convention.  When the authored name differs (or the file declares more
    # than one component), the first declaration is the stable editor target.
    return components[0]


def create_component_instance(component_class: Type[InxComponent]) -> InxComponent:
    """
    Create an instance of a component class.
    
    Args:
        component_class: The InxComponent subclass to instantiate
        
    Returns:
        New instance of the component
        
    """
    return component_class()


def load_and_create_component(
    file_path: str,
    asset_database=None,
    type_name: str = "",
    *,
    script_guid: str = "",
) -> Optional[InxComponent]:
    """
    Convenience function: Load first component from file and create instance.
    
    Args:
        file_path: Absolute path to the .py file
        
    Returns:
        New instance of the first component found, or None if the script
        has errors (the error is already logged to Console).
        
    Note:
        ``script_guid`` should be supplied when the caller already resolved
        ``file_path`` from a stable component identity. Packaged ``.pyc``
        artifacts do not necessarily support a reverse path-to-GUID lookup.

    Raises:
        ScriptLoadError: If AssetDatabase is missing or GUID cannot be resolved.
    """
    if asset_database is None and not script_guid:
        raise ScriptLoadError("AssetDatabase is required for script components (GUID-only mode)")

    # Resolve and store script GUID
    guid = script_guid or asset_database.get_guid_from_path(file_path)
    if not guid and asset_database is not None:
        from infernux.core.assets import AssetManager
        mutation = AssetManager.import_asset(
            file_path,
            database=asset_database,
            suppress_watcher_echo=False,
        )
        guid = mutation.guid
    if not guid:
        raise ScriptLoadError(f"Failed to resolve GUID for script: {file_path}")

    # A script file is one identity/publication unit even when it declares
    # several components. Methods on one component may refer to sibling types
    # through their module globals, so binding only the requested class leaves
    # those references on provisional module GUIDs and breaks component lookup.
    from infernux.components.registry import (
        component_types_for_script_path,
        publish_component_script_types,
    )
    # Instantiation consumes the published revision; only the script-refresh
    # transaction replaces it. Re-executing here strands earlier instances.
    component_types = component_types_for_script_path(file_path)
    already_published = bool(component_types) and all(
        candidate._asset_script_guid_ == guid for candidate in component_types
    )
    if not already_published:
        component_types = tuple(load_all_components_from_file(file_path, register=False))
    if not component_types:
        return None
    component_class = next(
        (candidate for candidate in component_types if type_name and candidate.__name__ == type_name),
        component_types[0],
    )

    if not already_published:
        from infernux.components.component_identity import bind_asset_script_guid
        for candidate in component_types:
            bind_asset_script_guid(candidate, guid, register=False)
        publish_component_script_types(file_path, component_types)

    instance = create_component_instance(component_class)
    instance._script_guid = guid
    instance._script_path = resolved_path(file_path)
    return instance


def get_component_info(component_class: Type[InxComponent]) -> dict:
    """
    Extract metadata from a component class.
    
    Args:
        component_class: The InxComponent subclass
        
    Returns:
        Dictionary with component metadata (name, docstring, fields)
    """
    from .fields import get_serialized_fields
    
    return {
        'name': component_class.__name__,
        'module': component_class.__module__,
        'docstring': inspect.getdoc(component_class) or "",
        'fields': list(get_serialized_fields(component_class).keys()),
    }


def _load_script_module(file_path, module_name, *, source_only=False, source=None, code=None):
    """Execute current bytes; a failed module never restores old definitions."""
    from .registry import unregister_component_script
    from .serializable_object import _candidate_serializable_scope, _publish_serializable_types
    from infernux.renderstack.render_effect_compiler import _stage_source_effect_features, _retire_source_effect_features
    from infernux.engine.project_context import package_script_reference

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ScriptLoadError(f"Cannot load script: {file_path}")
    reference = package_script_reference(file_path)
    if reference:
        from infernux.plugins.preload import _ensure_module_parents
        _ensure_module_parents(module_name, file_path, reference)
    module = importlib.util.module_from_spec(spec)
    unregister_component_script(file_path)
    _publish_serializable_types({}, {module_name})
    _retire_source_effect_features(file_path)
    sys.modules[module_name] = module
    data_types = {}
    try:
        with _stage_source_effect_features(file_path), _candidate_serializable_scope(data_types, module_name):
            if code is None:
                if source is None and file_path.endswith('.py'):
                    with open(file_path, 'rb') as stream:
                        source = stream.read()
                code = compile(source, file_path, 'exec', dont_inherit=True) if source is not None else spec.loader.get_code(module_name)
            exec(code, vars(module))
            _publish_serializable_types(data_types, {module_name})
    except Exception:
        sys.modules.pop(module_name, None)
        unregister_component_script(file_path)
        _publish_serializable_types({}, {module_name})
        _retire_source_effect_features(file_path)
        raise
    return module


def load_all_components_from_file(file_path, *, register=True, source_only=False, source=None, code=None):
    """Load current declarations; editor changes always use the coordinator."""
    file_path = resolve_script_path(file_path)
    module_name = get_script_module_name(file_path) or _unique_module_name_for_path(file_path)
    try:
        from infernux.engine.resources_manager import ResourcesManager
        manager = ResourcesManager.instance()
        from infernux.engine.reload.modules import coordinator_for_path
        coordinator = (
            manager._ensure_event_handler().reload if manager is not None
            else coordinator_for_path(file_path)
        )
        if coordinator is not None and file_path.endswith('.py'):
            snapshots = {file_path: source} if source is not None else None
            coordinator.ensure_ready((file_path,), snapshots=snapshots)
            module = sys.modules[module_name]
        else:
            with temporary_script_import_paths(file_path):
                module = _load_script_module(file_path, module_name, source_only=source_only, source=source, code=code)
    except Exception as exc:
        _record_script_error(file_path, exc)
        return []
    components = _component_classes_from_module(module, module_name)
    if register:
        from .registry import publish_component_script_types
        publish_component_script_types(file_path, components)
    _clear_script_error(file_path)
    return components
