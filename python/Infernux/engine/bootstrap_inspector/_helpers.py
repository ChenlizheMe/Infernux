"""Standalone utility functions for inspector wiring.

These helpers do not depend on any wiring-time closure state.
"""
from __future__ import annotations

from Infernux.debug import Debug


def _safe_sequence(values):
    if values is None:
        return []
    if isinstance(values, list):
        return values
    try:
        return list(values)
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return []


def _get_components_safe(obj):
    if obj is None:
        return []
    try:
        return _safe_sequence(obj.get_components())
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return []


def _get_py_components_safe(obj):
    if obj is None:
        return []
    try:
        return _safe_sequence(obj.get_py_components())
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return []


def _can_remove_component(obj, comp, type_name, is_native):
    """Check whether *comp* may be removed from *obj*."""
    if is_native:
        blockers = []
        if hasattr(obj, 'get_remove_component_blockers'):
            try:
                blockers = list(obj.get_remove_component_blockers(comp) or [])
            except RuntimeError:
                blockers = []
        can_remove = not blockers
        if can_remove and hasattr(obj, 'can_remove_component'):
            can_remove = bool(obj.can_remove_component(comp))
        if not can_remove:
            suffix = (
                f" required by: {', '.join(blockers)}"
                if blockers else
                "another component depends on it"
            )
            Debug.log_warning(f"Cannot remove '{type_name}' — {suffix}")
            return False
    return True


def _get_add_component_entries():
    """Build the list of addable component entries (native + engine + scripts)."""
    from Infernux.lib import InspectorAddComponentEntry, get_registered_component_types

    entries = []
    from Infernux.components.builtin_component import BuiltinComponent
    for type_name in sorted(get_registered_component_types()):
        if type_name == "Transform":
            continue
        e = InspectorAddComponentEntry()
        e.display_name = type_name
        # Use BuiltinComponent wrapper category if available
        wrapper_cls = BuiltinComponent._builtin_registry.get(type_name)
        e.category = getattr(wrapper_cls, '_component_category_', "Built-in") if wrapper_cls else "Built-in"
        e.is_native = True
        entries.append(e)

    from Infernux.renderstack.render_stack import RenderStack
    for display_name, _comp_cls in [("RenderStack", RenderStack)]:
        e = InspectorAddComponentEntry()
        e.display_name = display_name
        e.category = "Engine"
        e.is_native = False
        e.script_path = ""
        entries.append(e)

    # Engine-side Python-only components (e.g. SpriteRenderer)
    from Infernux.components.registry import get_all_types
    seen = {e.display_name for e in entries}
    for name, cls in get_all_types().items():
        if name in seen:
            continue
        menu_path = getattr(cls, '_component_menu_path_', None)
        if menu_path:
            e = InspectorAddComponentEntry()
            e.display_name = name
            e.category = getattr(cls, '_component_category_', 'Scripts')
            e.is_native = False
            e.script_path = ""
            entries.append(e)

    import os
    from Infernux.engine.project_context import get_project_root
    from Infernux.components.script_loader import load_component_from_file, ScriptLoadError

    project_root = get_project_root()
    if project_root and os.path.isdir(project_root):
        for dirpath, _dirnames, filenames in os.walk(project_root):
            rel = os.path.relpath(dirpath, project_root)
            if any(part.startswith('.') or part in (
                    '__pycache__', 'build', 'Library',
                    'ProjectSettings', 'Logs', 'Temp')
                   for part in rel.split(os.sep)):
                continue
            for fn in filenames:
                if not fn.endswith('.py') or fn.startswith('_'):
                    continue
                full = os.path.join(dirpath, fn)
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(4096)
                if 'InxComponent' not in content:
                    continue
                try:
                    comp_class = load_component_from_file(full)
                except (ScriptLoadError, Exception) as _exc:
                    Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                    continue
                e = InspectorAddComponentEntry()
                e.display_name = comp_class.__name__
                e.category = "Scripts"
                e.is_native = False
                e.script_path = full
                entries.append(e)
    return entries


def _load_script_component(script_path, asset_database):
    """Load a script component, returning the instance or ``None``."""
    from Infernux.components import load_and_create_component
    try:
        instance = load_and_create_component(
            script_path, asset_database=asset_database)
    except Exception as exc:
        Debug.log_error(f"Failed to load script '{script_path}': {exc}")
        return None
    if instance is None:
        Debug.log_error(f"No InxComponent found in '{script_path}'")
    return instance


def _remove_component_impl(obj_id, type_name, comp_id, is_native,
                           resolve_component, can_remove_component,
                           invalidate_cache, bump_values):
    """Remove a component from an object."""
    from Infernux.lib import SceneManager
    scene = SceneManager.instance().get_active_scene()
    obj = scene.find_by_id(obj_id) if scene else None
    if obj is None:
        return False
    comp = resolve_component(obj_id, comp_id, is_native)
    if comp is None:
        return False
    if not can_remove_component(obj, comp, type_name, is_native):
        return False
    from Infernux.engine.undo import UndoManager
    mgr = UndoManager.instance()
    if is_native:
        from Infernux.engine.undo import RemoveNativeComponentCommand
        if mgr:
            mgr.execute(RemoveNativeComponentCommand(obj.id, type_name, comp))
            invalidate_cache()
            bump_values()
            return True
        ok = obj.remove_component(comp) is not False
    else:
        from Infernux.engine.undo import RemovePyComponentCommand
        if mgr:
            mgr.execute(RemovePyComponentCommand(obj.id, comp))
            invalidate_cache()
            bump_values()
            return True
        ok = obj.remove_py_component(comp) is not False
    if ok:
        invalidate_cache()
        bump_values()
    return ok
