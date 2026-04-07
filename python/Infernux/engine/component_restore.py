"""
Shared Python component restoration logic for Infernux.

All code paths that need to recreate Python component instances from
C++ ``PendingPyComponent`` data should funnel through the single
function :func:`restore_pending_py_components` defined here.

This eliminates the previous duplication across scene_manager,
prefab_manager, and play_mode.
"""

import os
from typing import Optional, Any, List

from Infernux.debug import Debug
from Infernux.engine.project_context import resolve_script_path, resolve_guid_to_path


def resolve_script_from_guid(
    script_guid: str,
    asset_database=None,
) -> Optional[str]:
    """Resolve a script GUID to an absolute filesystem path.

    Handles:
    - Normal editor look-up via AssetDatabase
    - Packaged-build ``.py → .pyc`` fallback
    - Build-time GUID manifest fallback
    """
    script_path = None

    if script_guid and asset_database:
        raw = asset_database.get_path_from_guid(script_guid)
        if raw:
            script_path = resolve_script_path(raw)

    # Packaged-build fallback: use build-time GUID manifest
    if not script_path and script_guid:
        script_path = resolve_guid_to_path(script_guid)

    return script_path


def create_component_instance(
    script_guid: str,
    type_name: str,
    asset_database=None,
):
    """Create a Python component instance (or *None*) from GUID / type name.

    Returns ``(instance, script_path)`` — *instance* may be ``None`` if
    the script cannot be loaded.
    """
    script_path = resolve_script_from_guid(script_guid, asset_database)

    instance = None
    if script_path and os.path.exists(script_path):
        from Infernux.components.script_loader import load_and_create_component
        instance = load_and_create_component(
            script_path, asset_database=asset_database, type_name=type_name,
        )
        if instance is not None and script_guid:
            instance._script_guid = script_guid
    elif not script_path or not os.path.exists(script_path if script_path else ""):
        # No path available — try the global type registry
        from Infernux.components.registry import get_type
        comp_class = get_type(type_name)
        if comp_class:
            instance = comp_class()
            if script_guid:
                instance._script_guid = script_guid

    return instance, script_path


def _make_broken_component(pc, script_path: Optional[str]):
    """Create a :class:`BrokenComponent` placeholder that preserves data."""
    from Infernux.components.component import BrokenComponent

    broken = BrokenComponent()
    broken._broken_type_name = pc.type_name
    broken._script_guid = getattr(pc, "script_guid", "") or ""
    broken._broken_fields_json = pc.fields_json or "{}"

    error_msg = None
    if script_path:
        from Infernux.components.script_loader import get_script_error_by_path
        error_msg = get_script_error_by_path(script_path)
    if not error_msg:
        error_msg = (
            f"Cannot find script for component '{pc.type_name}' "
            f"(guid={getattr(pc, 'script_guid', '')})"
            if not script_path
            else (
                f"Script '{script_path}' contains no InxComponent subclass "
                f"for '{pc.type_name}'"
            )
        )
    broken._broken_error = error_msg
    broken.enabled = pc.enabled
    return broken


def restore_single_component(scene, pc, asset_database=None):
    """Restore one ``PendingPyComponent`` into a live component.

    On success the new :class:`InxComponent` is attached to its
    ``GameObject`` and ``_call_on_after_deserialize`` is invoked.

    On failure a :class:`BrokenComponent` placeholder is attached so the
    serialised data is preserved.

    Returns the attached component (may be ``BrokenComponent``), or
    ``None`` if the target ``GameObject`` no longer exists.
    """
    go = scene.find_by_id(pc.game_object_id)
    if not go:
        Debug.log_warning(
            f"Cannot restore component '{pc.type_name}': "
            f"GameObject {pc.game_object_id} not found"
        )
        return None

    instance, script_path = create_component_instance(
        getattr(pc, "script_guid", "") or "",
        pc.type_name,
        asset_database=asset_database,
    )

    if instance is None:
        broken = _make_broken_component(pc, script_path)
        go.add_py_component(broken)
        Debug.log_warning(
            f"Component '{pc.type_name}' on '{go.name}' loaded as broken "
            f"placeholder — fix the script to restore functionality."
        )
        return broken

    # Apply serialized fields
    if pc.fields_json:
        instance._deserialize_fields(pc.fields_json)

    instance.enabled = pc.enabled
    go.add_py_component(instance)
    instance._call_on_after_deserialize()
    return instance


def restore_pending_py_components(
    scene,
    asset_database=None,
    *,
    clear_registries: bool = False,
    pre_warm_renderstack: bool = False,
    batch_on_after_deserialize: bool = False,
):
    """Restore **all** pending Python components in *scene*.

    Parameters
    ----------
    scene :
        The C++ Scene that may contain pending Python component data.
    asset_database :
        Optional AssetDatabase for GUID → path resolution.
    clear_registries :
        If ``True``, clears :class:`InxComponent` active-instance and
        :class:`BuiltinComponent` caches **before** restoring.
        Set this when the entire scene has been replaced (load / new).
    pre_warm_renderstack :
        If ``True``, pre-warm the RenderStack discovery cache before
        restoration so ``on_after_deserialize`` can call
        ``discover_passes()`` without a full directory walk.
    batch_on_after_deserialize :
        If ``True``, suppresses individual ``_call_on_after_deserialize``
        during restore and performs a single batch call at the end.
        Used by play-mode which attaches all components first.
    """
    if clear_registries:
        from Infernux.components.component import InxComponent
        InxComponent._clear_all_instances()
        from Infernux.components.builtin_component import BuiltinComponent
        BuiltinComponent._clear_cache()
        from Infernux.gizmos.collector import notify_scene_changed
        notify_scene_changed()

    if not scene.has_pending_py_components():
        return

    pending = scene.take_pending_py_components()
    if not pending:
        return

    if pre_warm_renderstack:
        try:
            from Infernux.renderstack.discovery import discover_passes, discover_pipelines
            discover_passes()
            discover_pipelines()
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

    restored = 0
    deferred_callbacks: List[Any] = []

    for pc in pending:
        try:
            if batch_on_after_deserialize:
                # Create instance but defer on_after_deserialize
                go = scene.find_by_id(pc.game_object_id)
                if not go:
                    Debug.log_warning(
                        f"Cannot restore component: "
                        f"GameObject {pc.game_object_id} not found"
                    )
                    continue

                instance, script_path = create_component_instance(
                    getattr(pc, "script_guid", "") or "",
                    pc.type_name,
                    asset_database=asset_database,
                )

                if instance is None:
                    broken = _make_broken_component(pc, script_path)
                    go.add_py_component(broken)
                    Debug.log_warning(
                        f"Component '{pc.type_name}' on '{go.name}' loaded "
                        f"as broken placeholder."
                    )
                    continue

                if pc.fields_json:
                    instance._deserialize_fields(
                        pc.fields_json,
                        _skip_on_after_deserialize=True,
                    )
                instance.enabled = pc.enabled
                go.add_py_component(instance)
                deferred_callbacks.append(instance)
                restored += 1
            else:
                comp = restore_single_component(
                    scene, pc, asset_database=asset_database,
                )
                if comp is not None:
                    from Infernux.components.component import BrokenComponent
                    if not isinstance(comp, BrokenComponent):
                        restored += 1
        except Exception as exc:
            Debug.log_error(
                f"Failed to restore component '{pc.type_name}' on "
                f"GameObject {pc.game_object_id}: {exc}"
            )
            # Last-resort BrokenComponent so data is not lost
            try:
                go = scene.find_by_id(pc.game_object_id)
                if go:
                    broken = _make_broken_component(pc, None)
                    broken._broken_error = f"Restore failed: {exc}"
                    go.add_py_component(broken)
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass

    # Batch on_after_deserialize
    for comp in deferred_callbacks:
        try:
            comp._call_on_after_deserialize()
        except Exception as exc:
            Debug.log_warning(
                f"on_after_deserialize failed for "
                f"{type(comp).__name__}: {exc}"
            )

    Debug.log_internal(
        f"Restored {restored}/{len(pending)} Python component(s)"
    )
