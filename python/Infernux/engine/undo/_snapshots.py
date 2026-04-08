"""Snapshot/restore helpers and live-object resolvers for Inspector tracker."""

from __future__ import annotations

import json as _json
from typing import Any

from Infernux.debug import Debug
from Infernux.engine.undo._helpers import _get_active_scene


# -- Live-object resolvers --

def _get_live_game_object(game_object_id: int):
    if not game_object_id:
        return None
    scene = _get_active_scene()
    if not scene:
        return None
    try:
        return scene.find_by_id(game_object_id)
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return None


def _get_live_transform(game_object_id: int):
    obj = _get_live_game_object(game_object_id)
    if obj is None:
        return None
    try:
        t = obj.get_transform()
        if t is not None:
            return t
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    return getattr(obj, 'transform', None)


def _get_nth_live_native_component(game_object_id: int, type_name: str,
                                   ordinal: int = 0):
    if type_name == "Transform":
        return _get_live_transform(game_object_id)
    obj = _get_live_game_object(game_object_id)
    if obj is None or not hasattr(obj, 'get_components'):
        return None
    match_index = 0
    try:
        try:
            from Infernux.components.component import InxComponent
        except Exception:
            InxComponent = ()
        for comp in obj.get_components():
            try:
                if getattr(comp, 'type_name', None) != type_name:
                    continue
                if isinstance(comp, InxComponent) or hasattr(comp, 'get_py_component'):
                    continue
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                continue
            if match_index == ordinal:
                return comp
            match_index += 1
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    return None


def _get_nth_live_py_component(game_object_id: int, type_name: str,
                               ordinal: int = 0, script_guid: str = ""):
    obj = _get_live_game_object(game_object_id)
    if obj is None or not hasattr(obj, 'get_py_components'):
        return None
    match_index = 0
    try:
        for comp in obj.get_py_components():
            try:
                ct = getattr(comp, 'type_name', type(comp).__name__)
                if ct != type_name and type(comp).__name__ != type_name:
                    continue
                if script_guid and getattr(comp, '_script_guid', '') != script_guid:
                    continue
                if getattr(comp, '_is_destroyed', False):
                    continue
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                continue
            if match_index == ordinal:
                return comp
            match_index += 1
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    return None


# -- Snapshot/restore for game objects --

def _snap_game_object(obj) -> str:
    return _json.dumps({"name": obj.name, "active": obj.active})


def _restore_game_object(obj, snapshot: str) -> None:
    data = _json.loads(snapshot)
    obj.name = data["name"]
    obj.active = data["active"]


# -- Table-driven registry --

_SNAPSHOT_REGISTRY: dict[str, tuple] = {
    "game_object": (_get_live_game_object, _snap_game_object, _restore_game_object),
    "transform":   (_get_live_transform, lambda t: t.serialize(), lambda t, s: t.deserialize(s)),
    "native":      (_get_nth_live_native_component, lambda c: c.serialize(), lambda c, s: c.deserialize(s)),
    "py":          (_get_nth_live_py_component, lambda c: c._serialize_fields(), lambda c, s: c._deserialize_fields(s)),
    "renderstack": (_get_nth_live_py_component,
                    lambda c: _lazy_snapshot_renderstack(c),
                    lambda c, s: _lazy_restore_renderstack(c, s)),
}


def _lazy_snapshot_renderstack(stack):
    from Infernux.engine.undo._renderstack import snapshot_renderstack
    return snapshot_renderstack(stack)


def _lazy_restore_renderstack(stack, json_str):
    from Infernux.engine.undo._renderstack import restore_renderstack
    restore_renderstack(stack, json_str)


def _resolve_and_snap(kind: str, game_object_id: int, **kwargs) -> str:
    resolver, snap_fn, _ = _SNAPSHOT_REGISTRY[kind]
    target = resolver(game_object_id, **kwargs) if kwargs else resolver(game_object_id)
    if target is None:
        return ""
    return snap_fn(target)


def _resolve_and_restore(kind: str, game_object_id: int, snapshot: str, **kwargs) -> None:
    _, _, restore_fn = _SNAPSHOT_REGISTRY[kind]
    resolver = _SNAPSHOT_REGISTRY[kind][0]
    target = resolver(game_object_id, **kwargs) if kwargs else resolver(game_object_id)
    if target is None:
        return
    restore_fn(target, snapshot)


# -- Public thin wrappers (preserves call signatures) --

def snapshot_live_game_object(game_object_id: int) -> str:
    return _resolve_and_snap("game_object", game_object_id)

def restore_live_game_object(game_object_id: int, snapshot: str) -> None:
    _resolve_and_restore("game_object", game_object_id, snapshot)

def snapshot_live_transform(game_object_id: int) -> str:
    return _resolve_and_snap("transform", game_object_id)

def restore_live_transform(game_object_id: int, snapshot: str) -> None:
    _resolve_and_restore("transform", game_object_id, snapshot)

def snapshot_live_native_component(game_object_id: int, type_name: str,
                                   ordinal: int = 0) -> str:
    return _resolve_and_snap("native", game_object_id,
                             type_name=type_name, ordinal=ordinal)

def restore_live_native_component(game_object_id: int, type_name: str,
                                  ordinal: int, snapshot: str) -> None:
    _resolve_and_restore("native", game_object_id, snapshot,
                         type_name=type_name, ordinal=ordinal)

def snapshot_live_py_component(game_object_id: int, type_name: str,
                               ordinal: int = 0, script_guid: str = "") -> str:
    return _resolve_and_snap("py", game_object_id,
                             type_name=type_name, ordinal=ordinal,
                             script_guid=script_guid)

def restore_live_py_component(game_object_id: int, type_name: str,
                              ordinal: int, snapshot: str,
                              script_guid: str = "") -> None:
    _resolve_and_restore("py", game_object_id, snapshot,
                         type_name=type_name, ordinal=ordinal,
                         script_guid=script_guid)

def snapshot_live_renderstack_component(game_object_id: int, type_name: str,
                                        ordinal: int = 0,
                                        script_guid: str = "") -> str:
    return _resolve_and_snap("renderstack", game_object_id,
                             type_name=type_name, ordinal=ordinal,
                             script_guid=script_guid)

def restore_live_renderstack_component(game_object_id: int, type_name: str,
                                       ordinal: int, snapshot: str,
                                       script_guid: str = "") -> None:
    _resolve_and_restore("renderstack", game_object_id, snapshot,
                         type_name=type_name, ordinal=ordinal,
                         script_guid=script_guid)
