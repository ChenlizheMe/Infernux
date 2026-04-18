"""
Shared serialization helpers for InxComponent and SerializableObject.

Eliminates the duplicate dict-key ref dispatch and asset-ref creation
boilerplate that was copy-pasted between component.py and
serializable_object.py.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .serialized_field import FieldMetadata


# ──────────────────────────────────────────────────────────────────────
# Asset ref serialization table (type → dict key)
# ──────────────────────────────────────────────────────────────────────

def _serialize_asset_ref(value: Any) -> Optional[dict]:
    """Serialize an asset-ref-like object to its canonical dict form.

    Returns None if *value* is not a recognised asset-ref type.
    """
    from Infernux.core.asset_ref import (
        TextureRef, ShaderRef, AssetRefBase,
        get_all_asset_type_configs,
    )

    # Fixed non-ASSET ref types (TEXTURE / SHADER have their own FieldType)
    _FIXED_KEY_MAP: list[Tuple[type, str]] = [
        (TextureRef, "__texture_ref__"),
        (ShaderRef, "__shader_ref__"),
    ]

    for ref_type, dict_key in _FIXED_KEY_MAP:
        if isinstance(value, ref_type):
            d: dict = {dict_key: value.guid}
            if value.path_hint:
                d["__path_hint__"] = value.path_hint
            return d

    # Dynamic ASSET-typed refs from the registry
    if isinstance(value, AssetRefBase):
        for _name, cfg in get_all_asset_type_configs().items():
            if isinstance(value, cfg["ref_class"]):
                d = {cfg["dict_key"]: value.guid}
                if value.path_hint:
                    d["__path_hint__"] = value.path_hint
                return d

    return None


# ──────────────────────────────────────────────────────────────────────
# Vector serialization
# ──────────────────────────────────────────────────────────────────────

def serialize_vec(value: Any) -> Optional[list]:
    """Serialize a vec-like object (has x, y, [z, [w]]) to a float list.

    Returns None if *value* is not vec-like.
    """
    if hasattr(value, "x") and hasattr(value, "y"):
        if hasattr(value, "z"):
            if hasattr(value, "w"):
                return [float(value.x), float(value.y), float(value.z), float(value.w)]
            return [float(value.x), float(value.y), float(value.z)]
        return [float(value.x), float(value.y)]
    return None


# ──────────────────────────────────────────────────────────────────────
# Dict-key ref deserialization dispatch
# ──────────────────────────────────────────────────────────────────────

def deserialize_dict_ref(value: dict) -> Any:
    """Attempt to deserialize a dict into the appropriate ref wrapper.

    Returns the deserialized ref object, or *value* unchanged if no
    recognised dict-key marker was found.
    """
    if "__game_object__" in value:
        from .ref_wrappers import GameObjectRef
        return GameObjectRef(persistent_id=int(value["__game_object__"]))

    if "__prefab_ref__" in value:
        from .ref_wrappers import PrefabRef
        return PrefabRef(guid=value["__prefab_ref__"],
                         path_hint=value.get("__path_hint__", ""))

    if "__material_ref__" in value:
        from .ref_wrappers import MaterialRef
        return MaterialRef(guid=value["__material_ref__"],
                           path_hint=value.get("__path_hint__", ""))

    if "__texture_ref__" in value:
        from Infernux.core.asset_ref import TextureRef
        return TextureRef(guid=value["__texture_ref__"],
                          path_hint=value.get("__path_hint__", ""))

    if "__shader_ref__" in value:
        from Infernux.core.asset_ref import ShaderRef
        return ShaderRef(guid=value["__shader_ref__"],
                         path_hint=value.get("__path_hint__", ""))

    if "__audio_clip_ref__" in value:
        from Infernux.core.asset_ref import AudioClipRef
        return AudioClipRef(guid=value["__audio_clip_ref__"],
                            path_hint=value.get("__path_hint__", ""))

    # Dynamic ASSET-typed refs from the registry
    from Infernux.core.asset_ref import get_all_asset_type_configs
    for _name, cfg in get_all_asset_type_configs().items():
        dk = cfg["dict_key"]
        if dk in value:
            return cfg["ref_class"](guid=value[dk],
                                    path_hint=value.get("__path_hint__", ""))

    if "__component_ref__" in value:
        from .ref_wrappers import ComponentRef
        return ComponentRef._from_dict(value["__component_ref__"])

    if "__serializable_type__" in value:
        from .serializable_object import SerializableObject
        return SerializableObject._deserialize(value)

    return value


# ──────────────────────────────────────────────────────────────────────
# Null-value factory for ref field types
# ──────────────────────────────────────────────────────────────────────

def make_null_ref(field_type, field_meta=None) -> Any:
    """Return an empty/null ref for the given FieldType.

    Used when a serialized value is None but the field type implies a
    non-None wrapper (e.g. GameObjectRef(persistent_id=0)).
    """
    from .serialized_field import FieldType

    if field_type == FieldType.GAME_OBJECT:
        from .ref_wrappers import GameObjectRef
        return GameObjectRef(persistent_id=0)
    if field_type == FieldType.MATERIAL:
        from .ref_wrappers import MaterialRef
        return MaterialRef(guid="")
    if field_type == FieldType.TEXTURE:
        from Infernux.core.asset_ref import TextureRef
        return TextureRef()
    if field_type == FieldType.SHADER:
        from Infernux.core.asset_ref import ShaderRef
        return ShaderRef()
    if field_type == FieldType.ASSET:
        asset_type = getattr(field_meta, "asset_type", None) or "AudioClip"
        from Infernux.core.asset_ref import get_asset_type_config
        cfg = get_asset_type_config(asset_type)
        if cfg:
            return cfg["ref_class"]()
        from Infernux.core.asset_ref import AudioClipRef
        return AudioClipRef()
    if field_type == FieldType.COMPONENT:
        from .ref_wrappers import ComponentRef
        comp_type = getattr(field_meta, "component_type", "") or ""
        return ComponentRef(component_type=comp_type)
    if field_type == FieldType.SERIALIZABLE_OBJECT:
        so_cls = getattr(field_meta, "serializable_class", None)
        if so_cls is not None:
            return so_cls()
        return None
    return None
