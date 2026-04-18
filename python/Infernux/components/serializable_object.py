"""
SerializableObject — Base class for custom serializable data objects.

Similar to Unity's ``[Serializable]`` attribute on plain C# classes.
Supports the same ``serialized_field()`` descriptors as InxComponent, but
without lifecycle methods or undo/dirty tracking.  Can be nested inside
InxComponent fields or other SerializableObjects.

Example::

    from Infernux.components import SerializableObject, serialized_field

    class Stats(SerializableObject):
        hp: int = serialized_field(default=100)
        mp: float = serialized_field(default=50.0)
        name: str = serialized_field(default="default")

    class Enemy(InxComponent):
        stats: Stats = serialized_field(default=Stats())
        allies: list = list_field(element_type=FieldType.SERIALIZABLE_OBJECT, element_class=Stats)
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional, Type, TYPE_CHECKING
from Infernux.debug import Debug

if TYPE_CHECKING:
    from .serialized_field import FieldMetadata

_log = logging.getLogger("Infernux.serialize")

# Global registry: qualname → class.  Populated by __init_subclass__.
_SERIALIZABLE_REGISTRY: Dict[str, Type["SerializableObject"]] = {}


def get_serializable_class(qualname: str) -> Optional[Type["SerializableObject"]]:
    """Look up a registered SerializableObject subclass by qualname."""
    return _SERIALIZABLE_REGISTRY.get(qualname)


class SerializableObject:
    """Lightweight data container with serialized-field metadata.

    Subclass this to create custom serializable data types that can be used
    as InxComponent field values (scalars or list elements).

    * Field declarations follow the same syntax as InxComponent
      (``serialized_field()``, plain values, type annotations).
    * Instances are stored/restored as plain dicts with a
      ``__serializable_type__`` tag for polymorphic deserialization.
    * **No undo/dirty tracking** — that is handled at the InxComponent level.
    """

    _serialized_fields_: Dict[str, "FieldMetadata"] = {}

    # ------------------------------------------------------------------
    # Metaclass-style auto-registration
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls._serialized_fields_ = {}
        _SERIALIZABLE_REGISTRY[cls.__qualname__] = cls

        own_annotations = cls.__dict__.get('__annotations__', {})

        from .serialized_field import (
            FieldMetadata,
            SerializedFieldDescriptor,
            infer_field_type_from_value,
            resolve_annotation,
            HiddenField,
        )

        for attr_name in list(cls.__dict__):
            if attr_name.startswith("_"):
                continue

            attr = cls.__dict__[attr_name]

            if callable(attr) or isinstance(attr, (property, classmethod, staticmethod)):
                continue

            if isinstance(attr, HiddenField):
                continue

            if isinstance(attr, SerializedFieldDescriptor):
                meta = attr.metadata
                meta.name = attr_name
                cls._serialized_fields_[attr_name] = meta
                # Replace the heavy descriptor with None; __init__ will set
                # plain instance attributes from defaults.
                setattr(cls, attr_name, None)

            elif isinstance(attr, FieldMetadata):
                attr.name = attr_name
                cls._serialized_fields_[attr_name] = attr
                setattr(cls, attr_name, None)

            elif attr is None:
                ann = own_annotations.get(attr_name)
                if ann is not None:
                    meta = resolve_annotation(ann)
                    if meta is not None:
                        meta.name = attr_name
                        cls._serialized_fields_[attr_name] = meta
                        setattr(cls, attr_name, None)

            else:
                from enum import Enum as _Enum

                field_type = infer_field_type_from_value(attr)
                enum_type = type(attr) if isinstance(attr, _Enum) else None
                meta = FieldMetadata(
                    name=attr_name,
                    field_type=field_type,
                    default=attr,
                    enum_type=enum_type,
                )
                cls._serialized_fields_[attr_name] = meta
                # Leave the class attribute as-is (plain default value)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __getattribute__(self, name: str):
        if not name.startswith("_"):
            cls = object.__getattribute__(self, "__class__")
            fields = getattr(cls, "_serialized_fields_", {})
            meta = fields.get(name)
            if meta is not None:
                from .serialized_field import resolve_runtime_field_value
                data = object.__getattribute__(self, "__dict__")
                raw = data.get(name, meta.default)
                return resolve_runtime_field_value(raw, meta)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        cls = type(self)
        fields = getattr(cls, "_serialized_fields_", {})
        meta = fields.get(name)
        if meta is not None:
            from .serialized_field import normalize_runtime_field_value
            value = normalize_runtime_field_value(value, meta)
        object.__setattr__(self, name, value)

    def __init__(self, **kwargs):
        from .serialized_field import get_serialized_fields

        fields = get_serialized_fields(self.__class__)
        for name, meta in fields.items():
            if name in kwargs:
                setattr(self, name, kwargs[name])
            else:
                try:
                    setattr(self, name, copy.deepcopy(meta.default))
                except Exception:
                    setattr(self, name, meta.default)

    # ------------------------------------------------------------------
    # Serialization helpers (used by InxComponent._serialize_value)
    # ------------------------------------------------------------------

    def _serialize(self) -> dict:
        """Serialize this object to a JSON-friendly dict."""
        from .serialized_field import get_serialized_fields

        data: Dict[str, Any] = {"__serializable_type__": self.__class__.__qualname__}
        fields = get_serialized_fields(self.__class__)
        from .serialized_field import get_raw_field_value
        for name, meta in fields.items():
            value = get_raw_field_value(self, name)
            data[name] = _serialize_so_value(value)
        return data

    @classmethod
    def _deserialize(cls, data: dict) -> "SerializableObject":
        """Create an instance from a serialized dict.

        Uses ``__serializable_type__`` to resolve the concrete class from
        the global registry, falling back to *cls* when the tag is missing.
        """
        from .serialized_field import get_serialized_fields

        type_name = data.get("__serializable_type__", cls.__qualname__)
        actual_cls = _SERIALIZABLE_REGISTRY.get(type_name, cls)

        instance = actual_cls.__new__(actual_cls)
        # Initialize defaults first
        fields = get_serialized_fields(actual_cls)
        for name, meta in fields.items():
            try:
                setattr(instance, name, copy.deepcopy(meta.default))
            except Exception:
                setattr(instance, name, meta.default)

        # Apply stored values
        for name, value in data.items():
            if name.startswith("__"):
                continue
            if name in fields:
                setattr(instance, name, _deserialize_so_value(value, fields[name]))
        return instance

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        from .serialized_field import get_serialized_fields

        fields = get_serialized_fields(self.__class__)
        return all(
            getattr(self, n, None) == getattr(other, n, None)
            for n in fields
        )

    def __repr__(self):
        from .serialized_field import get_serialized_fields

        fields = get_serialized_fields(self.__class__)
        parts = [f"{n}={getattr(self, n, 'N/A')!r}" for n in fields]
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __deepcopy__(self, memo):
        from .serialized_field import get_serialized_fields

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        fields = get_serialized_fields(cls)
        for name in fields:
            value = getattr(self, name, None)
            setattr(result, name, copy.deepcopy(value, memo))
        return result


# ======================================================================
# Private serialization helpers (mirror InxComponent._serialize_value
# but without the heavy InxComponent-specific imports)
# ======================================================================

def _serialize_so_value(value: Any) -> Any:
    """Recursively serialize a value owned by a SerializableObject."""
    if isinstance(value, (bool, int, float, str, type(None))):
        return value

    if isinstance(value, list):
        return [_serialize_so_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_so_value(v) for k, v in value.items()}

    # Nested SerializableObject
    if isinstance(value, SerializableObject):
        return value._serialize()

    # Enum
    from enum import Enum as _Enum
    if isinstance(value, _Enum):
        return {"__enum__": type(value).__qualname__, "name": value.name}

    # GameObjectRef / PrefabRef / MaterialRef
    from .ref_wrappers import GameObjectRef, PrefabRef
    from Infernux.core.asset_ref import MaterialRef
    if isinstance(value, GameObjectRef):
        return {"__game_object__": value.persistent_id}
    if isinstance(value, PrefabRef):
        return value._serialize()
    if isinstance(value, MaterialRef):
        d: dict = {"__material_ref__": value.guid}
        if value._path_hint:
            d["__path_hint__"] = value._path_hint
        return d

    # Asset refs (TextureRef, ShaderRef, AudioClipRef) — shared helper
    from ._serialize_helpers import _serialize_asset_ref
    ref_dict = _serialize_asset_ref(value)
    if ref_dict is not None:
        return ref_dict

    # ComponentRef
    from .ref_wrappers import ComponentRef
    if isinstance(value, ComponentRef):
        return value._serialize()

    # Vec types — shared helper
    from ._serialize_helpers import serialize_vec
    vec_list = serialize_vec(value)
    if vec_list is not None:
        return vec_list

    _log.warning(
        "Cannot serialize SO value of type %s — returning None.",
        type(value).__name__,
    )
    return None


def _deserialize_so_value(value: Any, field_meta) -> Any:
    """Recursively deserialize a value for a SerializableObject field."""
    from .serialized_field import FieldType
    from ._serialize_helpers import make_null_ref, deserialize_dict_ref
    from Infernux.math import Vector2, Vector3, vec4f

    if hasattr(field_meta, "field_type"):
        field_type = field_meta.field_type
        element_type = getattr(field_meta, "element_type", None)
    else:
        field_type = field_meta
        element_type = None

    # Null values for ref types → return a null ref wrapper (not raw None)
    if value is None:
        return make_null_ref(field_type, field_meta)

    if field_type == FieldType.SERIALIZABLE_OBJECT:
        if isinstance(value, dict) and "__serializable_type__" in value:
            return SerializableObject._deserialize(value)
        so_cls = getattr(field_meta, "serializable_class", None)
        if so_cls and isinstance(value, dict):
            return so_cls._deserialize(value)
        return value

    if field_type == FieldType.COMPONENT:
        if isinstance(value, dict) and "__component_ref__" in value:
            from .ref_wrappers import ComponentRef
            return ComponentRef._from_dict(value["__component_ref__"])
        return value

    if field_type == FieldType.LIST:
        if not isinstance(value, list):
            return []
        if element_type == FieldType.SERIALIZABLE_OBJECT:
            elem_cls = getattr(field_meta, "element_class", None)
            result = []
            for item in value:
                if isinstance(item, dict):
                    if "__serializable_type__" in item:
                        result.append(SerializableObject._deserialize(item))
                    elif elem_cls:
                        result.append(elem_cls._deserialize(item))
                    else:
                        result.append(item)
                else:
                    result.append(item)
            return result
        return [_deserialize_so_value(item, element_type or FieldType.UNKNOWN) for item in value]

    if field_type == FieldType.VEC2:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return Vector2(float(value[0]), float(value[1]))
        return value
    if field_type == FieldType.VEC3:
        if isinstance(value, (list, tuple)) and len(value) >= 3:
            return Vector3(float(value[0]), float(value[1]), float(value[2]))
        return value
    if field_type == FieldType.VEC4:
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            return vec4f(float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        return value

    if field_type == FieldType.ENUM and isinstance(value, dict) and "__enum__" in value:
        enum_cls = getattr(field_meta, "enum_type", None)
        if enum_cls is not None and hasattr(enum_cls, "__getitem__"):
            try:
                return enum_cls[value["name"]]
            except KeyError as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass
        return value

    if isinstance(value, dict):
        return deserialize_dict_ref(value)

    return value
