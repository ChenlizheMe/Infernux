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
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .fields import FieldMetadata

# Global registry: module:qualname → class. Populated by __init_subclass__.
_SERIALIZABLE_REGISTRY: Dict[str, Type["SerializableObject"]] = {}

# The importer owns this temporary write set, just like its private modules.
# Ordinary consumers continue to read the single published registry.
_CANDIDATE_TYPES: ContextVar[tuple[dict[str, type], str] | None] = ContextVar(
    "infernux_candidate_serializable_types", default=None
)


@contextmanager
def _candidate_serializable_scope(types: dict[str, type], module_name: str):
    token = _CANDIDATE_TYPES.set((types, module_name))
    try:
        yield
    finally:
        _CANDIDATE_TYPES.reset(token)


def _publish_serializable_types(
    types: dict[str, type], module_names: set[str],
) -> dict[str, type | None]:
    """Replace the published modules' data types; return their exact before-image."""
    published = {
        identity: cls for identity, cls in types.items()
        if cls.__module__ in module_names
    }
    for identity in published:
        previous = _SERIALIZABLE_REGISTRY.get(identity)
        if previous is not None and previous.__module__ not in module_names:
            raise ValueError(
                f"serialized type ID {identity!r} is owned by module {previous.__module__!r}"
            )
    before = {
        identity: cls for identity, cls in _SERIALIZABLE_REGISTRY.items()
        if cls.__module__ in module_names
    }
    for identity in published:
        before.setdefault(identity, _SERIALIZABLE_REGISTRY.get(identity))
    for identity in before:
        _SERIALIZABLE_REGISTRY.pop(identity, None)
    _SERIALIZABLE_REGISTRY.update(published)
    return before


def _restore_serializable_types(before: dict[str, type | None]) -> None:
    for identity, cls in before.items():
        if cls is None:
            _SERIALIZABLE_REGISTRY.pop(identity, None)
        else:
            _SERIALIZABLE_REGISTRY[identity] = cls


def get_serializable_type_id(value: type | "SerializableObject") -> str:
    """Return the stable current-format identity for a serializable type."""
    value_type = value if isinstance(value, type) else type(value)
    explicit_id = value_type.__dict__.get("__serialized_type_id__", "")
    if explicit_id:
        if not isinstance(explicit_id, str):
            raise TypeError("__serialized_type_id__ must be a string")
        return explicit_id
    return f"{value_type.__module__}:{value_type.__qualname__}"


def get_serializable_class(type_id: str) -> Optional[Type["SerializableObject"]]:
    """Look up a registered SerializableObject subclass by module:qualname."""
    candidate = _CANDIDATE_TYPES.get()
    if candidate is not None and type_id in candidate[0]:
        return candidate[0][type_id]
    return _SERIALIZABLE_REGISTRY.get(type_id)


def get_registered_serializable_types() -> tuple[tuple[str, Type["SerializableObject"]], ...]:
    """Return the published type catalog in stable identity order."""
    return tuple(sorted(_SERIALIZABLE_REGISTRY.items(), key=lambda item: item[0]))


class SerializableObject:
    """Lightweight data container with serialized-field metadata.

    Subclass this to create custom serializable data types that can be used
    as InxComponent field values (scalars or list elements).

    * Field declarations follow the same syntax as InxComponent
      (``serialized_field()``, plain values, type annotations).
    * Instances use the current typed value-document identity for
      polymorphic deserialization.
    * **No undo/dirty tracking** — that is handled at the InxComponent level.
    """

    _serialized_fields_: Dict[str, "FieldMetadata"] = {}
    __serialized_type_id__ = ""

    # ------------------------------------------------------------------
    # Metaclass-style auto-registration
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        current_type_id = get_serializable_type_id(cls)
        if not isinstance(current_type_id, str) or not current_type_id:
            raise ValueError("serialized type ID must be a non-empty string")
        from .fields import _compile_serialized_fields

        _compile_serialized_fields(cls, descriptors=False)
        candidate = _CANDIDATE_TYPES.get()
        if candidate is not None and cls.__module__ == candidate[1]:
            types = candidate[0]
            previous = types.get(current_type_id)
            if previous is not None and previous.__module__ != cls.__module__:
                raise ValueError(f"duplicate candidate serialized type ID {current_type_id!r}")
            types[current_type_id] = cls
        else:
            # Trusted engine modules materialized during a candidate import
            # still own their ordinary declarations, not the project transaction.
            _SERIALIZABLE_REGISTRY[current_type_id] = cls

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __getattribute__(self, name: str):
        if not name.startswith("__"):
            from .fields import get_serialized_fields
            cls = object.__getattribute__(self, "__class__")
            fields = get_serialized_fields(cls)
            meta = fields.get(name)
            if meta is not None:
                from .fields import resolve_runtime_field_value
                data = object.__getattribute__(self, "__dict__")
                raw = data.get(name, meta.default)
                return resolve_runtime_field_value(raw, meta)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value):
        from .fields import get_serialized_fields
        cls = type(self)
        fields = get_serialized_fields(cls)
        meta = fields.get(name)
        if meta is not None:
            from .fields import normalize_runtime_field_value
            value = normalize_runtime_field_value(value, meta)
        object.__setattr__(self, name, value)

    def __init__(self, **kwargs):
        from .fields import copy_serialized_field_default, get_serialized_fields

        fields = get_serialized_fields(self.__class__)
        for name, meta in fields.items():
            if name in kwargs:
                setattr(self, name, kwargs[name])
            else:
                setattr(self, name, copy_serialized_field_default(meta))

    # ------------------------------------------------------------------
    # Serialization helpers (used by InxComponent._serialize_value)
    # ------------------------------------------------------------------

    def _serialize(self) -> dict:
        """Serialize this object to a JSON-friendly dict."""
        from .fields import get_serialized_fields

        fields_document: Dict[str, Any] = {}
        fields = get_serialized_fields(self.__class__)
        from .fields import get_raw_field_value
        for name, meta in fields.items():
            value = get_raw_field_value(self, name)
            fields_document[name] = _serialize_so_value(
                value, f"{get_serializable_type_id(self)}.{name}"
            )
        from .value_document import make_serializable_object
        return make_serializable_object(
            get_serializable_type_id(self),
            fields_document,
        )

    @classmethod
    def _prepare_document(cls, data: dict, path: str = "SerializableObject"):
        """Validate identity and normalize one document to current fields.

        The editor is intentionally forward-only while the data model is in
        rapid development.  Only currently declared field names are consumed;
        removed/renamed/extra keys are ignored and newly declared fields take
        their current defaults.
        """
        from .fields import copy_serialized_field_default, get_serialized_fields
        from .value_document import TYPE_KEY, SERIALIZABLE_OBJECT

        if not isinstance(data, dict):
            raise TypeError(f"{path}: SerializableObject document must be an object")
        if data.get(TYPE_KEY) != SERIALIZABLE_OBJECT:
            raise ValueError(f"{path}: invalid SerializableObject typed document")
        type_id = data.get("type_id")
        if not isinstance(type_id, str) or not type_id:
            raise ValueError(f"{path}: SerializableObject document requires type_id")
        actual_cls = get_serializable_class(type_id)
        if actual_cls is None:
            raise ValueError(f"{path}: unknown SerializableObject type_id {type_id!r}")

        fields = get_serialized_fields(actual_cls)
        fields_document = data.get("fields")
        if not isinstance(fields_document, dict):
            raise TypeError(f"{path}: SerializableObject fields must be an object")
        from .value_codec import VALUE_CODECS

        current_fields = {}
        for name, metadata in fields.items():
            if name in fields_document:
                current_fields[name] = fields_document[name]
            else:
                current_fields[name] = VALUE_CODECS.encode(
                    copy_serialized_field_default(metadata),
                    f"{path}.{name}.default",
                )

        from .fields import validate_serialized_field_document
        validate_serialized_field_document(current_fields, fields, owner_name=type_id)

        for name, meta in fields.items():
            VALUE_CODECS.validate(current_fields[name], meta, f"{path}.{name}")
        return (
            actual_cls,
            fields,
            {
                TYPE_KEY: SERIALIZABLE_OBJECT,
                "type_id": type_id,
                "fields": current_fields,
            },
        )

    @classmethod
    def _validate_document(cls, data: dict, path: str = "SerializableObject"):
        """Validate a complete object graph without constructing an instance."""
        actual_cls, fields, _document = cls._prepare_document(data, path)
        return actual_cls, fields

    @classmethod
    def _deserialize(cls, data: dict) -> "SerializableObject":
        """Validate, decode, and construct one current-schema object."""
        actual_cls, fields, document = cls._prepare_document(data)

        fields_document = document["fields"]
        decoded = {
            name: _deserialize_so_value(
                fields_document[name],
                meta,
                f"{get_serializable_type_id(actual_cls)}.{name}"
            )
            for name, meta in fields.items()
        }
        instance = actual_cls.__new__(actual_cls)
        for name, value in decoded.items():
            setattr(instance, name, value)
        return instance

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        from .fields import get_serialized_fields

        fields = get_serialized_fields(self.__class__)
        return all(
            getattr(self, n, None) == getattr(other, n, None)
            for n in fields
        )

    def __repr__(self):
        from .fields import get_serialized_fields

        fields = get_serialized_fields(self.__class__)
        parts = [f"{n}={getattr(self, n, 'N/A')!r}" for n in fields]
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __deepcopy__(self, memo):
        from .fields import get_raw_field_value, get_serialized_fields

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        fields = get_serialized_fields(cls)
        for name in fields:
            # Serialized asset/reference fields resolve through __getattribute__
            # for runtime use. Copies used by Undo and list editing must retain
            # their raw GUID/path wrapper instead of copying the resolved asset.
            value = get_raw_field_value(self, name)
            setattr(result, name, copy.deepcopy(value, memo))
        return result


def _serialize_so_value(value: Any, path: str = "SerializableObject.value") -> Any:
    """Encode a SerializableObject field with the shared strict registry."""
    from .value_codec import VALUE_CODECS

    return VALUE_CODECS.encode(value, path)


def _deserialize_so_value(
    value: Any, field_meta: Any, path: str = "SerializableObject.value"
) -> Any:
    """Decode a SerializableObject field with the shared strict registry."""
    from .value_codec import VALUE_CODECS

    return VALUE_CODECS.decode(value, field_meta, path)
