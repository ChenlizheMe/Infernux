"""Immutable field descriptions shared by runtime and editor consumers.

This module contains values only: no drawers, Undo commands or live instances.
Attribute values are encoded metadata, not Python callbacks or engine objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any


def _freeze(value: Any, path: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path}: schema numbers must be finite")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"{path}: schema keys must be strings")
        return MappingProxyType({key: _freeze(item, f"{path}.{key}") for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{path}: unsupported schema value {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FieldSchema:
    """Static, deeply immutable structure for one serialized property."""

    property_path: str
    value_type: str
    read_only: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path = str(self.property_path or "").strip()
        value_type = str(self.value_type or "").strip()
        if not path:
            raise ValueError("serialized property path must not be empty")
        if not value_type:
            raise ValueError("serialized property type must not be empty")
        if type(self.read_only) is not bool:
            raise TypeError("serialized property read_only must be a boolean")
        if not isinstance(self.attributes, Mapping):
            raise TypeError("serialized property attributes must be a mapping")
        object.__setattr__(self, "property_path", path)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "attributes", _freeze(self.attributes, path))

    def __deepcopy__(self, memo: dict[int, Any]) -> FieldSchema:
        return self

    def to_document(self) -> dict[str, Any]:
        """Return detached metadata suitable for catalog/transport encoding."""
        return {
            "property_path": self.property_path,
            "value_type": self.value_type,
            "read_only": self.read_only,
            "attributes": _thaw(self.attributes),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> FieldSchema:
        if set(document) != {"property_path", "value_type", "read_only", "attributes"}:
            raise ValueError("invalid serialized field schema document")
        return cls(**document)


def get_native_field_schemas(type_guid: str) -> tuple[FieldSchema, ...]:
    """Read every declared native field in semantic declaration order."""
    from Infernux.lib import _Infernux

    document = _Infernux._semantic_catalog_snapshot().type_document(type_guid)
    return tuple(
        FieldSchema.from_document(field_document)
        for field_document in document["fields"]
    )


def get_native_field_schema(type_guid: str, field_id: str) -> FieldSchema:
    """Read a declared native field; absent types/fields are not inferred from values."""
    for schema in get_native_field_schemas(type_guid):
        if schema.attributes["field_id"] == field_id:
            return schema
    raise KeyError(f"{type_guid}.{field_id}")


__all__ = ["FieldSchema", "get_native_field_schema", "get_native_field_schemas"]
