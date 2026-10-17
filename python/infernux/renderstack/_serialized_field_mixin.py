"""Shared mixin for auto-collecting serialized fields via __init_subclass__."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet


class SerializedFieldCollectorMixin:
    """Mixin that auto-collects class-level serialized fields.

    Subclasses that use this mixin get automatic ``_serialized_fields_``
    population via ``__init_subclass__``.  Override ``_reserved_attrs_``
    in each class hierarchy to exclude certain attribute names.
    """

    _reserved_attrs_: FrozenSet[str] = frozenset()
    _serialized_fields_: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        from infernux.components.fields import _compile_serialized_fields

        reserved = frozenset().union(
            *(base.__dict__.get("_reserved_attrs_", ()) for base in cls.__mro__)
        )
        declarations = cls.__dict__.keys() | cls.__dict__.get("__annotations__", {}).keys()
        reserved |= {name for name in declarations if name.startswith("_")}
        _compile_serialized_fields(cls, reserved_attrs=reserved)
