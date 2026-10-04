"""Type stubs for infernux.renderstack._serialized_field_mixin."""

from __future__ import annotations

from typing import Any, Dict, FrozenSet


class SerializedFieldCollectorMixin:
    """Mixin that auto-collects class-level serialized fields via ``__init_subclass__``.

    Subclasses get automatic ``_serialized_fields_`` population.
    Override ``_reserved_attrs_`` to exclude certain attribute names.
    """

    _reserved_attrs_: FrozenSet[str]
    _serialized_fields_: Dict[str, Any]

    def __init_subclass__(cls, **kwargs: Any) -> None: ...
