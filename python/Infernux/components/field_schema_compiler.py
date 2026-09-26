"""Compile authored field metadata into immutable, value-only descriptions.

The common value codec owns defaults. Inspector callbacks remain on the
declaration and are never included in a transportable schema.
"""

from __future__ import annotations

from Infernux.field_schema import FieldSchema

from .fields import FieldMetadata, FieldType, copy_serialized_field_default
from .value_codec import VALUE_CODECS


class FieldSchemaError(ValueError):
    def __init__(self, code: str, path: str, detail: str):
        self.code = code
        self.path = path
        super().__init__(f"{path}: {detail}")


def compile_field_schema(metadata: FieldMetadata, path: str) -> FieldSchema:
    """Compile one declaration without constructing a component instance."""
    if not metadata.name:
        raise FieldSchemaError("invalid_field_identity", path, "field declaration requires a name")
    kind = metadata.field_type
    if kind is FieldType.UNKNOWN:
        raise FieldSchemaError("unsupported_value_kind", path, "UNKNOWN cannot describe a semantic field")

    attributes = {
        "field_id": metadata.field_id if metadata.field_id is not None else metadata.name,
        "serialized": True,
        "hidden": metadata.hidden,
        "former_names": metadata.former_names,
        "range": metadata.range,
        "tooltip": metadata.tooltip,
        "display_name_key": metadata.display_name_key,
        "header": metadata.header,
        "space": metadata.space,
        "group": metadata.group,
        "info_text": metadata.info_text,
        "multiline": metadata.multiline,
        "slider": metadata.slider,
        "drag_speed": metadata.drag_speed,
        "required_component": metadata.required_component,
        "component_type": metadata.component_type,
        "asset_type": metadata.asset_type,
        "hdr": metadata.hdr,
        "curve_non_negative": metadata.curve_non_negative,
        "inspector_conditional_visibility": metadata.visible_when is not None,
    }
    element = metadata.element_type
    if kind is FieldType.LIST:
        if element is None or element is FieldType.UNKNOWN:
            raise FieldSchemaError("invalid_element_type", path, "LIST requires a declared element type")
        attributes["element_type"] = str(element)

    if kind is FieldType.ENUM or (kind is FieldType.LIST and element is FieldType.ENUM):
        enum_type = metadata.enum_type
        if not isinstance(enum_type, type) or not getattr(enum_type, "__members__", None):
            raise FieldSchemaError("invalid_enum_type", path, "ENUM requires a resolved enum declaration")
        from Infernux.lib import _Infernux

        enum_id = (
            f"native:infernux.{enum_type.__name__}"
            if getattr(_Infernux, enum_type.__name__, None) is enum_type
            else f"{enum_type.__module__}:{enum_type.__qualname__}"
        )
        attributes["enum"] = {
            "type_id": enum_id,
            "members": [
                {"name": name, "value": VALUE_CODECS.encode(member.value, f"{path}.{name}")}
                for name, member in enum_type.__members__.items()
            ],
            "labels": metadata.enum_labels,
        }

    if kind is FieldType.SERIALIZABLE_OBJECT or (
        kind is FieldType.LIST and element is FieldType.SERIALIZABLE_OBJECT
    ):
        from .serializable_object import SerializableObject, get_serializable_type_id

        nested = metadata.element_class if kind is FieldType.LIST else metadata.serializable_class
        if not isinstance(nested, type) or not issubclass(nested, SerializableObject):
            raise FieldSchemaError("invalid_nested_type", path, "nested data requires a resolved SerializableObject type")
        attributes["serializable_type"] = get_serializable_type_id(nested)

    try:
        default = copy_serialized_field_default(metadata)
        document = VALUE_CODECS.encode(default, f"{path}.default")
        VALUE_CODECS.validate(document, metadata, f"{path}.default")
    except (TypeError, ValueError) as exc:
        raise FieldSchemaError("invalid_default", path, str(exc)) from exc
    attributes["default"] = document
    return FieldSchema(path, str(kind), read_only=metadata.readonly, attributes=attributes)
