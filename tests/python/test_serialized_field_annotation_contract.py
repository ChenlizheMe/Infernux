"""Typed descriptors use annotations, not the contents of their default list."""
from enum import Enum

import pytest

from infernux.components import InxComponent, SerializableObject, serialized_field
from infernux.components.fields import FieldType, get_serialized_fields
from infernux.components._component_registration import candidate_component_registration_scope
from infernux.components.field_schema_compiler import FieldSchemaError


class AnnotationMode(Enum):
    IDLE = 0
    CHASE = 1


class AnnotationStats(SerializableObject):
    health: int = serialized_field(default=100)


def declare(base, annotation, default, **options):
    with candidate_component_registration_scope():
        return type("TypedDescriptorProbe", (base,), {
            "__module__": __name__,
            "__annotations__": {"value": annotation},
            "value": serialized_field(default=default, **options),
        })


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
@pytest.mark.parametrize("annotation,element_type,element_class,enum_type", [
    (list[int], FieldType.INT, None, None),
    (list[str], FieldType.STRING, None, None),
    (list[AnnotationMode], FieldType.ENUM, None, AnnotationMode),
    (list[AnnotationStats], FieldType.SERIALIZABLE_OBJECT, AnnotationStats, None),
])
def test_empty_typed_list_has_complete_element_contract(base, annotation, element_type, element_class, enum_type):
    owner = declare(base, annotation, [], tooltip="authored list", hidden=True)
    metadata = get_serialized_fields(owner)["value"]
    assert metadata.field_type is FieldType.LIST
    assert metadata.element_type is element_type
    assert metadata.element_class is element_class
    assert metadata.enum_type is enum_type
    assert metadata.tooltip == "authored list" and metadata.hidden
    first, second = owner(), owner()
    assert first.value == second.value == []
    assert first.value is not second.value
    assert owner._field_schemas_["value"].attributes["element_type"] == str(element_type)


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
def test_numeric_annotation_controls_kind_and_default_coercion(base):
    owner = declare(base, float, 1, range=(0, 10), header="Movement")
    metadata = get_serialized_fields(owner)["value"]
    assert metadata.field_type is FieldType.FLOAT
    assert type(owner().value) is float and owner().value == 1.0
    assert metadata.header == "Movement" and metadata.range == (0, 10)


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
def test_nonempty_enum_list_keeps_the_enum_declaration(base):
    owner = declare(base, list[AnnotationMode], [AnnotationMode.CHASE])
    assert owner().value == [AnnotationMode.CHASE]
    schema = owner._field_schemas_["value"]
    assert schema.attributes["enum"]["type_id"].endswith(":AnnotationMode")


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
def test_annotated_list_rejects_default_of_the_wrong_element_kind(base):
    with pytest.raises(FieldSchemaError, match="invalid|INT"):
        declare(base, list[int], ["wrong"])


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
def test_explicit_field_and_element_options_remain_authoritative(base):
    owner = declare(base, list[str], [1], field_type=FieldType.LIST, element_type=FieldType.INT)
    metadata = get_serialized_fields(owner)["value"]
    assert metadata.element_type is FieldType.INT and owner().value == [1]
    scalar = declare(base, float, 1, field_type=FieldType.INT)
    assert get_serialized_fields(scalar)["value"].field_type is FieldType.INT
    assert type(scalar().value) is int
