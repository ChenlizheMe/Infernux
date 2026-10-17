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
        return type(
            "TypedDescriptorProbe",
            (base,),
            {
                "__module__": __name__,
                "__annotations__": {"value": annotation},
                "value": serialized_field(default=default, **options),
            },
        )


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
def test_typed_declarations_configure_save_and_reopen(tmp_path, scene, base):
    import json

    class TypedLists(base):
        ratio: float = 1
        numbers: list[int] = serialized_field(default=[])
        labels: list[str] = serialized_field(default=[])
        modes: list[AnnotationMode] = serialized_field(default=[AnnotationMode.CHASE])
        members: list[AnnotationStats] = serialized_field(default=[])
        override: list[str] = serialized_field(default=[1], field_type=FieldType.LIST, element_type=FieldType.INT)
        scalar: float = serialized_field(default=1, field_type=FieldType.INT)

    if base is InxComponent:
        first = scene.create_game_object("Authored typed fields").add_py_component(TypedLists())
        untouched = scene.create_game_object("Default typed fields").add_py_component(TypedLists())
    else:
        first, untouched = TypedLists(), TypedLists()
    assert type(first.ratio) is float and first.ratio == 1.0
    assert type(first.scalar) is int and first.override == [1]
    first.numbers.append(7)
    first.labels.append("saved")
    first.members.append(AnnotationStats(health=31))
    assert untouched.numbers == untouched.labels == untouched.members == []
    fields = get_serialized_fields(TypedLists)
    assert fields["numbers"].element_type is FieldType.INT
    assert fields["labels"].element_type is FieldType.STRING
    assert fields["modes"].enum_type is AnnotationMode
    assert fields["members"].element_class is AnnotationStats
    path = tmp_path / (base.__name__ + ".json")
    document = first._serialize_fields_document() if base is InxComponent else first._serialize()
    path.write_text(json.dumps(document), encoding="utf-8")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if base is InxComponent:
        restored = scene.create_game_object("Reopened typed fields").add_py_component(TypedLists())
        restored._deserialize_fields_document(saved)
        assert restored._serialize_fields_document() == saved
    else:
        restored = SerializableObject._deserialize(saved)
        assert type(restored) is TypedLists and restored._serialize() == saved
    assert restored.numbers == [7] and restored.labels == ["saved"]
    assert restored.modes == [AnnotationMode.CHASE] and restored.members[0].health == 31
    restored.members[0].health = 99
    assert first.members[0].health == 31


@pytest.mark.parametrize("base", [InxComponent])
def test_annotated_list_rejects_default_of_the_wrong_element_kind(base):
    with pytest.raises(FieldSchemaError, match="invalid|INT"):
        declare(base, list[int], ["wrong"])
