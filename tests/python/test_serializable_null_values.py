"""Explicit nulls round-trip separately from missing-field defaults."""

from infernux.components import SerializableObject, FieldType, serialized_field


class NullableData(SerializableObject):
    value = serialized_field(default=17)


class NullValueOwner(SerializableObject):
    child = serialized_field(default_factory=NullableData)
    children = serialized_field(
        default=[None, NullableData()],
        field_type=FieldType.LIST,
        element_type=FieldType.SERIALIZABLE_OBJECT,
        element_class=NullableData,
    )


def test_explicit_null_and_missing_defaults_survive_document_reopen(tmp_path):
    import json

    owner = NullValueOwner()
    owner.child = None
    path = tmp_path / "nullable.json"
    path.write_text(json.dumps(owner._serialize()), encoding="utf-8")
    document = json.loads(path.read_text(encoding="utf-8"))
    restored = SerializableObject._deserialize(document)
    assert restored.child is None
    assert restored.children[0] is None and restored.children[1].value == 17
    assert restored._serialize() == document
    document["fields"].pop("child")
    missing = SerializableObject._deserialize(document)
    assert isinstance(missing.child, NullableData) and missing.child.value == 17
