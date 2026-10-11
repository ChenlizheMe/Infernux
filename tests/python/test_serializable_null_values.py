"""Explicit nulls round-trip separately from missing-field defaults."""
from infernux.components import SerializableObject, FieldType, serialized_field


class NullableData(SerializableObject):
    value = serialized_field(default=17)


class NullValueOwner(SerializableObject):
    child = serialized_field(default_factory=NullableData)
    children = serialized_field(default=[None, NullableData()], field_type=FieldType.LIST,
                                element_type=FieldType.SERIALIZABLE_OBJECT, element_class=NullableData)


def test_explicit_null_replaces_declared_object_default_and_roundtrips():
    owner = NullValueOwner()
    owner.child = None
    document = owner._serialize()
    assert document["fields"]["child"] is None
    restored = SerializableObject._deserialize(document)
    assert restored.child is None
    assert restored._serialize() == document


def test_nullable_list_elements_keep_position_and_nonnull_value():
    original = NullValueOwner()
    document = original._serialize()
    restored = SerializableObject._deserialize(document)
    assert restored.children[0] is None
    assert restored.children[1].value == 17
    assert restored._serialize() == document


def test_absent_field_still_uses_its_declared_object_default():
    document = NullValueOwner()._serialize()
    document["fields"].pop("child")
    restored = SerializableObject._deserialize(document)
    assert isinstance(restored.child, NullableData)
    assert restored.child.value == 17
