"""Field identities are declarations, not guessed aliases of Python names."""

import pytest

from Infernux.components import InxComponent, SerializableObject, serialized_field
from Infernux.components._component_registration import candidate_component_registration_scope
from Infernux.components.fields import get_field_schema, get_serialized_fields


def _declare(base, **fields):
    with candidate_component_registration_scope():
        return type("FieldIdentityProbe", (base,), fields)


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
def test_explicit_identity_is_separate_from_authored_name(base):
    owner = _declare(base, speed=serialized_field(2.0, field_id="movement-speed"), count=1)
    fields = get_serialized_fields(owner)
    assert fields["speed"].name == "speed"
    assert fields["speed"].field_id == "movement-speed"
    assert get_field_schema(owner, "speed").attributes["field_id"] == "movement-speed"
    assert get_field_schema(owner, "count").attributes["field_id"] == "count"
    assert not hasattr(owner(), "movement-speed")


@pytest.mark.parametrize("identity", ["", " ", " padded", "padded ", 7, False, []])
def test_invalid_explicit_identity_is_rejected_at_declaration(identity):
    with pytest.raises(ValueError, match="field_id"):
        serialized_field(1, field_id=identity)


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
def test_duplicate_identity_is_rejected_including_default_name(base):
    with pytest.raises(ValueError, match="duplicate field_id 'speed'"):
        _declare(base, speed=1, velocity=serialized_field(2, field_id="speed"))


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
def test_inherited_identity_is_validated_without_mutating_parent(base):
    parent = _declare(base, speed=serialized_field(1.0, field_id="movement-speed"))
    original = get_field_schema(parent, "speed")
    with pytest.raises(ValueError, match="duplicate field_id 'movement-speed'"):
        _declare(parent, other=serialized_field(2.0, field_id="movement-speed"))
    child = _declare(parent, speed=serialized_field(3.0, field_id="movement-speed"))
    assert child().speed == 3.0
    assert parent().speed == 1.0
    assert get_field_schema(parent, "speed") is original
    assert get_field_schema(child, "speed").attributes["field_id"] == "movement-speed"


def test_data_document_keeps_authored_keys_not_semantic_ids():
    owner = _declare(SerializableObject, speed=serialized_field(2.0, field_id="movement-speed"))
    value = owner()
    value.speed = 7.0
    document = value._serialize()
    assert document["fields"]["speed"] == 7.0
    assert "movement-speed" not in document["fields"]
    assert SerializableObject._deserialize(document).speed == 7.0


@pytest.mark.parametrize("base", [InxComponent, SerializableObject])
def test_multiple_inheritance_cannot_merge_distinct_fields_with_the_same_id(base):
    left = _declare(base, first=serialized_field(1, field_id="shared"))
    right = _declare(base, second=serialized_field(2, field_id="shared"))
    with candidate_component_registration_scope(), pytest.raises(ValueError, match="duplicate field_id"):
        type("InvalidIdentityMerge", (left, right), {})


def test_rejected_data_declaration_does_not_replace_registered_type():
    from Infernux.components.serializable_object import get_serializable_class, get_serializable_type_id

    existing = _declare(SerializableObject, value=1)
    with pytest.raises(ValueError, match="duplicate field_id"):
        _declare(SerializableObject, value=1, other=serialized_field(2, field_id="value"))
    assert get_serializable_class(get_serializable_type_id(existing)) is existing
