"""Tests for infernux.components.serializable_object — SerializableObject base class."""

from infernux.components.serializable_object import (
    SerializableObject,
)
from infernux.components.fields import serialized_field, FieldType
from infernux.components.fields import (
    get_serialized_fields,
)


def test_inherited_reference_serializes_raw_identity_not_resolved_object():
    import copy
    from infernux.components.ref_wrappers import GameObjectRef
    from infernux.components.fields import get_raw_field_value

    class Base(SerializableObject):
        target: GameObjectRef
        _target = serialized_field(None, field_type=FieldType.GAME_OBJECT, hidden=True)

    class Derived(Base):
        label = "reference owner"

    original = Derived(target=GameObjectRef(persistent_id=42), _target=GameObjectRef(persistent_id=43))
    document = original._serialize()
    for value in (copy.deepcopy(original), SerializableObject._deserialize(document)):
        assert get_raw_field_value(value, "target").persistent_id == 42
        assert get_raw_field_value(value, "_target").persistent_id == 43
        assert value._serialize() == document


def test_data_annotations_cover_nested_lists_enums_and_string_annotations():
    from enum import Enum

    class Mode(Enum):
        idle = 0
        moving = 1

    class Composite(SerializableObject):
        stats: Stats
        members: list[Stats]
        mode: Mode = Mode.moving

    first, second = Composite(), Composite()
    fields = get_serialized_fields(Composite)
    assert fields["stats"].serializable_class is Stats
    assert fields["members"].element_class is Stats
    assert fields["mode"].enum_type is Mode
    first.stats.hp = 7
    first.members.append(Stats(hp=3))
    assert second.stats.hp == 100 and second.members == []
    restored = SerializableObject._deserialize(first._serialize())
    assert restored.stats.hp == 7 and restored.members[0].hp == 3
    assert restored.mode is Mode.moving

    resolved = type(
        "StringAnnotationData",
        (SerializableObject,),
        {
            "__module__": __name__,
            "__annotations__": {"stats": "Stats", "values": "list[int]", "hidden": "int"},
            "hidden": serialized_field(default=4, hidden=True),
        },
    )
    fields = get_serialized_fields(resolved)
    assert fields["stats"].serializable_class is Stats
    assert fields["values"].element_type == FieldType.INT
    assert fields["hidden"].hidden
    assert resolved()._serialize()["fields"]["hidden"] == 4


# ── Test data classes ──


class Stats(SerializableObject):
    hp: int = serialized_field(default=100)
    mp: float = serialized_field(default=50.0)
    name: str = serialized_field(default="default")


class Nested(SerializableObject):
    inner: int = serialized_field(default=0)
