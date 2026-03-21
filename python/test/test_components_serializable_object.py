"""Tests for InfEngine.components.serializable_object — SerializableObject base class."""

from InfEngine.components.serializable_object import (
    SerializableObject,
    _SERIALIZABLE_REGISTRY,
    get_serializable_class,
)
from InfEngine.components.serialized_field import serialized_field, FieldType


# ── Test data classes ──

class Stats(SerializableObject):
    hp: int = serialized_field(default=100)
    mp: float = serialized_field(default=50.0)
    name: str = serialized_field(default="default")


class Nested(SerializableObject):
    inner: int = serialized_field(default=0)


# ══════════════════════════════════════════════════════════════════════
# Registration
# ══════════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_subclass_auto_registered(self):
        assert Stats.__qualname__ in _SERIALIZABLE_REGISTRY

    def test_get_serializable_class(self):
        cls = get_serializable_class(Stats.__qualname__)
        assert cls is Stats

    def test_unknown_returns_none(self):
        assert get_serializable_class("NoSuchClass") is None


# ══════════════════════════════════════════════════════════════════════
# Field metadata collection
# ══════════════════════════════════════════════════════════════════════

class TestFieldCollection:
    def test_serialized_fields_collected(self):
        fields = Stats._serialized_fields_
        assert "hp" in fields
        assert "mp" in fields
        assert "name" in fields

    def test_field_defaults(self):
        fields = Stats._serialized_fields_
        assert fields["hp"].default == 100
        assert fields["mp"].default == 50.0
        assert fields["name"].default == "default"


# ══════════════════════════════════════════════════════════════════════
# Construction
# ══════════════════════════════════════════════════════════════════════

class TestConstruction:
    def test_defaults(self):
        s = Stats()
        assert s.hp == 100
        assert s.mp == 50.0
        assert s.name == "default"

    def test_kwargs(self):
        s = Stats(hp=999, name="hero")
        assert s.hp == 999
        assert s.name == "hero"
        assert s.mp == 50.0  # default

    def test_setattr(self):
        s = Stats()
        s.hp = 42
        assert s.hp == 42


# ══════════════════════════════════════════════════════════════════════
# Serialization round-trip
# ══════════════════════════════════════════════════════════════════════

class TestSerialization:
    def test_serialize_produces_dict_with_type_tag(self):
        s = Stats()
        data = s._serialize()
        assert "__serializable_type__" in data
        assert data["hp"] == 100

    def test_deserialize_restores_values(self):
        s = Stats(hp=42, mp=7.5, name="test")
        data = s._serialize()
        restored = Stats._deserialize(data)
        assert restored.hp == 42
        assert restored.mp == 7.5
        assert restored.name == "test"

    def test_polymorphic_deserialize(self):
        s = Stats(hp=1)
        data = s._serialize()
        # Deserialize through base class resolves to Stats
        restored = SerializableObject._deserialize(data)
        assert type(restored) is Stats
        assert restored.hp == 1


# ══════════════════════════════════════════════════════════════════════
# Equality and repr
# ══════════════════════════════════════════════════════════════════════

class TestDunder:
    def test_eq_same_values(self):
        a = Stats(hp=10, mp=20.0, name="x")
        b = Stats(hp=10, mp=20.0, name="x")
        assert a == b

    def test_neq_different_values(self):
        a = Stats(hp=10)
        b = Stats(hp=20)
        assert a != b

    def test_neq_different_types(self):
        s = Stats()
        n = Nested()
        assert s != n

    def test_repr(self):
        s = Stats()
        r = repr(s)
        assert "Stats" in r
        assert "hp=" in r

    def test_deepcopy(self):
        import copy
        s = Stats(hp=42)
        s2 = copy.deepcopy(s)
        assert s2.hp == 42
        s2.hp = 0
        assert s.hp == 42  # original unchanged
