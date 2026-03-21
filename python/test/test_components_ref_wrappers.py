"""Tests for InfEngine.components.ref_wrappers — GameObjectRef, ComponentRef."""

import copy

from InfEngine.components.ref_wrappers import GameObjectRef, ComponentRef


# ══════════════════════════════════════════════════════════════════════
# GameObjectRef
# ══════════════════════════════════════════════════════════════════════

class TestGameObjectRef:
    def test_empty_ref_is_falsy(self):
        ref = GameObjectRef()
        assert not ref
        assert ref.persistent_id == 0

    def test_from_persistent_id(self):
        ref = GameObjectRef(persistent_id=42)
        assert ref.persistent_id == 42

    def test_eq_none_when_empty(self):
        ref = GameObjectRef()
        assert ref == None  # noqa: E711 — intentional None comparison

    def test_eq_by_persistent_id(self):
        a = GameObjectRef(persistent_id=7)
        b = GameObjectRef(persistent_id=7)
        assert a == b

    def test_neq_by_persistent_id(self):
        a = GameObjectRef(persistent_id=7)
        b = GameObjectRef(persistent_id=8)
        assert a != b

    def test_hash_by_persistent_id(self):
        a = GameObjectRef(persistent_id=7)
        b = GameObjectRef(persistent_id=7)
        assert hash(a) == hash(b)

    def test_copy(self):
        ref = GameObjectRef(persistent_id=99)
        ref2 = copy.copy(ref)
        assert ref2.persistent_id == 99
        assert ref2 is not ref

    def test_deepcopy(self):
        ref = GameObjectRef(persistent_id=99)
        ref2 = copy.deepcopy(ref)
        assert ref2.persistent_id == 99
        assert ref2 is not ref

    def test_repr_none(self):
        ref = GameObjectRef()
        r = repr(ref)
        assert "None" in r
        assert "0" in r

    def test_getattr_returns_none_when_empty(self):
        ref = GameObjectRef()
        assert ref.name is None
        assert ref.transform is None


# ══════════════════════════════════════════════════════════════════════
# ComponentRef
# ══════════════════════════════════════════════════════════════════════

class TestComponentRef:
    def test_empty_ref_is_falsy(self):
        ref = ComponentRef()
        assert not ref
        assert ref.go_id == 0
        assert ref.component_type == ""

    def test_with_go_id_and_type(self):
        ref = ComponentRef(go_id=10, component_type="Light")
        assert ref.go_id == 10
        assert ref.component_type == "Light"

    def test_eq_by_go_id_and_type(self):
        a = ComponentRef(go_id=5, component_type="Camera")
        b = ComponentRef(go_id=5, component_type="Camera")
        assert a == b

    def test_neq_different_type(self):
        a = ComponentRef(go_id=5, component_type="Camera")
        b = ComponentRef(go_id=5, component_type="Light")
        assert a != b

    def test_hash(self):
        a = ComponentRef(go_id=5, component_type="Camera")
        b = ComponentRef(go_id=5, component_type="Camera")
        assert hash(a) == hash(b)

    def test_copy(self):
        ref = ComponentRef(go_id=3, component_type="X")
        ref2 = copy.copy(ref)
        assert ref2.go_id == 3
        assert ref2.component_type == "X"

    def test_serialize_round_trip(self):
        ref = ComponentRef(go_id=42, component_type="Rigidbody")
        data = ref._serialize()
        assert "__component_ref__" in data

        restored = ComponentRef._from_dict(data["__component_ref__"])
        assert restored.go_id == 42
        assert restored.component_type == "Rigidbody"

    def test_display_name_unresolved(self):
        ref = ComponentRef()
        assert ref.display_name == "None"

    def test_repr_unresolved(self):
        ref = ComponentRef(go_id=1, component_type="Test")
        r = repr(ref)
        assert "None" in r
        assert "Test" in r

    def test_getattr_returns_none_when_unresolved(self):
        ref = ComponentRef(go_id=1, component_type="Missing")
        assert ref.some_method is None
