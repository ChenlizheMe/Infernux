"""Tests for InfEngine.components.registry — get_type, get_all_types, T accessor."""

from InfEngine.components.component import InfComponent
from InfEngine.components.registry import get_type, get_all_types, T


# ── Test components ──

class _TestCompA(InfComponent):
    pass

class _TestCompB(_TestCompA):
    pass


# ══════════════════════════════════════════════════════════════════════
# get_type
# ══════════════════════════════════════════════════════════════════════

class TestGetType:
    def test_finds_direct_subclass(self):
        cls = get_type("_TestCompA")
        assert cls is _TestCompA

    def test_finds_nested_subclass(self):
        cls = get_type("_TestCompB")
        assert cls is _TestCompB

    def test_returns_none_for_unknown(self):
        assert get_type("NonExistentType12345") is None


# ══════════════════════════════════════════════════════════════════════
# get_all_types
# ══════════════════════════════════════════════════════════════════════

class TestGetAllTypes:
    def test_includes_test_types(self):
        all_types = get_all_types()
        assert "_TestCompA" in all_types
        assert "_TestCompB" in all_types

    def test_values_are_classes(self):
        all_types = get_all_types()
        for cls in all_types.values():
            assert isinstance(cls, type)


# ══════════════════════════════════════════════════════════════════════
# T accessor
# ══════════════════════════════════════════════════════════════════════

class TestTypeAccessor:
    def test_attribute_access(self):
        assert T._TestCompA is _TestCompA

    def test_returns_none_for_unknown(self):
        assert T.NonExistentComp99999 is None

    def test_repr(self):
        r = repr(T)
        assert "<ComponentTypes:" in r
