"""Tests for InfEngine.components.builtin_component — CppProperty, BuiltinComponent (real C++ backend)."""

from __future__ import annotations

from enum import IntEnum

import pytest

from InfEngine.components.builtin_component import BuiltinComponent, CppProperty
from InfEngine.components.serialized_field import FieldType, get_serialized_fields
import InfEngine.lib as lib


# ── Test helpers ──

class DemoEnum(IntEnum):
    A = 1
    B = 2


class DemoCpp:
    """Minimal stand-in for a C++ component (needed for CppProperty __set_name__)."""
    def __init__(self):
        self.mode = 2
        self.raw = 11
        self.locked = 5
        self.intensity = 1.0


class DemoBuiltin(BuiltinComponent):
    _cpp_type_name = "DemoBuiltin"

    mode = CppProperty("mode", FieldType.ENUM, default=DemoEnum.A, enum_type=DemoEnum)
    raw = CppProperty("raw", FieldType.INT, default=7)
    locked = CppProperty("locked", FieldType.INT, default=3, readonly=True)
    intensity = CppProperty("intensity", FieldType.FLOAT, default=1.0)


class LazyEnumBuiltin(BuiltinComponent):
    _cpp_type_name = "LazyEnumBuiltin"

    mode = CppProperty("mode", FieldType.ENUM, default=DemoEnum.A, enum_type="DemoEnum")


# ══════════════════════════════════════════════════════════════════════
# CppProperty descriptor
# ══════════════════════════════════════════════════════════════════════

class TestCppPropertyDefaults:
    def test_returns_default_without_cpp(self):
        demo = DemoBuiltin()
        assert demo.mode is DemoEnum.A
        assert demo.raw == 7
        assert demo.intensity == 1.0

    def test_set_name_assigns_metadata_name(self):
        desc = DemoBuiltin.__dict__["raw"]
        assert desc.metadata.name == "raw"


class TestCppPropertyReadWrite:
    def test_reads_from_cpp_and_casts_enum(self):
        demo = DemoBuiltin()
        demo._cpp_component = DemoCpp()
        assert demo.mode is DemoEnum.B   # cpp.mode = 2, DemoEnum(2) == B
        assert demo.raw == 11

    def test_writes_to_cpp(self):
        cpp = DemoCpp()
        demo = DemoBuiltin()
        demo._cpp_component = cpp

        demo.mode = DemoEnum.A
        demo.raw = 42

        assert cpp.mode == 1
        assert cpp.raw == 42

    def test_readonly_rejects_set(self):
        demo = DemoBuiltin()
        demo._cpp_component = DemoCpp()
        with pytest.raises(AttributeError):
            demo.locked = 10

    def test_lazy_enum_type_resolved_from_lib(self):
        """Test lazy enum resolution using the real InfEngine.lib."""
        lib.DemoEnum = DemoEnum
        try:
            demo = LazyEnumBuiltin()
            demo._cpp_component = DemoCpp()
            assert demo.mode is DemoEnum.B
        finally:
            del lib.DemoEnum


class TestCppPropertyEdgeCases:
    def test_class_access_returns_descriptor(self):
        desc = DemoBuiltin.mode
        assert isinstance(desc, CppProperty)

    def test_runtime_error_falls_to_default(self):
        class BadCpp:
            @property
            def raw(self):
                raise RuntimeError("dead")
        demo = DemoBuiltin()
        demo._cpp_component = BadCpp()
        assert demo.raw == 7


# ══════════════════════════════════════════════════════════════════════
# BuiltinComponent subclass
# ══════════════════════════════════════════════════════════════════════

class TestBuiltinComponent:
    def test_registered_in_builtin_registry(self):
        assert "DemoBuiltin" in BuiltinComponent._builtin_registry
        assert BuiltinComponent._builtin_registry["DemoBuiltin"] is DemoBuiltin

    def test_isinstance_inf_component(self):
        from InfEngine.components.component import InfComponent
        demo = DemoBuiltin()
        assert isinstance(demo, InfComponent)

    def test_serialized_fields_contain_cpp_properties(self):
        fields = get_serialized_fields(DemoBuiltin)
        assert "mode" in fields
        assert "raw" in fields
        assert "locked" in fields

    def test_repr_unbound(self):
        demo = DemoBuiltin()
        r = repr(demo)
        assert "DemoBuiltin" in r
        assert "bound=False" in r

    def test_repr_bound(self):
        cpp = DemoCpp()
        cpp.component_id = 42
        cpp.enabled = True
        demo = DemoBuiltin()
        demo._bind_cpp(cpp, type("FakeGO", (), {"id": 1})())
        r = repr(demo)
        assert "bound=True" in r

    def test_clear_cache(self):
        BuiltinComponent._wrapper_cache.clear()
        BuiltinComponent._clear_cache()
        assert len(BuiltinComponent._wrapper_cache) == 0
