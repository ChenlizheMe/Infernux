from __future__ import annotations


import infernux.lib as native_lib
from infernux.core.material import Material


class _NativeMaterial:
    def __init__(self, *, deleted: bool = False, save_result: bool = True):
        self.guid = "test-material"
        self.file_path = "Assets/Test.mat"
        self.name = "Test"
        self.deleted = deleted
        self.save_result = save_result
        self.save_count = 0

    def is_deleted(self) -> bool:
        return self.deleted

    def save(self) -> bool:
        self.save_count += 1
        return self.save_result

    def serialize_document(self) -> dict:
        return {
            "name": self.name,
            "shaders": {"vertex": {}, "fragment": {}},
            "renderState": {"renderQueue": 2000},
            "properties": {},
        }


class _AssetRegistry:
    current = None

    @classmethod
    def instance(cls):
        return cls.current


class _MaterialRegistry:
    def __init__(self, native=None, error: Exception | None = None):
        self.native = native
        self.error = error

    def load_material(self, _path: str):
        if self.error is not None:
            raise self.error
        return self.native

    def get_builtin_material(self, _name: str):
        if self.error is not None:
            raise self.error
        return self.native


def test_material_template_roundtrip_uses_canonical_native_color_and_document_semantics():
    from infernux.engine.ui.project_file_ops import _new_material_document

    native = native_lib.InxMaterial("Authored", "Unlit")
    native.set_color("baseColor", (1.0, 0.55, 0.12, 1.0))
    expected = native.serialize_document()["properties"]["baseColor"]["type"]
    document = _new_material_document("Created")
    assert "path_hint" not in document["shaders"]["vertex"] and "path_hint" not in document["shaders"]["fragment"]
    assert document["properties"]["baseColor"]["type"] == expected
    assert native.deserialize_document(document)
    material = Material(native)
    assert material.to_dict() == material.serialize_document() == native.serialize_document()
    assert {"shaders", "renderState", "properties"} <= material.to_dict().keys()


def test_pending_material_owners_keep_independent_save_failure_and_retirement_state():
    first_native, second_native = _NativeMaterial(), _NativeMaterial()
    first, second = Material(first_native), Material(second_native)
    try:
        first._auto_save()
        second._auto_save()
        assert not Material._pending_saves
        first._last_save_time = second._last_save_time = float("inf")
        first._auto_save()
        second._auto_save()
        assert set(Material._pending_saves) == {id(first), id(second)}
        first_native.save_result = False
        first._flush_save()
        assert first_native.save_count == 2 and first._save_pending
        assert Material._pending_saves[id(first)] is first
        first_native.deleted = True
        first._flush_save()
        assert first_native.save_count == 2 and not first._save_pending
        assert set(Material._pending_saves) == {id(second)}
        second.dispose()
        assert not second._save_pending and not Material._pending_saves
    finally:
        Material._pending_saves.clear()
