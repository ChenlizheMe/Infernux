from __future__ import annotations

import pytest

import infernux.lib as lib_module

from infernux.lib import (
    GameObject,
    InvalidNativeObjectError,
    Vector3,
    _unwrap_vec3,
)


class _FakeQuat:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


class _FakeLiveTransform:
    def __init__(self):
        self.position = None
        self.rotation = None
        self.local_position = Vector3(1.0, 2.0, 3.0)
        self.local_rotation = _FakeQuat(0.0, 0.0, 0.0, 1.0)
        self.local_scale = Vector3(1.0, 1.0, 1.0)


class _FakeClone:
    def __init__(self):
        self.transform = _FakeLiveTransform()
        self.parent_calls = []

    def set_parent(self, parent, world_position_stays=True):
        self.parent_calls.append((parent, world_position_stays))


def test_query_and_contact_records_have_no_entity_bool_override():
    for cls in (lib_module.RaycastHit, lib_module.CollisionInfo):
        assert '__bool__' not in vars(cls)


class TestTransformVectorCoercion:
    def test_plain_sequence_becomes_vector3(self):
        value = _unwrap_vec3((1, 2.5, -3))
        assert isinstance(value, Vector3)
        assert (value.x, value.y, value.z) == pytest.approx((1.0, 2.5, -3.0))

    def test_invalid_sequence_is_left_for_native_diagnostic(self):
        value = _unwrap_vec3((1, 2))
        assert value == (1, 2)


def test_native_guard_does_not_retain_python_wrapper(scene):
    import gc
    import weakref

    owner = scene.create_game_object("BorrowedPythonWrapper")
    identity = owner.handle
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is None
    assert scene.resolve_game_object(identity).name == "BorrowedPythonWrapper"


def test_ordinary_native_error_text_is_not_reclassified():
    def fail(_owner):
        raise RuntimeError("Access violation is just this test's message")

    with pytest.raises(RuntimeError) as error:
        lib_module._call_native_game_object("fail", fail, None)
    assert type(error.value) is RuntimeError


def test_native_guard_preserves_readonly_properties(scene):
    owner = scene.create_game_object("ReadonlyIdentity")
    with pytest.raises(AttributeError):
        owner.id = 2


@pytest.mark.parametrize("wrong", [None, 42, "text", object()])
def test_native_liveness_rejects_non_native_self(wrong):
    with pytest.raises(TypeError):
        GameObject.__bool__(wrong)


def test_native_guard_preserves_python_subclass_methods():
    class UserCollider(lib_module._native_module.BoxCollider):
        def update(self):
            return self.serialize()

    instance = UserCollider()
    assert instance and instance.update()
    assert instance.update.__func__ is UserCollider.update
    replacement = lambda self: 42
    UserCollider.update = replacement
    assert instance.update.__func__ is replacement
    assert instance.update() == 42


class TestInstantiateOverloads:
    def test_instantiate_source_resolution_errors_are_not_suppressed(self):
        class BrokenReference:
            def resolve(self):
                raise RuntimeError("reference resolution failed")

        with pytest.raises(RuntimeError, match="reference resolution failed"):
            lib_module._resolve_game_object_instantiate_source(BrokenReference())

    def test_game_object_instantiate_accepts_prefab_ref_source(self, monkeypatch):
        clone = _FakeClone()
        prefab_ref = object()

        monkeypatch.setattr(lib_module, "_resolve_game_object_instantiate_source", lambda original: ("prefab", original))
        def instantiate_prefab(original, parent, world_space, configure_created):
            assert original is prefab_ref
            assert parent is None
            assert world_space is True
            configure_created(clone)
            return clone

        monkeypatch.setattr(lib_module, "_instantiate_prefab_reference", instantiate_prefab)

        assert GameObject.instantiate(prefab_ref) is clone

    def test_prefab_reference_without_guid_ignores_legacy_path_hint(self, monkeypatch):
        class LegacyPathOnlyReference:
            guid = ""

            @property
            def path_hint(self):
                raise AssertionError("legacy prefab path_hint must not be read")

        monkeypatch.setattr(
            "infernux.engine.prefab_manager.instantiate_prefab",
            lambda **_kwargs: pytest.fail("path-only PrefabRef must not be instantiated"),
        )

        assert lib_module._instantiate_prefab_reference(LegacyPathOnlyReference()) is None

    def test_prefab_reference_does_not_fallback_to_path_after_guid_failure(self, monkeypatch):
        class GuidReference:
            guid = " prefab-guid "

            @property
            def path_hint(self):
                raise AssertionError("prefab path_hint must not be read")

        database = object()
        registry = type("Registry", (), {"get_asset_database": lambda self: database})()
        monkeypatch.setattr(lib_module.AssetRegistry, "instance", staticmethod(lambda: registry))
        calls = []

        def instantiate_prefab(**kwargs):
            calls.append(kwargs)
            return None

        monkeypatch.setattr(
            "infernux.engine.prefab_manager.instantiate_prefab",
            instantiate_prefab,
        )

        assert lib_module._instantiate_prefab_reference(GuidReference()) is None
        assert len(calls) == 1
        assert calls[0]["guid"] == "prefab-guid"
        assert calls[0]["asset_database"] is database
        assert "file_path" not in calls[0]

    def test_game_object_instantiate_applies_position_rotation_and_parent(self, monkeypatch):
        clone = _FakeClone()
        source = object()
        parent = object()
        position = Vector3(9.0, 8.0, 7.0)
        rotation = _FakeQuat(0.0, 0.0, 0.0, 1.0)

        monkeypatch.setattr(lib_module, "_resolve_game_object_instantiate_source", lambda original: ("game_object", original))
        monkeypatch.setattr(lib_module, "_coerce_parent_game_object", lambda original: original)
        calls = []

        def instantiate_native(original, target_parent, world_space, configure_created):
            calls.append((original, target_parent, world_space))
            configure_created(clone)
            return clone

        monkeypatch.setattr(lib_module, "_native_game_object_instantiate", instantiate_native)

        result = GameObject.instantiate(source, position, rotation, parent)

        assert result is clone
        assert calls == [(source, parent, True)]
        assert clone.parent_calls == []
        assert clone.transform.position is position
        assert clone.transform.rotation is rotation
