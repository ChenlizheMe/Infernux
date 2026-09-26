from __future__ import annotations

import pytest

import Infernux.lib as lib_module

from Infernux.lib import (
    GameObject,
    InvalidNativeObjectError,
    Vector3,
    _install_native_lifetime_guard,
    _is_native_lifetime_error,
    _unwrap_vec3,
)


class _FakeDeadGameObject:
    @property
    def id(self):
        raise RuntimeError("Access violation - no RTTI data!")

    @property
    def transform(self):
        raise RuntimeError("Access violation - no RTTI data!")

    def get_transform(self):
        raise RuntimeError("Access violation - no RTTI data!")

    def get_children(self):
        raise RuntimeError("Access violation - no RTTI data!")

    def set_parent(self, parent):
        raise RuntimeError("Access violation - no RTTI data!")


class _FakeDeadComponent:
    @property
    def component_id(self):
        raise RuntimeError("Access violation - no RTTI data!")

    @property
    def enabled(self):
        raise RuntimeError("Access violation - no RTTI data!")

    @enabled.setter
    def enabled(self, value):
        raise RuntimeError("Access violation - no RTTI data!")

    def serialize(self):
        raise RuntimeError("Access violation - no RTTI data!")


class _FakeDeadTransform(_FakeDeadComponent):
    @property
    def position(self):
        raise RuntimeError("Access violation - no RTTI data!")

    @position.setter
    def position(self, value):
        raise RuntimeError("Access violation - no RTTI data!")

    def local_to_world_matrix(self):
        raise RuntimeError("Access violation - no RTTI data!")


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


for _cls in (_FakeDeadGameObject, _FakeDeadComponent, _FakeDeadTransform):
    _install_native_lifetime_guard(_cls)


class TestNativeLifetimeErrorClassifier:
    def test_detects_access_violation(self):
        assert _is_native_lifetime_error(RuntimeError("Access violation - no RTTI data!")) is True

    def test_ignores_other_runtime_errors(self):
        assert _is_native_lifetime_error(RuntimeError("some other runtime problem")) is False


def test_value_record_guard_preserves_truth_and_protects_native_access():
    class Record:
        distance = 1.0

        @property
        def collider(self):
            raise RuntimeError("native object has been destroyed")

    _install_native_lifetime_guard(Record, check_liveness=False)
    record = Record()
    assert bool(record) and record.distance == 1.0
    with pytest.raises(InvalidNativeObjectError):
        _ = record.collider
    # Repeated installation must not reintroduce an entity liveness check.
    _install_native_lifetime_guard(Record)
    assert bool(record)


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


class TestGuardedGameObject:
    def test_invalid_id_raises(self):
        with pytest.raises(InvalidNativeObjectError):
            _FakeDeadGameObject().id

    def test_invalid_transform_raises(self):
        go = _FakeDeadGameObject()
        with pytest.raises(InvalidNativeObjectError):
            go.transform
        with pytest.raises(InvalidNativeObjectError):
            go.get_transform()

    def test_invalid_children_raises(self):
        with pytest.raises(InvalidNativeObjectError):
            _FakeDeadGameObject().get_children()

    def test_invalid_game_object_is_falsey(self):
        assert bool(_FakeDeadGameObject()) is False


class TestGuardedComponent:
    def test_invalid_component_id_raises(self):
        with pytest.raises(InvalidNativeObjectError):
            _FakeDeadComponent().component_id

    def test_invalid_enabled_raises(self):
        with pytest.raises(InvalidNativeObjectError):
            _FakeDeadComponent().enabled

    def test_invalid_serialize_raises(self):
        with pytest.raises(InvalidNativeObjectError):
            _FakeDeadComponent().serialize()

    def test_invalid_setattr_raises(self):
        comp = _FakeDeadComponent()
        with pytest.raises(InvalidNativeObjectError):
            comp.enabled = True


class TestGuardedTransform:
    def test_invalid_position_raises(self):
        with pytest.raises(InvalidNativeObjectError):
            _FakeDeadTransform().position

    def test_invalid_matrix_raises(self):
        with pytest.raises(InvalidNativeObjectError):
            _FakeDeadTransform().local_to_world_matrix()

    def test_invalid_transform_is_falsey(self):
        assert bool(_FakeDeadTransform()) is False


def test_native_guard_binds_one_shared_function_and_preserves_signature():
    import inspect

    method = _FakeDeadGameObject().set_parent
    assert method.__func__ is _FakeDeadGameObject.set_parent
    assert str(inspect.signature(method)) == "(parent)"
    assert method.__name__ == "set_parent"
    assert _FakeDeadGameObject().set_parent.__func__ is method.__func__
    with pytest.raises(InvalidNativeObjectError):
        _FakeDeadGameObject.set_parent(_FakeDeadGameObject(), None)


def test_guard_does_not_wrap_python_component_methods_on_lookup():
    class UserComponent(_FakeDeadComponent):
        def update(self):
            return self.serialize()

    component = UserComponent()
    assert UserComponent.__getattribute__ is object.__getattribute__
    assert UserComponent.__setattr__ is object.__setattr__
    assert component.update.__func__ is UserComponent.update
    with pytest.raises(InvalidNativeObjectError):
        component.update()
    # Hot replacement is normal Python binding; there is no stale callable cache.
    replacement = lambda self: 42
    UserComponent.update = replacement
    assert component.update.__func__ is replacement
    assert component.update() == 42


def test_guard_wraps_property_accessors_once_without_changing_property_contract():
    descriptor = vars(_FakeDeadTransform)["position"]
    assert isinstance(descriptor, property)
    assert descriptor.fget._infernux_native_guarded
    assert descriptor.fset._infernux_native_guarded
    with pytest.raises(InvalidNativeObjectError):
        _FakeDeadTransform().position = Vector3(1, 2, 3)
    with pytest.raises(AttributeError):
        _FakeDeadGameObject().id = 2  # Read-only native properties remain read-only.


def test_guard_preserves_static_class_methods_and_ordinary_errors():
    class NativeFixture:
        @staticmethod
        def static(value):
            return value

        @classmethod
        def kind(cls):
            return cls

        def fail(self):
            raise RuntimeError("ordinary failure")

    _install_native_lifetime_guard(NativeFixture)
    first_method = NativeFixture.fail
    _install_native_lifetime_guard(NativeFixture)
    assert NativeFixture.fail is first_method
    instance = NativeFixture()
    assert instance.static(7) == NativeFixture.static(7) == 7
    assert instance.kind() is NativeFixture
    with pytest.raises(RuntimeError, match="^ordinary failure$") as error:
        instance.fail()
    assert type(error.value) is RuntimeError


def test_guard_does_not_retain_native_owner():
    import gc
    import weakref

    instance = _FakeDeadGameObject()
    reference = weakref.ref(instance)
    instance.get_children
    del instance
    gc.collect()
    assert reference() is None


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
            "Infernux.engine.prefab_manager.instantiate_prefab",
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
            "Infernux.engine.prefab_manager.instantiate_prefab",
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
