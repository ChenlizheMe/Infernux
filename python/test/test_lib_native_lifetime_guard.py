from __future__ import annotations

from InfEngine.lib import (
    Vector3,
    _install_native_lifetime_guard,
    _is_native_lifetime_error,
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


for _cls in (_FakeDeadGameObject, _FakeDeadComponent, _FakeDeadTransform):
    _install_native_lifetime_guard(_cls)


class TestNativeLifetimeErrorClassifier:
    def test_detects_access_violation(self):
        assert _is_native_lifetime_error(RuntimeError("Access violation - no RTTI data!")) is True

    def test_ignores_other_runtime_errors(self):
        assert _is_native_lifetime_error(RuntimeError("some other runtime problem")) is False


class TestGuardedGameObject:
    def test_invalid_id_becomes_zero(self):
        assert _FakeDeadGameObject().id == 0

    def test_invalid_transform_becomes_none(self):
        go = _FakeDeadGameObject()
        assert go.transform is None
        assert go.get_transform() is None

    def test_invalid_children_becomes_empty_list(self):
        assert _FakeDeadGameObject().get_children() == []

    def test_invalid_game_object_is_falsey(self):
        assert bool(_FakeDeadGameObject()) is False


class TestGuardedComponent:
    def test_invalid_component_id_becomes_zero(self):
        assert _FakeDeadComponent().component_id == 0

    def test_invalid_enabled_becomes_false(self):
        assert _FakeDeadComponent().enabled is False

    def test_invalid_serialize_becomes_empty_json(self):
        assert _FakeDeadComponent().serialize() == "{}"

    def test_invalid_setattr_is_noop(self):
        comp = _FakeDeadComponent()
        comp.enabled = True


class TestGuardedTransform:
    def test_invalid_position_becomes_zero_vector(self):
        pos = _FakeDeadTransform().position
        assert isinstance(pos, Vector3)
        assert (pos.x, pos.y, pos.z) == (0.0, 0.0, 0.0)

    def test_invalid_matrix_becomes_identity(self):
        assert _FakeDeadTransform().local_to_world_matrix() == [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]

    def test_invalid_transform_is_falsey(self):
        assert bool(_FakeDeadTransform()) is False