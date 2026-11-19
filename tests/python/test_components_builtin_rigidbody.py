"""Tests for infernux.components.builtin.rigidbody — Rigidbody wrapper (real C++ backend)."""

from __future__ import annotations

import json

import pytest

from infernux.lib import ForceMode, PhysicsBackend, SceneManager, quatf
from tests.python.test_integration_physics import _collider_authoring_project, _physics_play
from infernux.components.builtin.rigidbody import (
    CollisionDetectionMode,
    Rigidbody,
    RigidbodyConstraints,
    RigidbodyInterpolation,
)


class TestVelocityAndForce:
    @pytest.mark.parametrize("backend", [PhysicsBackend.CPU, PhysicsBackend.GPU], ids=["cpu_scene", "gpu_scene"])
    def test_bound_wrapper_configures_moves_and_sleeps_a_dynamic_body(self, scene, backend, _physics_play):
        scene.physics_backend = backend
        owner = scene.create_game_object("Wrapper dynamics")
        owner.add_component("BoxCollider")
        rb = owner.add_component(Rigidbody)
        native = rb._cpp_component
        rb.mass = 2.0
        rb.drag = rb.angular_drag = 0.0
        rb.use_gravity = False
        assert native.mass == pytest.approx(2.0)
        assert native.drag == native.angular_drag == 0.0
        assert native.use_gravity is False

        runtime = _physics_play.start()
        with _physics_play.scope():
            manager = SceneManager.instance()
            rb = runtime.find("Wrapper dynamics").get_component(Rigidbody)
            native = rb._cpp_component
            assert runtime.physics_backend == backend
            _physics_play.frames(1, manager.get_fixed_time_step())
            rb.velocity = (1, 2, 3)
            rb.angular_velocity = (0, 0, 0)
            assert tuple(native.velocity) == pytest.approx((1, 2, 3))
            assert tuple(native.angular_velocity) == pytest.approx((0, 0, 0))
            rb.add_force((2, 0, 0))
            rb.add_torque((0, 0, 1), ForceMode.VelocityChange)
            rb.add_force_at_position((0, 2, 0), tuple(rb.world_center_of_mass), ForceMode.Impulse)
            _physics_play.frames(1, manager.get_fixed_time_step())
            assert tuple(native.velocity) == pytest.approx((1 + manager.get_fixed_time_step(), 3, 3), abs=1e-4)
            assert tuple(native.angular_velocity) == pytest.approx((0, 0, 1), abs=1e-4)
            assert tuple(rb.position) == pytest.approx(tuple(native.position))
            assert tuple(rb.rotation) == pytest.approx(tuple(native.rotation))

            rb.velocity = rb.angular_velocity = (0, 0, 0)
            rb.sleep()
            assert native.is_sleeping() and rb.is_sleeping()
            rb.wake_up()
            assert not native.is_sleeping() and not rb.is_sleeping()

    def test_kinematic_moves_reject_dynamic_rigidbody(self, cpp_rigidbody):
        rb = cpp_rigidbody
        with pytest.raises(RuntimeError, match="requires a kinematic Rigidbody"):
            rb.move_position((3, 2, 1))
        with pytest.raises(RuntimeError, match="requires a kinematic Rigidbody"):
            rb.move_rotation(quatf(0.0, 0.0, 0.0, 1.0))


class TestDetachedWrapperFailures:
    def test_native_state_access_requires_binding(self):
        rb = Rigidbody()
        operations = (
            lambda: rb.mass,
            lambda: setattr(rb, "mass", 2.0),
            lambda: rb.freeze_rotation,
            lambda: rb.is_sleeping(),
            lambda: rb.rotation,
            lambda: rb.add_force((1, 2, 3)),
            lambda: rb.add_torque((1, 2, 3)),
            lambda: rb.add_force_at_position((1, 2, 3), (0, 0, 0)),
            lambda: rb.move_position((1, 2, 3)),
            lambda: rb.move_rotation((0.0, 0.0, 0.0, 1.0)),
            lambda: rb.wake_up(),
            lambda: rb.sleep(),
        )
        for operation in operations:
            with pytest.raises(ReferenceError):
                operation()


@pytest.mark.parametrize("backend", [PhysicsBackend.CPU, PhysicsBackend.GPU], ids=["cpu_scene", "gpu_scene"])
def test_body_backend_is_owned_by_scene_not_component_authoring(scene, backend, _physics_play):
    from infernux.lib import Physics as NativePhysics
    from infernux.physics import Physics

    scene.physics_backend = backend
    owner = scene.create_game_object("Scene backend contract")
    owner.add_component("BoxCollider")
    body = owner.add_component(Rigidbody)
    native = body._require_cpp_component()
    assert scene.physics_backend == backend
    for name in ("gpu", "cross_backend_interaction"):
        assert not hasattr(type(body), name)
        assert not hasattr(native, name)
    assert not hasattr(Physics, "set_rigidbody_backend")
    assert not hasattr(NativePhysics, "set_rigidbody_backend")

    document = json.loads(native.serialize())
    assert "gpu" not in document
    assert "cross_backend_interaction" not in document
    assert native.deserialize(json.dumps(document))
    assert scene.physics_backend == backend
    assert json.loads(native.serialize()) == document

    runtime = _physics_play.start()
    with _physics_play.scope():
        runtime_body = runtime.find("Scene backend contract").get_component(Rigidbody)
        assert runtime_body is not body
        assert runtime.physics_backend == backend
        opposite = PhysicsBackend.GPU if backend == PhysicsBackend.CPU else PhysicsBackend.CPU
        with pytest.raises(RuntimeError, match="loaded runtime Scene"):
            runtime.physics_backend = opposite
        assert runtime.physics_backend == backend
        runtime_document = json.loads(runtime_body._require_cpp_component().serialize())
        assert runtime_document["component_id"] != document["component_id"]
        assert runtime_document == dict(document, component_id=runtime_body.component_id)
