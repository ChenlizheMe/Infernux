"""Constraint retirement must leave the surviving joint usable immediately."""
import math

import pytest

from infernux.lib import PhysicsBackend, SceneManager, Vector3
from tests.python.test_integration_physics import _collider_authoring_project, _physics_play


@pytest.fixture(autouse=True)
def fixed_step(engine):
    manager = SceneManager.instance()
    previous = manager.get_fixed_time_step()
    manager.set_fixed_time_step(1 / 60)
    yield
    manager.set_fixed_time_step(previous)


def make_joint(scene, kind, support_scene=None, *, start=True):
    support_scene = support_scene or scene
    support = support_scene.create_game_object("Joint support")
    support.transform.position = Vector3(0, 3, 0)
    support_body = support.add_component("Rigidbody")
    support_body.is_kinematic = True
    support_body.use_gravity = False
    support_body.drag = 0
    support.add_component("BoxCollider")
    owner = scene.create_game_object("Joint mechanism")
    owner.transform.position = Vector3(2, 3, 0)
    body = owner.add_component("Rigidbody")
    body.use_gravity = False
    body.drag = body.angular_drag = 0
    body.interpolation = 0
    owner.add_component("BoxCollider")
    joint = owner.add_component(kind)
    joint.axis = Vector3(0, 0, 1) if kind == "HingeJoint" else Vector3(1, 0, 0)
    joint.connected_body = support_body
    if start:
        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        steps(3)
        assert displacement(body) == pytest.approx(0, abs=.015)
    return owner, body, joint, support, support_body


def steps(count, play=None):
    if play is not None:
        play.frames(count)
        return
    for _ in range(count):
        SceneManager.instance().step(1 / 60)


def displacement(body, play=None):
    body.velocity = Vector3(0, 6, 0)
    before = body.position.y
    steps(10, play)
    after = body.position.y
    body.velocity = Vector3(0, 0, 0)
    return after - before


def current(joint, kind):
    return getattr(joint, "current_angle" if kind == "HingeJoint" else "current_position")


@pytest.mark.parametrize("kind", ["HingeJoint", "SliderJoint"])
@pytest.mark.parametrize("operation", ["destroy_target", "unload_target", "remove_target_collider"])
@pytest.mark.parametrize("read_path", ["native", "public"])
@pytest.mark.parametrize("backend", [PhysicsBackend.CPU, PhysicsBackend.GPU], ids=["cpu_scene", "gpu_scene"])
def test_target_retirement_publishes_inactive_joint_before_next_fixed_step(scene, kind, operation, read_path, backend, _physics_play):
    scene.physics_backend = backend
    manager = SceneManager.instance()
    support_scene = manager.create_scene("Support world") if operation == "unload_target" else scene
    support_scene.physics_backend = backend
    make_joint(scene, kind, support_scene, start=False)
    support_scene_name = support_scene.name
    # Play snapshots remap references across resident Scenes. A standalone
    # authoring asset cannot encode another Scene's native component identity.
    scene = _physics_play.start(publish_asset=operation != "unload_target")
    with _physics_play.scope():
        support_scene = manager.get_scene(support_scene_name)
        assert scene.physics_backend == support_scene.physics_backend == backend
        owner = scene.find("Joint mechanism")
        body = owner.get_component("Rigidbody")
        joint = owner.get_component(kind)
        support = support_scene.find("Joint support")
        support_body = support.get_component("Rigidbody")
        steps(3, _physics_play)
        assert displacement(body, _physics_play) == pytest.approx(0, abs=.015)
        joint_handle = joint.handle
        support_handle = support.handle
        native_joint = joint._require_cpp_component()
        if operation == "destroy_target":
            support_scene.destroy_game_object(support)
            support_scene.process_pending_destroys()
            assert support_scene.resolve_game_object(support_handle) is None
        elif operation == "unload_target":
            world_id = support_scene.world_id
            manager.unload_scene(support_scene)
            assert manager.get_scene_by_world_id(world_id) is None
        else:
            assert support.remove_component(support.get_component("BoxCollider"))
            # An enabled Rigidbody retains its endpoint with an empty collision shape.
            assert joint.connected_body is support_body
            assert displacement(body, _physics_play) == pytest.approx(0, abs=.015)
            assert support.remove_component(support_body)
        assert scene.resolve_component(joint_handle) is not None
        assert joint.connected_body is None
        reader = native_joint if read_path == "native" else joint
        assert current(reader, kind) == 0
        steps(1, _physics_play)
        assert current(joint, kind) == 0
        assert scene.resolve_component(joint_handle) is not None
        assert displacement(body, _physics_play) == pytest.approx(1, abs=.015)
        # Reuse the exact public object after retirement. An explicit None means
        # fixed world, unlike a missing nonzero target reference.
        joint.connected_body = None
        steps(1, _physics_play)
        assert displacement(body, _physics_play) == pytest.approx(0, abs=.015)
        assert math.isfinite(current(joint, kind))


@pytest.mark.parametrize("kind", ["HingeJoint", "SliderJoint"])
@pytest.mark.parametrize("switch", ["joint", "body", "owner"])
def test_joint_repeated_disable_enable_rebuilds_constraint(scene, kind, switch, _physics_play):
    make_joint(scene, kind, start=False)
    runtime = _physics_play.start()
    with _physics_play.scope():
        owner = runtime.find("Joint mechanism")
        body = owner.get_component("Rigidbody")
        joint = owner.get_component(kind)
        support_body = runtime.find("Joint support").get_component("Rigidbody")
        steps(3, _physics_play)
        assert displacement(body, _physics_play) == pytest.approx(0, abs=.015)
        target = {"joint": joint, "body": support_body, "owner": owner}[switch]
        attr = "active" if switch == "owner" else "enabled"
        for _ in range(3):
            setattr(target, attr, False)
            steps(1, _physics_play)
            assert current(joint, kind) == 0
            if switch != "owner":
                assert displacement(body, _physics_play) == pytest.approx(1, abs=.015)
            setattr(target, attr, True)
            steps(1, _physics_play)
            assert displacement(body, _physics_play) == pytest.approx(0, abs=.015)
            assert math.isfinite(current(joint, kind))


@pytest.mark.parametrize("kind", ["HingeJoint", "SliderJoint"])
def test_removed_joint_stays_dead_while_replacement_binds(scene, kind):
    owner, body, joint, _, support_body = make_joint(scene, kind)
    handle = joint.handle
    assert owner.remove_component(joint)
    assert scene.resolve_component(handle) is None
    with pytest.raises(ReferenceError):
        current(joint, kind)
    assert displacement(body) == pytest.approx(1, abs=.015)
    replacement = owner.add_component(kind)
    replacement.axis = Vector3(0, 0, 1) if kind == "HingeJoint" else Vector3(1, 0, 0)
    replacement.connected_body = support_body
    steps(1)
    assert displacement(body) == pytest.approx(0, abs=.015)


@pytest.mark.parametrize("kind", ["HingeJoint", "SliderJoint"])
def test_owner_scene_unload_releases_constraint_on_other_scene_body(scene, kind):
    manager = SceneManager.instance()
    support_scene = manager.create_scene("Surviving support")
    _, _, _, _, support_body = make_joint(scene, kind, support_scene)
    world_id = scene.world_id
    manager.unload_scene(scene)
    assert manager.get_scene_by_world_id(world_id) is None
    support_body.is_kinematic = False
    steps(1)
    assert displacement(support_body) == pytest.approx(1, abs=.015)


@pytest.mark.parametrize("kind", ["HingeJoint", "SliderJoint"])
def test_constraint_owner_retirement_and_reconnection_stress(scene, kind):
    for cycle in range(32):
        owner, body, joint, support, support_body = make_joint(scene, kind)
        target_collider = support.get_component("BoxCollider")
        assert support.remove_component(target_collider)
        assert current(joint._require_cpp_component(), kind) == 0
        assert current(joint, kind) == 0
        assert joint.connected_body is support_body
        assert displacement(body) == pytest.approx(0, abs=.015)
        support_body.enabled = False
        assert current(joint, kind) == 0
        assert displacement(body) == pytest.approx(1, abs=.015)
        support_body.enabled = True
        support.add_component("BoxCollider")
        steps(1)
        assert displacement(body) == pytest.approx(0, abs=.015)
        # Both destruction orders must unregister the non-owning constraint
        # owner reference before either component allocation can be reused.
        first, second = (owner, support) if cycle % 2 else (support, owner)
        scene.destroy_game_object(first)
        scene.process_pending_destroys()
        if first is support:
            assert current(joint, kind) == 0
        scene.destroy_game_object(second)
        scene.process_pending_destroys()
