"""Public Instantiate must publish native lifecycle receivers without a script."""
import pytest

from infernux import Instantiate, InxComponent, serialized_field
from infernux.lib import Physics, SceneManager, Vector3


class CloneStepProbe(InxComponent):
    ticks: int = serialized_field(default=0)

    def fixed_update(self, delta_time):
        self.ticks += 1


def anchor(body):
    # Rotate the nonzero local joint anchor by the actual solver orientation.
    q, p = body.rotation, body.position
    return (p.x + .25 - .5 * (q.y*q.y + q.z*q.z),
            p.y + .5 * (q.w*q.z + q.x*q.y),
            p.z + .5 * (q.x*q.z - q.w*q.y))


@pytest.mark.parametrize("kind", ["HingeJoint", "SliderJoint"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("cloned", [False, True])
@pytest.mark.parametrize("with_script", [False, True])
def test_native_clone_receives_fixed_steps_and_rebuilds_body(scene, kind, nested, cloned, with_script):
    manager = SceneManager.instance()
    dt = manager.get_fixed_time_step()

    def steps(count):
        for _ in range(count):
            manager.step(dt)

    def actor(name, x):
        owner = scene.create_game_object(name)
        owner.transform.position = Vector3(x, 3, 0)
        body = owner.add_component("Rigidbody")
        body.use_gravity = False
        body.drag = body.angular_drag = 0
        body.interpolation = 0
        owner.add_component("BoxCollider")
        return owner, body

    prototype, prototype_body = actor("Prototype", 0)
    root = scene.create_game_object("Mechanism root") if nested else prototype
    if nested:
        prototype.set_parent(root, True)
    joint = prototype.add_component(kind)
    joint.anchor = Vector3(.25, 0, 0)
    joint.axis = Vector3(0, 0, 1) if kind == "HingeJoint" else Vector3(1, 0, 0)
    joint.use_limits = False
    if with_script:
        prototype.add_component(CloneStepProbe)
    _, free_body = actor("Unconstrained control", 8)
    manager.play()
    manager.pause()
    Physics.sync_transforms()
    steps(3)
    if cloned:
        clone = Instantiate(root)
        clone.transform.position = Vector3(4, 0 if nested else 3, 0)
        target = clone.get_child(0) if nested else clone
        body = target.get_component("Rigidbody")
        assert body.component_id != prototype_body.component_id
        assert target.get_component(kind).component_id != joint.component_id
    else:
        target, body = prototype, prototype_body
    if with_script:
        assert target.get_component(CloneStepProbe) is not None
    else:
        assert list(target.get_py_components() or []) == []
    Physics.sync_transforms()
    steps(3)
    # A later body replacement still needs native fixed dispatch to rebuild
    # the constraint, without setting a joint property or adding a script.
    for rebuild in (False, True):
        if rebuild:
            assert target.remove_component(target.get_component("BoxCollider"))
            target.add_component("BoxCollider")
            steps(3)
        before, free_y = anchor(body), free_body.position.y
        ticks_before = target.get_component(CloneStepProbe).ticks if with_script else 0
        time_before = manager.fixed_time
        body.velocity = free_body.velocity = Vector3(0, 6, 0)
        steps(10)
        assert manager.fixed_time - time_before == pytest.approx(10*dt, abs=1e-5)
        assert free_body.position.y - free_y == pytest.approx(60*dt, abs=.015)
        assert anchor(body) == pytest.approx(before, abs=.025)
        if with_script:
            assert target.get_component(CloneStepProbe).ticks == ticks_before + 10
        body.velocity = body.angular_velocity = free_body.velocity = Vector3(0, 0, 0)
