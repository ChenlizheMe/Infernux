"""Public motion inputs and selected collider outlines match real physics."""
import math

import numpy as np
import pytest

from infernux.components.builtin import Rigidbody, SphereCollider
from infernux.gizmos import Gizmos
from infernux.lib import Physics, SceneManager, Vector3, quatf
from infernux.math.coerce import quat_rotate


def rotation_value(values, form):
    return {"tuple": tuple, "list": list, "native": lambda v: quatf(*v)}[form](values)


@pytest.mark.parametrize("form", ["tuple", "list", "native"])
@pytest.mark.parametrize("angle", [0, .7, math.pi / 2])
def test_move_rotation_applies_public_quaternion_inputs(scene, form, angle):
    owner = scene.create_game_object("Kinematic quaternion input")
    body = owner.add_component("Rigidbody")
    assert isinstance(body, Rigidbody)
    body.is_kinematic = True
    owner.add_component("BoxCollider")
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    dt = manager.get_fixed_time_step()
    manager.step(dt)
    target = (0, math.sin(angle / 2), 0, math.cos(angle / 2))
    body.move_rotation(rotation_value(target, form))
    manager.step(dt)
    assert abs(sum(a * b for a, b in zip(body.rotation, target))) == pytest.approx(1, abs=2e-5)
    # Settling the kinematic target must not leave residual angular movement.
    manager.step(dt)
    assert abs(sum(a * b for a, b in zip(body.rotation, target))) == pytest.approx(1, abs=2e-5)


@pytest.mark.parametrize("form", ["tuple", "list", "native"])
@pytest.mark.parametrize("values", [(0, 0, 0, 0), (math.nan, 0, 0, 1), (0, math.inf, 0, 1)])
def test_move_rotation_keeps_native_geometry_rejection(scene, form, values):
    owner = scene.create_game_object("Invalid quaternion")
    body = owner.add_component("Rigidbody")
    body.is_kinematic = True
    owner.add_component("BoxCollider")
    Physics.sync_transforms()
    with pytest.raises(ValueError, match="finite non-zero quaternion"):
        body.move_rotation(rotation_value(values, form))


@pytest.mark.parametrize("form", ["tuple", "list", "native"])
@pytest.mark.parametrize("missing_body", [False, True])
def test_move_rotation_keeps_native_body_preconditions(scene, form, missing_body):
    owner = scene.create_game_object("Unmovable rigidbody")
    body = owner.add_component("Rigidbody")
    body.is_kinematic = missing_body
    if not missing_body:
        owner.add_component("BoxCollider")
    Physics.sync_transforms()
    message = "enabled Collider body" if missing_body else "kinematic Rigidbody"
    with pytest.raises(RuntimeError, match=message):
        body.move_rotation(rotation_value((0, 0, 0, 1), form))


@pytest.mark.parametrize("local,parent", [
    ((1, 1, 1), (1, 1, 1)), ((2, 3, 4), (1, 1, 1)),
    ((1, 1, 1), (3, 3, 3)), ((1, 2, .5), (2, 3, 4)),
    ((-2, 3, 4), (1, 1, 1)), ((1, -2, .5), (-2, 3, 4)),
])
@pytest.mark.parametrize("center", [(0, 0, 0), (.4, -.3, .2)])
@pytest.mark.parametrize("rotated", [False, True])
def test_selected_sphere_outline_matches_scaled_physics(scene, monkeypatch, local, parent, center, rotated):
    ancestor = scene.create_game_object("Scaled ancestor")
    ancestor.transform.position = Vector3(10, 20, -30)
    ancestor.transform.local_scale = Vector3(*parent)
    owner = scene.create_game_object("Scaled sphere")
    owner.set_parent(ancestor, False)
    owner.transform.local_scale = Vector3(*local)
    if rotated:
        ancestor.transform.rotation = quatf(0, 0, math.sin(.2), math.cos(.2))
        owner.transform.rotation = quatf(0, math.sin(.35), 0, math.cos(.35))
    sphere = owner.add_component("SphereCollider")
    assert isinstance(sphere, SphereCollider)
    sphere.radius = .6
    sphere.center = Vector3(*center)
    Physics.sync_transforms()
    scale = tuple(owner.transform.lossy_scale)
    rotated_center = quat_rotate(owner.transform.rotation, tuple(a * b for a, b in zip(center, scale)))
    expected_center = np.array(tuple(owner.transform.position)) + rotated_center
    radius = sphere.radius * max(abs(value) for value in scale)
    # Compare with six real Jolt surface hits, not only a duplicated transform formula.
    for axis in np.eye(3):
        for sign in (-1, 1):
            direction = axis * sign
            hit = sphere.raycast(Vector3(*(expected_center + direction * (radius + 3))),
                                 Vector3(*(-direction)), 2 * radius + 6)
            assert hit is not None
            assert tuple(hit.point) == pytest.approx(expected_center + direction * radius, abs=2e-4)
    drawn = []
    old_color = Gizmos.color
    monkeypatch.setattr(Gizmos, "draw_wire_sphere", staticmethod(
        lambda point, size: drawn.append((tuple(point), size))))
    sphere.on_draw_gizmos_selected()
    assert len(drawn) == 1
    assert drawn[0][0] == pytest.approx(expected_center, abs=2e-4)
    assert drawn[0][1] == pytest.approx(radius + .02, abs=2e-4)
    assert Gizmos.color == old_color
