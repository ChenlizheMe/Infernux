"""Sweeps return a target surface point, independently of shape travel distance."""
import math

import numpy as np
import pytest

from infernux.lib import PrimitiveType, Vector3, quatf
from infernux.lib import Physics as NativePhysics
from infernux.physics import Physics


@pytest.mark.parametrize("cast", ["ray", "sphere", "box", "capsule", "capsule_sphere"])
@pytest.mark.parametrize("target", ["box", "mesh"])
@pytest.mark.parametrize("angle", [0, .61])
@pytest.mark.parametrize("position", [(0, 0, 0), (30, -10, 25)])
def test_cast_point_belongs_to_target_surface(scene, cast, target, angle, position):
    if target == "mesh":
        owner = scene.create_primitive(PrimitiveType.Cube, "Mesh surface")
        assert owner.remove_component(owner.get_component("BoxCollider"))
        collider = owner.add_component("MeshCollider")
        owner.transform.local_scale = Vector3(10, 1, 10)
    else:
        owner = scene.create_game_object("Box surface")
        collider = owner.add_component("BoxCollider")
        collider.size = Vector3(10, 1, 10)
    rotation = quatf(0, 0, math.sin(angle / 2), math.cos(angle / 2))
    owner.transform.rotation = rotation
    owner.transform.position = Vector3(*position)
    NativePhysics.sync_transforms()
    if target == "mesh":
        assert not collider.is_cooking and collider.shape_error == ""
    normal = np.array([-math.sin(angle), math.cos(angle), 0])
    tangent = np.array([.3 * math.cos(angle), .3 * math.sin(angle), .2])
    origin = np.array(position) + normal * 3 + tangent
    # A non-unit direction verifies that distance remains world-space travel.
    direction = -normal * 7
    if cast == "ray":
        hit = Physics.raycast(origin, direction, 6)
        extent = 0
    elif cast == "sphere":
        hit = Physics.sphere_cast(origin, .25, direction, 6)
        extent = .25
    elif cast == "box":
        hit = Physics.box_cast(origin, (.25, .25, .25), direction, rotation, 6)
        extent = .25
    elif cast == "capsule":
        hit = Physics.capsule_cast(origin - normal * .25, origin + normal * .25, .25, direction, 6)
        extent = .5
    else:
        hit = Physics.capsule_cast(origin, origin, .25, direction, 6)
        extent = .25
    assert hit is not None
    assert hit.collider.component_id == collider.component_id
    point = np.array([hit.point.x, hit.point.y, hit.point.z])
    assert hit.distance == pytest.approx(2.5 - extent, abs=2e-4)
    np.testing.assert_allclose([hit.normal.x, hit.normal.y, hit.normal.z], normal, atol=2e-3)
    assert np.dot(point - position, normal) == pytest.approx(.5, abs=2e-4)
    assert np.linalg.norm(point - position) < 5


@pytest.mark.parametrize("cast", ["sphere", "box", "capsule"])
@pytest.mark.parametrize("offset", [0, 30])
def test_initial_overlap_reports_target_surface(scene, cast, offset):
    owner = scene.create_game_object("Overlapping surface")
    owner.transform.position = Vector3(offset, 0, 0)
    collider = owner.add_component("BoxCollider")
    collider.size = Vector3(10, 1, 10)
    origin = (offset + .3, .6, .2)
    if cast == "sphere":
        hit = Physics.sphere_cast(origin, .25, (0, -1, 0), 2)
    elif cast == "box":
        hit = Physics.box_cast(origin, (.25, .25, .25), (0, -1, 0), max_distance=2)
    else:
        hit = Physics.capsule_cast((offset + .3, .6, .2), (offset + .3, 1.1, .2), .25, (0, -1, 0), 2)
    assert hit is not None
    assert hit.collider.component_id == collider.component_id
    assert hit.distance == pytest.approx(0, abs=1e-5)
    assert hit.point.y == pytest.approx(.5, abs=2e-4)
    np.testing.assert_allclose([hit.normal.x, hit.normal.y, hit.normal.z], [0, 1, 0], atol=2e-3)


@pytest.mark.parametrize("query_triggers", [False, True])
@pytest.mark.parametrize("cast", ["sphere", "box", "capsule"])
def test_filtered_compound_hit_point_matches_selected_member(scene, query_triggers, cast):
    owner = scene.create_game_object("Compound surface")
    solid = owner.add_component("BoxCollider")
    solid.size = Vector3(10, 1, 10)
    trigger = owner.add_component("SphereCollider")
    trigger.center = Vector3(0, 2, 0)
    trigger.radius = .5
    trigger.is_trigger = True
    if cast == "sphere":
        hit = Physics.sphere_cast((0, 5, 0), .25, (0, -1, 0), 8, query_triggers=query_triggers)
    elif cast == "box":
        hit = Physics.box_cast((0, 5, 0), (.25, .25, .25), (0, -1, 0),
                               max_distance=8, query_triggers=query_triggers)
    else:
        hit = Physics.capsule_cast((0, 4.75, 0), (0, 5.25, 0), .25, (0, -1, 0), 8,
                                   query_triggers=query_triggers)
    assert hit is not None
    assert hit.collider.component_id == (trigger if query_triggers else solid).component_id
    assert hit.point.y == pytest.approx(2.5 if query_triggers else .5, abs=2e-4)
