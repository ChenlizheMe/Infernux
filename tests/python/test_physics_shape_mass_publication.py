"""Shape edits preserve authored mass and publish matching inertia/DOF state."""
import numpy as np
import pytest

from infernux import Instantiate
from infernux.components.builtin.rigidbody import RigidbodyConstraints as Constraints
from infernux.lib import ForceMode, Physics, PrimitiveType, Vector3
from infernux.physics import Physics as PublicPhysics


MASKS = [0, int(Constraints.FreezePositionX | Constraints.FreezeRotationZ), int(Constraints.FreezePosition)]


@pytest.mark.parametrize("constraints", MASKS)
def test_mass_edit_preserves_locked_axes(scene, constraints):
    owner = scene.create_game_object("Mass authoring")
    owner.add_component("BoxCollider")
    body = owner.add_component("Rigidbody")
    body.constraints = constraints
    Physics.sync_transforms()
    body.mass = 4
    state = PublicPhysics.get_rigidbody_states([body])
    np.testing.assert_allclose(state["inverse_mass"][0],
                               [0 if constraints & bit else .25 for bit in (2, 4, 8)])
    body.add_force(Vector3(1, 1, 1), ForceMode.Impulse)
    response = PublicPhysics.get_rigidbody_states([body])
    np.testing.assert_allclose(response["linear_velocity"], state["inverse_mass"], atol=1e-6)


@pytest.mark.parametrize("constraints", MASKS)
@pytest.mark.parametrize("motion", ["dynamic", "kinematic", "static"])
def test_thin_shape_edit_survives_motion_transition(scene, constraints, motion):
    owner = scene.create_game_object("Thin authored body")
    collider = owner.add_component("BoxCollider")
    body = owner.add_component("Rigidbody")
    body.mass = 5
    body.constraints = constraints
    body.use_gravity = False
    body.is_kinematic = motion == "kinematic"
    body.enabled = motion != "static"
    Physics.sync_transforms()
    collider.size = Vector3(.001, .001, 5)
    Physics.sync_transforms()
    body.enabled = True
    body.is_kinematic = False
    Physics.sync_transforms()
    state = PublicPhysics.get_rigidbody_states([body])
    reference = Instantiate(owner)
    reference.transform.position = Vector3(100, 0, 0)
    Physics.sync_transforms()
    cold = PublicPhysics.get_rigidbody_states([reference.get_component("Rigidbody")])
    np.testing.assert_allclose(state["inverse_mass"][0],
                               [0 if constraints & bit else .2 for bit in (2, 4, 8)], atol=1e-6)
    np.testing.assert_allclose(state["inverse_inertia"], cold["inverse_inertia"], atol=1e-6)
    body.add_torque(Vector3(.1, .2, .3), ForceMode.Impulse)
    response = PublicPhysics.get_rigidbody_states([body])
    np.testing.assert_allclose(response["angular_velocity"][0],
                               state["inverse_inertia"][0] @ np.array([.1, .2, .3]), atol=1e-6)


@pytest.mark.parametrize("shape", ["BoxCollider", "SphereCollider", "CapsuleCollider", "CylinderCollider", "MeshCollider"])
@pytest.mark.parametrize("operation", ["resize", "center", "scale", "add_member", "remove_member", "disable_member"])
@pytest.mark.parametrize("constraints", MASKS)
def test_shape_edit_preserves_authored_dynamics(scene, shape, operation, constraints):
    if shape == "MeshCollider":
        owner = scene.create_primitive(PrimitiveType.Cube, "Edited mesh body")
        assert owner.remove_component(owner.get_component("BoxCollider"))
    else:
        owner = scene.create_game_object("Edited body")
    body = owner.add_component("Rigidbody")
    body.mass = 2
    body.constraints = constraints
    body.use_gravity = False
    collider = owner.add_component(shape)
    if shape == "MeshCollider":
        collider.convex = True
    if operation in ("remove_member", "disable_member"):
        sibling = owner.add_component("SphereCollider")
        sibling.center = Vector3(2, 0, 0)
    Physics.sync_transforms()
    expected_mass = np.array([0 if constraints & bit else .5 for bit in (2, 4, 8)])
    before = PublicPhysics.get_rigidbody_states([body])
    np.testing.assert_allclose(before["inverse_mass"][0], expected_mass)
    body.velocity = Vector3(.3, .4, .5)
    body.angular_velocity = Vector3(.1, .2, .3)
    before = PublicPhysics.get_rigidbody_states([body])
    if operation == "resize":
        if shape == "MeshCollider":
            owner.get_component("MeshRenderer").set_primitive_mesh(PrimitiveType.Sphere)
        elif shape == "BoxCollider":
            collider.size = Vector3(2, 3, 4)
        elif shape == "CapsuleCollider":
            collider.height *= 2
        else:
            collider.radius *= 2
    elif operation == "center":
        collider.center = Vector3(.25, .5, 0)
    elif operation == "scale":
        owner.transform.local_scale = Vector3(2, 2, 2)
    elif operation == "add_member":
        owner.add_component("SphereCollider").center = Vector3(2, 0, 0)
    elif operation == "remove_member":
        assert owner.remove_component(sibling)
    else:
        sibling.enabled = False
    Physics.sync_transforms()
    if shape == "MeshCollider":
        assert not collider.is_cooking and collider.shape_error == ""
    after = PublicPhysics.get_rigidbody_states([body])
    assert body.mass == 2
    assert body.constraints == constraints
    np.testing.assert_allclose(after["inverse_mass"][0], expected_mass, atol=1e-6)
    np.testing.assert_allclose(after["linear_velocity"], before["linear_velocity"], atol=1e-6)
    np.testing.assert_allclose(after["angular_velocity"], before["angular_velocity"], atol=1e-6)
    # A newly built body from the same final authoring state provides the cold
    # construction oracle, including compound COM and geometric inertia.
    reference = Instantiate(owner)
    reference.transform.position = Vector3(100, 0, 0)
    Physics.sync_transforms()
    reference_state = PublicPhysics.get_rigidbody_states([reference.get_component("Rigidbody")])
    np.testing.assert_allclose(after["inverse_inertia"], reference_state["inverse_inertia"], rtol=1e-5, atol=1e-6)
    body.add_force(Vector3(1, 1, 1), ForceMode.Impulse)
    body.add_torque(Vector3(.2, -.3, .4), ForceMode.Impulse)
    response = PublicPhysics.get_rigidbody_states([body])
    np.testing.assert_allclose(response["linear_velocity"][0] - after["linear_velocity"][0], expected_mass, atol=1e-6)
    np.testing.assert_allclose(response["angular_velocity"][0] - after["angular_velocity"][0],
                               after["inverse_inertia"][0] @ np.array([.2, -.3, .4]), atol=1e-6)
