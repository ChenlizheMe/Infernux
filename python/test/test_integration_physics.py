"""Integration tests — Physics simulation with real Jolt backend (real engine)."""
from __future__ import annotations

import json
import math

import pytest

from Infernux.components import InxComponent
from Infernux.engine.scene_document_transaction import SceneDocumentTransaction
from Infernux.physics import Physics as PublicPhysics
from Infernux.timing import Time
from Infernux.lib import BoxCollider as NativeBoxCollider
from Infernux.lib import MeshCollider as NativeMeshCollider
from Infernux.lib import (
    ForceMode,
    CollisionDetectionMode,
    EngineConfig,
    GameObject,
    InxPhysicMaterial,
    Physics,
    PrimitiveType,
    RigidbodyConstraints,
    SceneManager,
    Vector3,
    quatf,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _step_frames(n: int = 60, dt: float = 1.0 / 60.0):
    """Advance the physics simulation by *n* frames."""
    sm = SceneManager.instance()
    for _ in range(n):
        sm.step(dt)


def _assign_physic_material(collider, *, friction=0.4, bounciness=0.0,
                            friction_combine=0, bounce_combine=0):
    material = InxPhysicMaterial()
    material.friction = friction
    material.bounciness = bounciness
    material.friction_combine = friction_combine
    material.bounce_combine = bounce_combine
    collider.physic_material = material
    return material


def _make_ground(scene):
    """Create a large static ground plane (BoxCollider, no Rigidbody)."""
    ground = scene.create_game_object("Ground")
    ground.transform.position = Vector3(0, 0, 0)
    ground.transform.local_scale = Vector3(100, 1, 100)
    ground.add_component("BoxCollider")
    return ground


def _make_ball(scene, *, pos=None, mass=1.0, radius=0.5):
    """Create a dynamic sphere with Rigidbody + SphereCollider."""
    ball = scene.create_game_object("Ball")
    ball.transform.position = pos or Vector3(0, 10, 0)
    rb = ball.add_component("Rigidbody")
    rb.mass = mass
    col = ball.add_component("SphereCollider")
    col.radius = radius
    return ball, rb


def _rotate_vector(rotation, vector):
    """Rotate a Vector3 by the engine's xyzw quaternion value."""
    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    vx, vy, vz = vector.x, vector.y, vector.z
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return Vector3(
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def test_additive_scene_rigidbody_collides_with_other_scene_ground(scene):
    """All loaded Scenes contribute to the one authoritative Jolt World."""
    manager = SceneManager.instance()
    _make_ground(scene)
    additive = manager.create_scene("additive_physics")
    try:
        ball, rigidbody = _make_ball(
            additive, pos=Vector3(0.0, 4.0, 0.0), radius=0.5
        )
        # Keep A active: activation chooses default authoring/camera policy and
        # must not decide whether B participates in simulation or queries.
        manager.set_active_scene(scene)
        manager.play()
        manager.pause()
        _step_frames(180)

        assert rigidbody.position.y == pytest.approx(1.0, abs=0.08)
        hit = Physics.raycast(
            Vector3(0.0, 3.0, 0.0), Vector3(0.0, -1.0, 0.0), 10.0
        )
        assert hit is not None
        assert hit.game_object.id == ball.id
    finally:
        if manager.is_playing():
            manager.stop()
        manager.unload_scene(additive)
        manager.set_active_scene(scene)


class TestHingeJoint:
    def test_world_hinge_preserves_anchor_and_limits_rotation(self, scene):
        body = scene.create_game_object("WorldHinge")
        body.transform.position = Vector3(1.0, 3.0, 0.0)
        rigidbody = body.add_component("Rigidbody")
        rigidbody.angular_drag = 0.05
        collider = body.add_component("BoxCollider")
        collider.size = Vector3(0.5, 0.5, 0.5)
        hinge = body.add_component("HingeJoint")
        hinge.anchor = Vector3(-1.0, 0.0, 0.0)
        hinge.axis = Vector3(0.0, 0.0, 1.0)
        hinge.use_limits = True
        hinge.minimum_angle = -35.0
        hinge.maximum_angle = 35.0

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(240)

        rotated_anchor = _rotate_vector(rigidbody.rotation, Vector3(-1.0, 0.0, 0.0))
        world_anchor = rigidbody.position + rotated_anchor
        assert world_anchor.x == pytest.approx(0.0, abs=0.025)
        assert world_anchor.y == pytest.approx(3.0, abs=0.025)
        assert world_anchor.z == pytest.approx(0.0, abs=0.025)
        assert -35.5 <= hinge.current_angle <= 35.5
        assert abs(hinge.current_angle) > 20.0
        assert abs(rigidbody.position.z) < 0.025

    def test_connected_body_reference_and_serialized_contract(self, scene):
        support = scene.create_game_object("HingeSupport")
        support.transform.position = Vector3(0.0, 3.0, 0.0)
        support_body = support.add_component("Rigidbody")
        support_body.is_kinematic = True
        support.add_component("BoxCollider")

        body = scene.create_game_object("ConnectedHinge")
        body.transform.position = Vector3(1.0, 3.0, 0.0)
        rigidbody = body.add_component("Rigidbody")
        body.add_component("BoxCollider")
        hinge = body.add_component("HingeJoint")
        hinge.anchor = Vector3(-0.5, 0.0, 0.0)
        hinge.axis = Vector3(0.0, 0.0, 1.0)
        hinge.connected_body = support_body
        hinge.enable_collision = False

        assert hinge.connected_body.component_id == support_body.component_id
        document = json.loads(hinge.serialize())
        assert document["connected_body_component_id"] == support_body.component_id
        assert document["anchor"] == [-0.5, 0.0, 0.0]

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(120)
        assert math.isfinite(hinge.current_angle)
        rotated_anchor = _rotate_vector(rigidbody.rotation, Vector3(-0.5, 0.0, 0.0))
        world_anchor = rigidbody.position + rotated_anchor
        assert world_anchor.x == pytest.approx(0.5, abs=0.025)
        assert world_anchor.y == pytest.approx(3.0, abs=0.025)
        assert world_anchor.z == pytest.approx(0.0, abs=0.025)

    def test_cloned_hierarchy_remaps_connected_body(self, scene):
        root = scene.create_game_object("HingeGraph")
        support = scene.create_game_object("Support")
        support.set_parent(root, False)
        support_body = support.add_component("Rigidbody")
        support_body.is_kinematic = True
        support.add_component("BoxCollider")

        door = scene.create_game_object("Door")
        door.set_parent(root, False)
        door.transform.local_position = Vector3(1.0, 0.0, 0.0)
        door.add_component("Rigidbody")
        door.add_component("BoxCollider")
        hinge = door.add_component("HingeJoint")
        hinge.connected_body = support_body

        clone = GameObject.instantiate(root)
        cloned_support = clone.find_descendant("Support")
        cloned_door = clone.find_descendant("Door")
        cloned_support_body = cloned_support.get_component("Rigidbody")
        cloned_hinge = cloned_door.get_component("HingeJoint")

        assert cloned_hinge.connected_body.component_id == cloned_support_body.component_id
        assert cloned_hinge.connected_body.component_id != support_body.component_id

        document_clone = scene._instantiate_document(root.serialize_document())
        document_support_body = document_clone.find_descendant("Support").get_component("Rigidbody")
        document_hinge = document_clone.find_descendant("Door").get_component("HingeJoint")
        assert document_hinge.connected_body.component_id == document_support_body.component_id
        assert document_hinge.connected_body.component_id != support_body.component_id

    def test_scene_copy_remaps_serialized_connected_body(self, scene):
        support = scene.create_game_object("SerializedSupport")
        support_body = support.add_component("Rigidbody")
        support_body.is_kinematic = True
        support.add_component("BoxCollider")
        door = scene.create_game_object("SerializedDoor")
        door.add_component("Rigidbody")
        door.add_component("BoxCollider")
        hinge = door.add_component("HingeJoint")
        hinge.connected_body = support_body

        manager = SceneManager.instance()
        copied_scene = manager.create_scene("hinge_scene_copy")
        try:
            assert copied_scene._commit_document(scene.serialize_document())
            copied_support_body = copied_scene.find("SerializedSupport").get_component("Rigidbody")
            copied_hinge = copied_scene.find("SerializedDoor").get_component("HingeJoint")
            assert copied_hinge.connected_body.component_id == copied_support_body.component_id
            assert copied_hinge.connected_body.component_id != support_body.component_id
        finally:
            manager.unload_scene(copied_scene)
            manager.set_active_scene(scene)

    def test_connected_collision_switch_controls_pair_filter(self, scene):
        class CollisionProbe(InxComponent):
            events = None

            def awake(self):
                self.events = []

            def on_collision_enter(self, collision):
                self.events.append(collision)

        support = scene.create_game_object("CollisionSupport")
        support.transform.position = Vector3(0.0, 3.0, 0.0)
        support_body = support.add_component("Rigidbody")
        support_body.is_kinematic = True
        support.add_component("BoxCollider")

        body = scene.create_game_object("CollisionDoor")
        body.transform.position = Vector3(0.75, 3.0, 0.0)
        rigidbody = body.add_component("Rigidbody")
        rigidbody.use_gravity = False
        body.add_component("BoxCollider")
        hinge = body.add_component("HingeJoint")
        hinge.anchor = Vector3(-0.375, 0.0, 0.0)
        hinge.axis = Vector3(0.0, 0.0, 1.0)
        hinge.connected_body = support_body
        probe = body.add_component(CollisionProbe)

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(3)
        assert probe.events == []

        hinge.enable_collision = True
        _step_frames(3)
        assert len(probe.events) == 1

    def test_disable_releases_and_reenable_recreates_world_constraint(self, scene):
        body = scene.create_game_object("LifecycleHinge")
        body.transform.position = Vector3(1.0, 4.0, 0.0)
        rigidbody = body.add_component("Rigidbody")
        body.add_component("BoxCollider")
        hinge = body.add_component("HingeJoint")
        hinge.anchor = Vector3(-1.0, 0.0, 0.0)
        hinge.axis = Vector3(0.0, 0.0, 1.0)

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(30)
        hinge.enabled = False
        released_y = rigidbody.position.y
        _step_frames(30)
        assert rigidbody.position.y < released_y - 0.5

        hinge.enabled = True
        recreated_anchor = rigidbody.position + _rotate_vector(
            rigidbody.rotation, Vector3(-1.0, 0.0, 0.0)
        )
        _step_frames(30)
        held_anchor = rigidbody.position + _rotate_vector(
            rigidbody.rotation, Vector3(-1.0, 0.0, 0.0)
        )
        assert held_anchor.x == pytest.approx(recreated_anchor.x, abs=0.03)
        assert held_anchor.y == pytest.approx(recreated_anchor.y, abs=0.03)
        assert held_anchor.z == pytest.approx(recreated_anchor.z, abs=0.03)

    def test_destroy_and_play_stop_restore_do_not_retain_stale_constraint(self, scene):
        body = scene.create_game_object("RestoredHinge")
        body.transform.position = Vector3(1.0, 4.0, 0.0)
        body.add_component("Rigidbody")
        body.add_component("BoxCollider")
        hinge = body.add_component("HingeJoint")
        hinge.anchor = Vector3(-1.0, 0.0, 0.0)
        hinge.axis = Vector3(0.0, 0.0, 1.0)
        authored_document = scene.serialize_document()

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(2)
        scene.destroy_game_object(body)
        scene.process_pending_destroys()
        _step_frames(1)

        manager.stop()
        assert scene._commit_document(authored_document)
        restored = scene.find("RestoredHinge")
        assert restored is not None
        manager.play()
        manager.pause()
        _step_frames(5)
        restored_hinge = manager.get_active_scene().find("RestoredHinge").get_component("HingeJoint")
        assert math.isfinite(restored_hinge.current_angle)

    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("axis", Vector3(0.0, 0.0, 0.0)),
            ("minimum_angle", 1.0),
            ("maximum_angle", -1.0),
        ],
    )
    def test_hinge_rejects_invalid_configuration(self, scene, attribute, value):
        body = scene.create_game_object("InvalidHinge")
        body.add_component("Rigidbody")
        body.add_component("BoxCollider")
        hinge = body.add_component("HingeJoint")
        with pytest.raises((ValueError, TypeError)):
            setattr(hinge, attribute, value)


class TestSliderJoint:
    def test_world_slider_locks_rotation_and_all_but_authored_axis(self, scene):
        plate = scene.create_game_object("PressurePlate")
        plate.transform.position = Vector3(0.0, 3.0, 0.0)
        rigidbody = plate.add_component("Rigidbody")
        rigidbody.angular_drag = 0.0
        plate.add_component("BoxCollider")
        slider = plate.add_component("SliderJoint")
        slider.axis = Vector3(0.0, 1.0, 0.0)
        slider.minimum_distance = -1.0
        slider.maximum_distance = 0.5

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        rigidbody.add_force(Vector3(1000.0, 0.0, 700.0), ForceMode.Force)
        rigidbody.add_torque(Vector3(100.0, 200.0, 300.0), ForceMode.Force)
        _step_frames(240)

        assert rigidbody.position.x == pytest.approx(0.0, abs=0.02)
        assert rigidbody.position.z == pytest.approx(0.0, abs=0.02)
        assert rigidbody.position.y == pytest.approx(2.0, abs=0.03)
        assert slider.current_position == pytest.approx(-1.0, abs=0.03)
        assert rigidbody.rotation.x == pytest.approx(0.0, abs=0.01)
        assert rigidbody.rotation.y == pytest.approx(0.0, abs=0.01)
        assert rigidbody.rotation.z == pytest.approx(0.0, abs=0.01)
        assert abs(rigidbody.rotation.w) == pytest.approx(1.0, abs=0.01)

    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("axis", Vector3(0.0, 0.0, 0.0)),
            ("minimum_distance", 0.1),
            ("maximum_distance", -0.1),
        ],
    )
    def test_slider_rejects_invalid_configuration(self, scene, attribute, value):
        plate = scene.create_game_object("InvalidSlider")
        plate.add_component("Rigidbody")
        plate.add_component("BoxCollider")
        slider = plate.add_component("SliderJoint")
        with pytest.raises((ValueError, TypeError)):
            setattr(slider, attribute, value)


# ═══════════════════════════════════════════════════════════════════════════
# Gravity & free fall
# ═══════════════════════════════════════════════════════════════════════════

class TestPenetrationQuery:
    @pytest.mark.parametrize("shape_name", ["BoxCollider", "SphereCollider", "CapsuleCollider", "CylinderCollider"])
    def test_prediction_returns_separating_direction_without_moving_world(self, scene, shape_name):
        from Infernux.physics import Physics as PublicPhysics

        a = scene.create_game_object("predicted shape")
        a.transform.position = Vector3(20, 4, 0)
        collider_a = a.add_component(shape_name)
        if shape_name == "BoxCollider":
            collider_a.size = Vector3(1, 2, 1)
        else:
            collider_a.radius = 0.5
            if shape_name != "SphereCollider":
                collider_a.height = 2
                collider_a.direction = 1
        b = scene.create_game_object("other sphere")
        collider_b = b.add_component("SphereCollider")
        collider_b.radius = 0.5
        Physics.sync_transforms()
        before_count = Physics.body_count
        before_hit = Physics.raycast(Vector3(20, 10, 0), Vector3(0, -1, 0), 20)
        assert before_hit is not None
        identity = quatf(0, 0, 0, 1)
        hit = PublicPhysics.compute_penetration(collider_a, (0, 0, 0), identity,
                                               collider_b, (0.75, 0, 0), identity)
        assert hit is not None
        assert hit.direction.x == pytest.approx(-1, abs=1e-3)
        assert hit.distance == pytest.approx(0.25, abs=2e-3)
        shifted = hit.direction * (hit.distance + 0.005)
        assert PublicPhysics.compute_penetration(collider_a, shifted, identity,
                                                collider_b, (0.75, 0, 0), identity) is None
        assert a.transform.position.x == 20
        assert Physics.body_count == before_count
        after_hit = Physics.raycast(Vector3(20, 10, 0), Vector3(0, -1, 0), 20)
        assert after_hit.distance == pytest.approx(before_hit.distance)
        assert PublicPhysics.compute_penetration(collider_a, (0, 0, 0), identity,
                                                collider_b, (3, 0, 0), identity) is None

    def test_center_signed_scale_rotation_and_disabled_compound_member(self, scene):
        from Infernux.physics import Physics as PublicPhysics

        obj = scene.create_game_object("query compound member")
        obj.transform.local_scale = Vector3(-2, 1, 1)
        collider = obj.add_component("BoxCollider")
        collider.size = Vector3(1, 1, 1)
        collider.center = Vector3(0.5, 0, 0)
        sibling = obj.add_component("SphereCollider")
        sibling.center = Vector3(20, 0, 0)
        target = scene.create_game_object("target").add_component("SphereCollider")
        target.radius = 0.5
        target.is_trigger = True
        collider.enabled = False
        rotation = quatf(0, 0, 0.70710678, 0.70710678)
        hit = PublicPhysics.compute_penetration(collider, (0, 0, 0), rotation,
                                               target, (0, 0.25, 0), quatf(0, 0, 0, 1))
        assert hit.distance == pytest.approx(0.25, abs=1e-3)
        assert hit.direction.y == pytest.approx(-1, abs=1e-3)
        assert hit.point_b.y == pytest.approx(-0.25, abs=1e-3)
        assert collider.enabled is False

    @pytest.mark.parametrize("shape_name", ["CapsuleCollider", "CylinderCollider"])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_shape_axis_is_composed_with_predicted_rotation(self, scene, shape_name, axis):
        from Infernux.physics import Physics as PublicPhysics

        collider = scene.create_game_object("axis query").add_component(shape_name)
        collider.radius = 0.5
        collider.height = 2
        collider.direction = axis
        target = scene.create_game_object("axis target").add_component("SphereCollider")
        target.radius = 0.5
        rotated_axis = [(0, 1, 0), (-1, 0, 0), (0, 0, 1)][axis]
        hit = PublicPhysics.compute_penetration(collider, (0, 0, 0), quatf(0, 0, 0.70710678, 0.70710678),
            target, tuple(x * 1.25 for x in rotated_axis), quatf(0, 0, 0, 1))
        assert hit.distance == pytest.approx(0.25, abs=2e-3)
        actual = (hit.direction.x, hit.direction.y, hit.direction.z)
        assert actual == pytest.approx(tuple(-x for x in rotated_axis), abs=2e-3)

    def test_invalid_pose_and_unsupported_mesh_are_not_no_hit(self, scene):
        from Infernux.physics import Physics as PublicPhysics

        sphere = scene.create_game_object("sphere").add_component("SphereCollider")
        mesh = scene.create_game_object("uncooked mesh").add_component("MeshCollider")
        identity = quatf(0, 0, 0, 1)
        with pytest.raises(ValueError, match="non-zero quaternion"):
            PublicPhysics.compute_penetration(sphere, (0, 0, 0), quatf(0, 0, 0, 0), sphere, (1, 0, 0), identity)
        with pytest.raises(ValueError, match="finite"):
            PublicPhysics.compute_penetration(sphere, (float("nan"), 0, 0), identity, sphere, (1, 0, 0), identity)
        with pytest.raises(ValueError, match="convex = true"):
            PublicPhysics.compute_penetration(mesh, (0, 0, 0), identity, sphere, (1, 0, 0), identity)


class TestColliderRaycast:
    def test_targets_one_compound_member_without_world_filters(self, scene):
        obj = scene.create_game_object("filtered compound")
        obj.layer = 2  # IgnoreRaycast for ordinary world queries.
        sphere = obj.add_component("SphereCollider")
        sphere.radius = 0.5
        box = obj.add_component("BoxCollider")
        box.center = Vector3(3, 0, 0)
        box.size = Vector3(1, 2, 2)
        box.is_trigger = True
        Physics.sync_transforms()

        origin = Vector3(-2, 0, 0)
        direction = Vector3(1, 0, 0)
        assert Physics.raycast(origin, direction, 10) is None
        hit = box.raycast(origin, direction, 10)
        assert hit is not None
        assert hit.distance == pytest.approx(4.5, abs=1e-4)
        assert hit.point.x == pytest.approx(2.5, abs=1e-4)
        assert hit.normal.x == pytest.approx(-1, abs=1e-4)
        assert sphere.raycast(Vector3(1, 0, 0), direction, 10) is None

    def test_uses_current_world_pose_and_validates_the_ray(self, scene):
        obj = scene.create_game_object("rotated collider")
        obj.transform.position = Vector3(5, 2, 0)
        obj.transform.rotation = quatf(0, 0, 0.70710678, 0.70710678)
        box = obj.add_component("BoxCollider")
        box.size = Vector3(4, 2, 2)
        Physics.sync_transforms()

        hit = box.raycast(Vector3(5, -3, 0), Vector3(0, 1, 0), 10)
        assert hit is not None
        assert hit.point.y == pytest.approx(0, abs=1e-3)
        assert hit.normal.y == pytest.approx(-1, abs=1e-3)
        with pytest.raises(ValueError, match="direction"):
            box.raycast(Vector3(), Vector3(), 10)

    @pytest.mark.parametrize(
        ("shape_name", "expected_x"),
        [("BoxCollider", 6.0), ("SphereCollider", 6.0),
         ("CapsuleCollider", 5.5), ("CylinderCollider", 5.5)],
    )
    def test_closest_point_uses_primitive_geometry_and_preserves_interior_points(
            self, scene, shape_name, expected_x):
        obj = scene.create_game_object(f"closest {shape_name}")
        obj.transform.position = Vector3(5, 2, 0)
        collider = obj.add_component(shape_name)
        if shape_name == "BoxCollider":
            collider.size = Vector3(2, 2, 2)
        elif shape_name == "SphereCollider":
            collider.radius = 1
        else:
            collider.radius = 0.5
            collider.height = 2
            collider.direction = 1
        Physics.sync_transforms()

        outside = collider.closest_point(Vector3(8, 2, 0))
        assert outside.x == pytest.approx(expected_x, abs=2e-3)
        assert outside.y == pytest.approx(2, abs=2e-3)
        inside = collider.closest_point(Vector3(5, 2.1, 0))
        assert (inside.x, inside.y, inside.z) == pytest.approx((5, 2.1, 0), abs=1e-6)

    def test_closest_point_ignores_an_overlapping_compound_sibling(self, scene):
        obj = scene.create_game_object("closest compound")
        sphere = obj.add_component("SphereCollider")
        sphere.radius = 1
        box = obj.add_component("BoxCollider")
        box.center = Vector3(3, 0, 0)
        box.size = Vector3(1, 2, 2)
        Physics.sync_transforms()

        closest = box.closest_point(Vector3(0, 0, 0))
        assert closest.x == pytest.approx(2.5, abs=2e-3)
        assert closest.y == pytest.approx(0, abs=2e-3)
        assert closest.z == pytest.approx(0, abs=2e-3)

    def test_closest_point_supports_cooked_convex_mesh(self, scene):
        obj = scene.create_primitive(PrimitiveType.Cube, "closest convex mesh")
        mesh = obj.add_component("MeshCollider")
        mesh.convex = True
        Physics.sync_transforms()

        closest = mesh.closest_point(Vector3(3, 0, 0))
        assert closest.x == pytest.approx(0.5, abs=2e-3)
        assert closest.y == pytest.approx(0, abs=2e-3)
        assert closest.z == pytest.approx(0, abs=2e-3)

        inside = mesh.closest_point(Vector3(0.1, 0, 0))
        assert (inside.x, inside.y, inside.z) == pytest.approx((0.1, 0, 0), abs=1e-6)

    def test_closest_point_supports_ready_static_non_convex_mesh(self, scene):
        obj = scene.create_primitive(PrimitiveType.Cube, "closest triangle mesh")
        mesh = obj.add_component("MeshCollider")
        assert mesh.convex is False
        Physics.sync_transforms()

        closest = mesh.closest_point(Vector3(3, 0, 0))
        assert closest.x == pytest.approx(0.5, abs=2e-2)
        assert closest.y == pytest.approx(0, abs=2e-2)
        assert closest.z == pytest.approx(0, abs=2e-2)

        inside = mesh.closest_point(Vector3(0.1, 0, 0))
        assert (inside.x, inside.y, inside.z) == pytest.approx((0.1, 0, 0), abs=1e-6)

    def test_kinematic_non_convex_mesh_query_follows_published_transform(self, scene):
        import numpy as np

        from Infernux.physics import Physics as PublicPhysics

        obj = scene.create_game_object("kinematic triangle mesh")
        renderer = obj.add_component("MeshRenderer")
        positions = np.array(
            [[-1, 0, -1], [-1, 0, 1], [1, 0, -1], [1, 0, 1]], dtype=np.float32
        )
        normals = np.tile([0, 1, 0], (4, 1)).astype(np.float32)
        uvs = np.zeros((4, 2), dtype=np.float32)
        indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.uint32)
        renderer.set_inline_mesh_data(positions, normals, uvs, indices)
        mesh = obj.add_component("MeshCollider")
        body = obj.add_component("Rigidbody")
        body.is_kinematic = True
        Physics.sync_transforms()

        first = PublicPhysics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 20)
        assert first is not None
        assert first.point.y == pytest.approx(0.0, abs=1e-3)

        obj.transform.position = Vector3(0, 2, 0)
        Physics.sync_transforms()
        moved = PublicPhysics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 20)
        assert moved is not None
        assert moved.point.y == pytest.approx(2.0, abs=1e-3)

        # A visual mesh mutation is not a collision mutation until the author
        # explicitly publishes a new immutable cooking generation.
        positions[:, 1] = 1.0
        renderer.set_inline_mesh_data(positions, normals, uvs, indices)
        Physics.sync_transforms()
        unchanged = PublicPhysics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 20)
        assert unchanged is not None
        assert unchanged.point.y == pytest.approx(2.0, abs=1e-3)
        mesh.recook()
        Physics.sync_transforms()
        recooked = PublicPhysics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 20)
        assert recooked is not None
        assert recooked.point.y == pytest.approx(3.0, abs=1e-3)

    def test_compute_penetration_supports_ready_convex_mesh(self, scene):
        mesh_object = scene.create_primitive(PrimitiveType.Cube, "penetration convex mesh")
        mesh = mesh_object.add_component("MeshCollider")
        mesh.convex = True

        other = scene.create_game_object("penetration box")
        other.transform.position = Vector3(0.75, 0, 0)
        box = other.add_component("BoxCollider")
        box.size = Vector3(1, 1, 1)
        Physics.sync_transforms()

        hit = Physics.compute_penetration(
            mesh, mesh_object.transform.position, mesh_object.transform.rotation,
            box, other.transform.position, other.transform.rotation,
        )
        assert hit is not None
        assert hit.distance > 0.0
        assert hit.direction.x < -0.9

        other.transform.position = Vector3(3, 0, 0)
        Physics.sync_transforms()
        assert Physics.compute_penetration(
            mesh, mesh_object.transform.position, mesh_object.transform.rotation,
            box, other.transform.position, other.transform.rotation,
        ) is None


class TestRigidbodyStateBatch:
    def test_contact_stream_publishes_resolved_geometry_for_custom_solvers(self, scene):
        import numpy as np
        from Infernux.physics import Physics as PublicPhysics

        _make_ground(scene)
        _, ball = _make_ball(scene, pos=Vector3(0, 2, 0), radius=0.5)
        sm = SceneManager.instance()
        PublicPhysics.set_contact_event_stream_enabled(True)
        sm.play()
        sm.pause()
        _step_frames(40, sm.get_fixed_time_step())

        events = PublicPhysics.get_contact_events()
        PublicPhysics.set_contact_event_stream_enabled(False)
        assert set(events) == {"type", "body_ids", "sub_shape_ids", "contact_point",
                               "contact_normal", "relative_velocity"}
        assert events["type"].dtype == np.uint8
        assert events["body_ids"].dtype == np.uint32
        assert events["body_ids"].shape[1] == 2
        assert events["contact_point"].shape[1] == 3
        assert events["contact_normal"].shape[1] == 3
        assert events["relative_velocity"].shape[1] == 3
        assert events["type"].size > 0
        assert np.all(np.isfinite(events["contact_point"]))
        assert np.all(np.isfinite(events["contact_normal"]))
        assert np.all(events["body_ids"] != np.uint32(0xFFFFFFFF))

    def test_contact_impulse_stream_publishes_actual_solver_impulses(self, scene):
        """Expose the solved Jolt lambda without making custom solvers guess it."""
        import numpy as np
        from Infernux.physics import Physics as PublicPhysics

        _make_ground(scene)
        _, ball = _make_ball(scene, pos=Vector3(0, 2, 0), radius=0.5)
        sm = SceneManager.instance()
        PublicPhysics.set_contact_impulse_stream_enabled(True)
        try:
            sm.play()
            sm.pause()
            _step_frames(40, sm.get_fixed_time_step())
            impulses = PublicPhysics.get_contact_impulses()
        finally:
            PublicPhysics.set_contact_impulse_stream_enabled(False)

        assert set(impulses) == {"body_ids", "sub_shape_ids", "contact_point",
                                  "contact_normal", "impulse"}
        assert impulses["body_ids"].dtype == np.uint32
        assert impulses["body_ids"].shape[1] == 2
        assert impulses["sub_shape_ids"].dtype == np.uint32
        assert impulses["sub_shape_ids"].shape[1] == 2
        assert impulses["contact_point"].shape[1] == 3
        assert impulses["contact_normal"].shape[1] == 3
        assert impulses["impulse"].shape[1] == 3
        assert impulses["body_ids"].shape[0] > 0
        assert np.all(np.isfinite(impulses["contact_point"]))
        assert np.all(np.isfinite(impulses["contact_normal"]))
        assert np.all(np.isfinite(impulses["impulse"]))
        assert np.any(np.linalg.norm(impulses["impulse"], axis=1) > 0.0)
    def test_box_descriptors_are_flattened_and_reuse_capacity(self, scene):
        import Infernux as inx
        import numpy as np
        from Infernux.physics import Physics as PublicPhysics

        box_object = scene.create_game_object("box descriptor")
        box_object.transform.position = Vector3(3, 4, 5)
        box_object.transform.local_scale = Vector3(2, 3, 4)
        box_object.transform.rotation = quatf(0, 0, 0.70710678, 0.70710678)
        box_body = box_object.add_component("Rigidbody")
        box_body.use_gravity = False
        box = box_object.add_component("BoxCollider")
        box.center = Vector3(0.5, 0, 0)
        box.size = Vector3(1, 2, 3)
        sphere_body = _make_ball(scene, pos=Vector3(8, 4, 0))[1]
        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(1, manager.get_fixed_time_step())

        created = PublicPhysics.get_rigidbody_box_states([box_body, sphere_body])
        assert created["count"] == 1
        assert created["body_index"].tolist() == [0]
        np.testing.assert_allclose(created["center"], [[3, 5, 5]], atol=1e-5)
        np.testing.assert_allclose(created["half_extents"], [[1, 3, 6]], atol=1e-5)
        np.testing.assert_allclose(created["rotation"], [[0, 0, np.sqrt(0.5), np.sqrt(0.5)]], atol=1e-5)

        out = {
            "body_index": inx.buffer(shape=4, dtype=np.int32, device="cpu"),
            "center": inx.buffer(shape=4, dtype=inx.vector3, device="cpu"),
            "rotation": inx.buffer(shape=4, dtype=inx.vector4, device="cpu"),
            "half_extents": inx.buffer(shape=4, dtype=inx.vector3, device="cpu"),
            "friction": inx.buffer(shape=4, dtype=np.float32, device="cpu"),
            "bounciness": inx.buffer(shape=4, dtype=np.float32, device="cpu"),
        }
        assert PublicPhysics.get_rigidbody_box_states([box_body, sphere_body], out=out) is out
        assert out["count"] == 1
        assert out["body_index"][0] == 0
        np.testing.assert_allclose(out["half_extents"].numpy()[0], [1, 3, 6])

    def test_broadphase_bounds_query_returns_only_nearby_rigidbodies(self, scene):
        from Infernux.physics import Physics as PublicPhysics

        nearby = [_make_ball(scene, pos=Vector3(index * 1.5, 2, 0))[1] for index in range(3)]
        for index in range(128):
            _make_ball(scene, pos=Vector3(1000 + index * 3, 2, 0))
        static = scene.create_game_object("near static collider")
        static.transform.position = Vector3(0, 2, 0)
        static.add_component("BoxCollider")
        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(1, manager.get_fixed_time_step())

        candidates = PublicPhysics.query_rigidbodies_in_bounds((-1, 0, -2), (5, 4, 2))
        assert {body.component_id for body in candidates} == {body.component_id for body in nearby}
        assert PublicPhysics.query_rigidbodies_in_bounds((900, 0, -2), (999, 4, 2)) == []

    def test_broadphase_bounds_query_rejects_inverted_bounds(self, scene):
        from Infernux.physics import Physics as PublicPhysics

        with pytest.raises(ValueError, match="minimum must not exceed maximum"):
            PublicPhysics.query_rigidbodies_in_bounds((1, 0, 0), (0, 1, 1))

    def test_reuses_engine_cpu_buffers_for_snapshot_and_feedback(self, scene):
        import Infernux as inx
        import numpy as np
        from Infernux.physics import Physics as PublicPhysics

        bodies = [_make_ball(scene, pos=Vector3(index * 2, 4, 0), mass=2)[1]
                  for index in range(3)]
        for body in bodies:
            body.use_gravity = False
        manager = SceneManager.instance()
        manager.play()
        manager.pause()

        capacity = 8
        state = {
            "position": inx.buffer(shape=capacity, dtype=inx.vector3, device="cpu"),
            "rotation": inx.buffer(shape=capacity, dtype=inx.vector4, device="cpu"),
            "center_of_mass": inx.buffer(shape=capacity, dtype=inx.vector3, device="cpu"),
            "linear_velocity": inx.buffer(shape=capacity, dtype=inx.vector3, device="cpu"),
            "angular_velocity": inx.buffer(shape=capacity, dtype=inx.vector3, device="cpu"),
            "inverse_mass": inx.buffer(shape=capacity, dtype=inx.vector3, device="cpu"),
            "inverse_inertia": inx.buffer(shape=(capacity, 3, 3), dtype=np.float32, device="cpu"),
        }
        assert PublicPhysics.get_rigidbody_states(bodies, out=state) is state
        np.testing.assert_allclose(state["position"].numpy()[:3], [[0, 4, 0], [2, 4, 0], [4, 4, 0]])

        linear = inx.buffer(shape=3, dtype=inx.vector3, device="cpu",
                            data=np.full((3, 3), 0.25, np.float32))
        angular = inx.buffer(shape=3, dtype=inx.vector3, device="cpu")
        PublicPhysics.apply_rigidbody_impulses(bodies, linear, angular)
        PublicPhysics.get_rigidbody_states(bodies, out=state)
        np.testing.assert_allclose(state["linear_velocity"].numpy()[:3], np.full((3, 3), 0.125), atol=1e-6)

    def test_aggregated_impulse_batch_and_boundary_validation(self, scene):
        import numpy as np
        from Infernux.physics import Physics as PublicPhysics

        bodies = [_make_ball(scene, pos=Vector3(i * 3, 4, 0), mass=2 + i * 2)[1]
                  for i in range(2)]
        for rb in bodies:
            rb.use_gravity = False
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        impulse = np.array([[0.2, 0.4, 0.6], [-0.1, 0.3, 0.5]], dtype=np.float32)
        torque = np.array([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]], dtype=np.float32)
        before = PublicPhysics.get_rigidbody_states(bodies)
        with pytest.raises(ValueError):
            PublicPhysics.apply_rigidbody_impulses(bodies, impulse[:1], torque)
        with pytest.raises(TypeError):
            PublicPhysics.apply_rigidbody_impulses(bodies, impulse.astype(np.float64), torque)
        invalid = torque.copy()
        invalid[1, 1] = np.nan
        with pytest.raises(ValueError):
            PublicPhysics.apply_rigidbody_impulses(bodies, impulse, invalid)
        np.testing.assert_array_equal(PublicPhysics.get_rigidbody_states(bodies)["linear_velocity"], 0)
        PublicPhysics.apply_rigidbody_impulses(bodies, impulse, torque)
        after = PublicPhysics.get_rigidbody_states(bodies)
        np.testing.assert_allclose(after["linear_velocity"], before["inverse_mass"] * impulse, atol=1e-6)
        np.testing.assert_allclose(after["angular_velocity"],
                                   np.einsum("nij,nj->ni", before["inverse_inertia"], torque), atol=1e-6)
        _step_frames(1, sm.get_fixed_time_step())
        assert all(rb.velocity.y > 0 for rb in bodies)

    @pytest.mark.parametrize("constraints", [0,
        int(RigidbodyConstraints.FreezePositionX) | int(RigidbodyConstraints.FreezeRotationY)])
    def test_snapshot_predicts_actual_off_center_impulse(self, scene, constraints):
        import numpy as np
        from Infernux.physics import Physics as PublicPhysics

        obj = scene.create_game_object("rotated body snapshot")
        obj.transform.position = Vector3(3, 4, 5)
        obj.transform.rotation = quatf(0.38268343, 0, 0, 0.92387953)
        rb = obj.add_component("Rigidbody")
        rb.mass = 2
        rb.use_gravity = False
        rb.constraints = int(constraints)
        collider = obj.add_component("BoxCollider")
        collider.size = Vector3(1, 2, 3)
        collider.center = Vector3(0.5, 0, 0)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        state = PublicPhysics.get_rigidbody_states([rb])
        assert set(state) == {"position", "rotation", "center_of_mass", "linear_velocity",
                              "angular_velocity", "inverse_mass", "inverse_inertia"}
        for array in state.values():
            assert array.dtype == np.float32 and array.flags.c_contiguous
        np.testing.assert_allclose(state["position"], [[3, 4, 5]], atol=1e-5)
        np.testing.assert_allclose(state["center_of_mass"], [[3.5, 4, 5]], atol=1e-5)
        expected_mass = [[0 if constraints else 0.5, 0.5, 0.5]]
        np.testing.assert_allclose(state["inverse_mass"], expected_mass)
        # Local inverse inertia of a 2 kg box, rotated 45 degrees about X.
        diagonal = 6 / np.array([13, 10, 5], dtype=np.float64)
        c = np.sqrt(0.5)
        rotation = np.array([[1, 0, 0], [0, c, -c], [0, c, c]])
        expected_inertia = rotation @ np.diag(diagonal) @ rotation.T
        if constraints:
            expected_inertia[1, :] = expected_inertia[:, 1] = 0
        np.testing.assert_allclose(state["inverse_inertia"][0], expected_inertia, atol=1e-5)
        impulse = np.array([0.2, -0.3, 0.4], dtype=np.float32)
        offset = np.array([0.4, 0.1, -0.2], dtype=np.float32)
        point = state["center_of_mass"][0] + offset
        rb.add_force_at_position(Vector3(*impulse), Vector3(*point), ForceMode.Impulse)
        after = PublicPhysics.get_rigidbody_states([rb])
        np.testing.assert_allclose(after["linear_velocity"][0], state["inverse_mass"][0] * impulse, atol=1e-5)
        np.testing.assert_allclose(after["angular_velocity"][0],
                                   state["inverse_inertia"][0] @ np.cross(offset, impulse), atol=1e-5)
        np.testing.assert_array_equal(state["linear_velocity"], 0)  # Owned snapshot, not live storage.

    def test_kinematic_and_empty_batches(self, scene):
        import numpy as np
        from Infernux.physics import Physics as PublicPhysics

        _, rb = _make_ball(scene)
        rb.is_kinematic = True
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        rb.velocity = Vector3(1, 2, 3)
        state = PublicPhysics.get_rigidbody_states([rb])
        np.testing.assert_array_equal(state["inverse_mass"], 0)
        np.testing.assert_array_equal(state["inverse_inertia"], 0)
        np.testing.assert_allclose(state["linear_velocity"], [[1, 2, 3]])
        empty = PublicPhysics.get_rigidbody_states([])
        assert empty["position"].shape == (0, 3)
        assert empty["rotation"].shape == (0, 4)
        assert empty["inverse_inertia"].shape == (0, 3, 3)

    def test_unpublished_body_has_no_synthetic_physics_state(self, scene):
        import numpy as np
        from Infernux.physics import Physics as PublicPhysics

        rb = scene.create_game_object("no collider").add_component("Rigidbody")
        with pytest.raises(RuntimeError, match="active physics body"):
            PublicPhysics.get_rigidbody_states([rb])
        _, valid = _make_ball(scene)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        with pytest.raises(RuntimeError, match="active physics body"):
            PublicPhysics.apply_rigidbody_impulses([valid, rb], np.ones((2, 3), np.float32),
                                                  np.zeros((2, 3), np.float32))
        np.testing.assert_array_equal(PublicPhysics.get_rigidbody_states([valid])["linear_velocity"], 0)

    def test_zero_feedback_preserves_sleep_and_kinematics_ignore_impulses(self, scene):
        import numpy as np
        from Infernux.physics import Physics as PublicPhysics

        _, dynamic = _make_ball(scene, pos=Vector3(0, 4, 0))
        _, kinematic = _make_ball(scene, pos=Vector3(4, 4, 0))
        kinematic.is_kinematic = True
        dynamic.use_gravity = False
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        dynamic.sleep()
        kinematic.velocity = Vector3(1, 0, 0)
        impulses = np.array([[0, 0, 0], [1, 1, 1]], np.float32)
        PublicPhysics.apply_rigidbody_impulses([dynamic, kinematic], impulses, impulses)
        assert dynamic.is_sleeping()
        np.testing.assert_array_equal(PublicPhysics.get_rigidbody_states([kinematic])["linear_velocity"], [[1, 0, 0]])
        empty = np.empty((0, 3), np.float32)
        PublicPhysics.apply_rigidbody_impulses([], empty, empty)


class TestPointVelocity:
    def test_off_center_feedback_impulse_changes_linear_and_angular_motion(self, scene):
        _, rb = _make_ball(scene, pos=Vector3(3, 4, 5), mass=2, radius=0.5)
        rb.use_gravity = False
        rb.drag = rb.angular_drag = 0
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        center = rb.world_center_of_mass
        point = Vector3(center.x + 0.5, center.y, center.z)
        rb.add_force_at_position(Vector3(0, 0, 1), point, ForceMode.Impulse)
        # A published Jolt body receives an impulse immediately, exactly once.
        # The coupling solver must therefore submit at its chosen fixed boundary.
        assert rb.velocity.z == pytest.approx(0.5, abs=1e-4)
        _step_frames(1, sm.get_fixed_time_step())
        assert rb.velocity.z == pytest.approx(0.5, abs=1e-4)
        # Sphere inertia = 2/5 * mass * radius^2 = 0.2 kg m^2.
        assert rb.angular_velocity.y == pytest.approx(-2.5, abs=1e-4)
        center = rb.world_center_of_mass
        velocity = rb.get_point_velocity(Vector3(center.x + 0.5, center.y, center.z))
        assert velocity.z == pytest.approx(1.75, abs=1e-4)
        _step_frames(1, sm.get_fixed_time_step())
        assert rb.velocity.z == pytest.approx(0.5, abs=1e-4)
        assert rb.angular_velocity.y == pytest.approx(-2.5, abs=1e-4)

    def test_world_points_include_rotation_about_offset_center_of_mass(self, scene):
        import numpy as np

        ball, rb = _make_ball(scene, pos=Vector3(3, 4, 5))
        rb.use_gravity = False
        ball.get_component("SphereCollider").center = Vector3(0.5, 0, 0)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        rb.velocity = Vector3(1, 2, 3)
        rb.angular_velocity = Vector3(0, 0, 2)
        center = rb.world_center_of_mass
        point = Vector3(center.x + 1, center.y, center.z)
        velocity = rb.get_point_velocity(point)
        assert (velocity.x, velocity.y, velocity.z) == pytest.approx((1, 4, 3))

        points = np.array([[center.x, center.y, center.z],
                           [point.x, point.y, point.z]], dtype=np.float32)
        output = np.empty_like(points)
        assert rb.get_point_velocities(points, output) is output
        np.testing.assert_allclose(output, [[1, 2, 3], [1, 4, 3]])
        assert rb.get_point_velocities(points, points) is points
        np.testing.assert_allclose(points, output)

    def test_batch_uses_caller_owned_numpy_output(self, scene):
        import numpy as np

        _, rb = _make_ball(scene)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        points = np.zeros((2, 3), dtype=np.float32)
        output = np.full_like(points, 17)
        with pytest.raises(ValueError):
            rb.get_point_velocities(points, output[:1])
        with pytest.raises(TypeError):
            rb.get_point_velocities(points.astype(np.float64), output)
        points[1, 0] = np.nan
        with pytest.raises(ValueError):
            rb.get_point_velocities(points, output)
        np.testing.assert_array_equal(output, 17)
        points[1, 0] = 0
        output.flags.writeable = False
        with pytest.raises(ValueError):
            rb.get_point_velocities(points, output)
        storage = np.zeros((3, 3), dtype=np.float32)
        with pytest.raises(ValueError):
            rb.get_point_velocities(storage[:2], storage[1:])
        empty = np.empty((0, 3), dtype=np.float32)
        assert rb.get_point_velocities(empty, empty) is empty


class TestGravity:
    def test_player_style_scene_rebuild_preserves_static_collisions(self, scene):
        """A full native graph rebuild must still publish physics bodies on Play."""
        Physics.set_gravity(Vector3(0, -9.81, 0))
        _make_ground(scene)
        _make_ball(scene, pos=Vector3(0, 4, 0))

        snapshot = scene.serialize_document()
        transaction = SceneDocumentTransaction(
            scene,
            document=snapshot,
            after_publish=lambda: scene.set_playing(True),
        )
        assert transaction.run_to_completion() is True

        restored_ball = scene.find("Ball")
        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(180)

        assert restored_ball.transform.position.y > 0.9

    def test_set_gravity_persists(self, scene):
        Physics.set_gravity(Vector3(0, -9.81, 0))
        g = Physics.get_gravity()
        assert g.y == pytest.approx(-9.81, abs=0.01)

    def test_custom_gravity(self, scene):
        Physics.set_gravity(Vector3(0, -20, 0))
        g = Physics.get_gravity()
        assert g.y == pytest.approx(-20, abs=0.01)
        Physics.set_gravity(Vector3(0, -9.81, 0))  # restore

    def test_ball_falls_under_gravity(self, scene):
        """A dynamic sphere above the ground should lose altitude."""
        Physics.set_gravity(Vector3(0, -9.81, 0))
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 10, 0))

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        y0 = ball.transform.position.y
        _step_frames(60)
        y1 = ball.transform.position.y

        assert y1 < y0, f"Ball should fall: {y0} → {y1}"

    def test_no_gravity_ball_stays(self, scene):
        """With gravity off a dynamic body should not fall."""
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 10, 0))
        rb.use_gravity = False

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        y0 = ball.transform.position.y
        _step_frames(30)
        y1 = ball.transform.position.y

        assert y1 == pytest.approx(y0, abs=0.1)


class TestRigidbodyPose:
    def test_velocity_set_before_deferred_body_creation_is_applied(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        _, rb = _make_ball(scene, pos=Vector3(0, 5, 0))
        rb.velocity = Vector3(2, 3, 4)
        rb.angular_velocity = Vector3(0.25, 0.5, 0.75)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        assert tuple(rb.velocity) == pytest.approx((2, 3, 4))
        assert tuple(rb.angular_velocity) == pytest.approx((0.25, 0.5, 0.75), abs=0.001)

    def test_pose_assignment_preserves_velocity(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        ball, rb = _make_ball(scene, pos=Vector3(0, 5, 0))
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        rb.velocity = Vector3(3, 4, 5)
        rb.angular_velocity = Vector3(0.5, 1.0, 1.5)
        rb.position = Vector3(7, 8, 9)
        rb.rotation = quatf(0.0, 0.0, 0.70710678, 0.70710678)

        assert tuple(rb.velocity) == pytest.approx((3, 4, 5))
        assert tuple(rb.angular_velocity) == pytest.approx((0.5, 1.0, 1.5))
        assert tuple(rb.position) == pytest.approx((7, 8, 9))
        assert tuple(ball.transform.position) == pytest.approx((7, 8, 9))

    def test_pose_assignment_rejects_invalid_values(self, scene):
        _, rb = _make_ball(scene)
        with pytest.raises(ValueError, match="finite"):
            rb.position = Vector3(float("nan"), 0, 0)
        with pytest.raises(ValueError, match="non-zero quaternion"):
            rb.rotation = quatf(0, 0, 0, 0)
        with pytest.raises(ValueError, match="finite"):
            rb.velocity = Vector3(1, 0, 0) / 0.0

    def test_kinematic_move_does_not_teleport_transform_before_step(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        body, rigidbody = _make_ball(scene, pos=Vector3(0, 5, 0))
        rigidbody.is_kinematic = True

        class KinematicMoveProbe(InxComponent):
            def awake(self):
                self.position_during_move = None
                self.did_move = False

            def fixed_update(self, _delta_time):
                if self.did_move:
                    return
                rigidbody.move_position(Vector3(4, 5, 0))
                self.position_during_move = tuple(body.transform.position)
                self.did_move = True

        probe = body.add_component(KinematicMoveProbe)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        sm.step()

        assert probe.position_during_move == pytest.approx((0, 5, 0))
        assert tuple(rigidbody.position) == pytest.approx((4, 5, 0), abs=0.001)
        assert tuple(body.transform.position) == pytest.approx((4, 5, 0), abs=0.001)

    def test_kinematic_move_rejects_non_kinematic_body(self, scene):
        _, rigidbody = _make_ball(scene)

        with pytest.raises(RuntimeError, match="requires a kinematic Rigidbody"):
            rigidbody.move_position(Vector3(1, 2, 3))


class TestFixedTiming:
    def test_manual_step_is_exclusive_and_runs_one_shared_fixed_frame(self, scene):
        """Manual stepping must be the paused form of the automatic path.

        Calling ``step`` while Play is running is intentionally a no-op, so a
        tooling/editor request cannot advance the World beside the automatic
        accumulator. Once paused, one call runs exactly one fixed bundle and
        the normal script phases around it, regardless of the display delta
        supplied by the caller.
        """
        probe_object = scene.create_game_object("ManualStepProbe")

        class ManualStepProbe(InxComponent):
            def awake(self):
                self.calls = []

            def fixed_update(self, _delta_time):
                self.calls.append("fixed")

            def physics_pre_step(self, _delta_time):
                self.calls.append("pre")

            def physics_post_step(self, _delta_time):
                self.calls.append("post")

            def update(self, _delta_time):
                self.calls.append("update")

            def late_update(self, _delta_time):
                self.calls.append("late")

        probe = probe_object.add_component(ManualStepProbe)
        manager = SceneManager.instance()
        manager.play()

        fixed_delta = manager.get_fixed_time_step()
        before_fixed = manager.fixed_time
        before_frame = manager.runtime_frame_count

        # Automatic Play owns the World while unpaused; the manual request
        # cannot create a second simulation or run a duplicate script frame.
        manager.step(0.5)
        assert manager.fixed_time == pytest.approx(before_fixed)
        assert manager.runtime_frame_count == before_frame
        assert probe.calls == []

        manager.pause()
        manager.step(0.5)

        assert manager.fixed_time == pytest.approx(before_fixed + fixed_delta)
        assert manager.runtime_frame_count == before_frame + 1
        assert probe.calls == ["fixed", "pre", "post", "update", "late"]

    def test_public_physics_stages_bridge_the_same_jolt_step(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        body, rigidbody = _make_ball(scene, pos=Vector3(0, 5, 0))
        rigidbody.use_gravity = False
        rigidbody.drag = 0.0

        class PhysicsStageProbe(InxComponent):
            def awake(self):
                self.calls = []
                self.fixed_velocity = None
                self.pre_velocity = None
                self.post_velocity = None
                self.post_position = None

            def fixed_update(self, _delta_time):
                self.calls.append("fixed")
                self.fixed_velocity = tuple(rigidbody.velocity)

            def physics_pre_step(self, _delta_time):
                self.calls.append("pre")
                self.pre_velocity = tuple(rigidbody.velocity)
                rigidbody.add_force(
                    Vector3(3, 0, 0), ForceMode.VelocityChange
                )

            def physics_post_step(self, _delta_time):
                self.calls.append("post")
                self.post_velocity = tuple(rigidbody.velocity)
                self.post_position = tuple(body.transform.position)

        probe = body.add_component(PhysicsStageProbe)
        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()

        assert probe.calls == ["fixed", "pre", "post"]
        assert probe.fixed_velocity == pytest.approx((0, 0, 0), abs=1e-4)
        assert probe.pre_velocity == pytest.approx((0, 0, 0), abs=1e-4)
        assert probe.post_velocity == pytest.approx((3, 0, 0), abs=1e-4)
        assert probe.post_position[0] == pytest.approx(
            3.0 * manager.get_fixed_time_step(), abs=1e-3
        )

    def test_fixed_time_is_current_inside_fixed_update(self, scene):
        probe_object = scene.create_game_object("FixedTimeProbe")

        class FixedTimeProbe(InxComponent):
            def awake(self):
                self.samples = []

            def fixed_update(self, _delta_time):
                self.samples.append((Time.fixed_time, Time.fixed_unscaled_time))

        probe = probe_object.add_component(FixedTimeProbe)
        manager = SceneManager.instance()
        manager.time_scale = 1.0
        manager.play()
        manager.pause()
        manager.step()
        manager.step()

        fixed_delta = manager.get_fixed_time_step()
        assert probe.samples == pytest.approx(
            [(fixed_delta, fixed_delta), (fixed_delta * 2.0, fixed_delta * 2.0)]
        )


# ═══════════════════════════════════════════════════════════════════════════
# Collision — ball lands on ground
# ═══════════════════════════════════════════════════════════════════════════

class TestCollision:
    def test_ball_lands_on_ground(self, scene):
        """Ball should collide with ground and stop roughly at ground level."""
        Physics.set_gravity(Vector3(0, -9.81, 0))
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 5, 0), radius=0.5)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(180)  # 3 seconds — plenty of time to settle
        y = ball.transform.position.y
        # Ground top is at Y=0.5 (center 0, scale 1 → half-height 0.5)
        # Ball radius 0.5 → ball center rests at about Y=1.0
        assert y < 5.0, "Ball should have fallen"
        assert y > -1.0, "Ball should not fall through ground"

    def test_bounce_combine_mode_changes_contact_response(self, scene):
        ground = _make_ground(scene)
        ground_collider = ground.get_component("BoxCollider")
        _assign_physic_material(ground_collider, bounciness=0.0, bounce_combine=0)

        maximum_ball, _ = _make_ball(scene, pos=Vector3(-2, 3, 0))
        maximum_collider = maximum_ball.get_component("SphereCollider")
        _assign_physic_material(maximum_collider, bounciness=1.0, bounce_combine=3)

        minimum_ball, _ = _make_ball(scene, pos=Vector3(2, 3, 0))
        minimum_collider = minimum_ball.get_component("SphereCollider")
        _assign_physic_material(minimum_collider, bounciness=1.0, bounce_combine=1)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(40)

        maximum_peak = maximum_ball.transform.position.y
        minimum_peak = minimum_ball.transform.position.y
        for _ in range(50):
            sm.step(1.0 / 60.0)
            maximum_peak = max(maximum_peak, maximum_ball.transform.position.y)
            minimum_peak = max(minimum_peak, minimum_ball.transform.position.y)

        assert maximum_peak > minimum_peak + 0.75

    def test_compound_material_uses_contacted_subshape(self, scene):
        ground = scene.create_game_object("CompoundMaterialGround")
        primary = ground.add_component("BoxCollider")
        primary.size = Vector3(1, 1, 1)
        primary.center = Vector3(20, 0, 0)
        _assign_physic_material(primary, bounciness=1.0, bounce_combine=3)

        contacted = ground.add_component("BoxCollider")
        contacted.size = Vector3(10, 1, 10)
        _assign_physic_material(contacted, bounciness=0.0, bounce_combine=1)

        ball, _ = _make_ball(scene, pos=Vector3(0, 3, 0))
        ball_collider = ball.get_component("SphereCollider")
        _assign_physic_material(ball_collider, bounciness=1.0, bounce_combine=0)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(40)

        peak_after_contact = ball.transform.position.y
        for _ in range(50):
            sm.step(1.0 / 60.0)
            peak_after_contact = max(peak_after_contact, ball.transform.position.y)

        assert peak_after_contact < 1.5

    def test_box_stack_settles_without_lateral_drift(self, scene):
        Physics.set_gravity(Vector3(0, -9.81, 0))
        ground = scene.create_game_object("StackGround")
        ground.transform.position = Vector3(0, -0.5, 0)
        ground_collider = ground.add_component("BoxCollider")
        ground_collider.size = Vector3(20, 1, 20)

        boxes = []
        for index in range(8):
            box = scene.create_game_object(f"StackBox{index}")
            box.transform.position = Vector3(0, 0.5 + index * 1.01, 0)
            rigidbody = box.add_component("Rigidbody")
            collider = box.add_component("BoxCollider")
            collider.size = Vector3(1, 1, 1)
            boxes.append((box, rigidbody))

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(600)

        for index, (box, rigidbody) in enumerate(boxes):
            assert abs(box.transform.position.x) < 0.1
            assert abs(box.transform.position.z) < 0.1
            assert box.transform.position.y == pytest.approx(0.5 + index, abs=0.12)
            assert rigidbody.is_sleeping()

    def test_contact_friction_slows_non_rotating_box(self, scene):
        Physics.set_gravity(Vector3(0, -9.81, 0))

        rigidbodies = []
        for name, z, friction in (("Frictionless", -3.0, 0.0), ("HighFriction", 3.0, 1.0)):
            floor = scene.create_game_object(f"{name}Floor")
            floor.transform.position = Vector3(0, -0.5, z)
            floor_collider = floor.add_component("BoxCollider")
            floor_collider.size = Vector3(30, 1, 4)
            _assign_physic_material(floor_collider, friction=friction, friction_combine=3)

            box = scene.create_game_object(f"{name}Box")
            box.transform.position = Vector3(-8, 0.5, z)
            rigidbody = box.add_component("Rigidbody")
            rigidbody.constraints = int(RigidbodyConstraints.FreezeRotation)
            rigidbody.drag = 0.0
            collider = box.add_component("BoxCollider")
            collider.size = Vector3(1, 1, 1)
            _assign_physic_material(collider, friction=friction, friction_combine=3)
            rigidbody.velocity = Vector3(5, 0, 0)
            rigidbodies.append(rigidbody)

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(90)

        frictionless, high_friction = rigidbodies
        assert frictionless.velocity.x > 4.5
        assert abs(high_friction.velocity.x) < 0.25


# ═══════════════════════════════════════════════════════════════════════════
# Rigidbody properties in play mode
# ═══════════════════════════════════════════════════════════════════════════

class TestRigidbodyInScene:
    @pytest.mark.parametrize(
        "attribute,value",
        [
            ("mass", 0.0),
            ("drag", float("nan")),
            ("angular_drag", -0.1),
            ("constraints", 1),
            ("interpolation", 7),
            ("max_linear_velocity", -1.0),
        ],
    )
    def test_rigidbody_setters_reject_invalid_values(self, scene, attribute, value):
        _, rigidbody = _make_ball(scene)
        with pytest.raises((ValueError, TypeError)):
            setattr(rigidbody, attribute, value)

    def test_mass_affects_physics(self, scene):
        _make_ground(scene)
        heavy, rb_heavy = _make_ball(scene, pos=Vector3(-3, 10, 0), mass=100.0)
        light, rb_light = _make_ball(scene, pos=Vector3(3, 10, 0), mass=0.1)
        rb_heavy.use_gravity = True
        rb_light.use_gravity = True

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(30)

        # Both should have fallen (gravity is mass-independent in Newtonian physics,
        # but drag or solver differences may cause slight variation)
        assert heavy.transform.position.y < 10.0
        assert light.transform.position.y < 10.0

    def test_kinematic_body_does_not_fall(self, scene):
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 10, 0))
        rb.is_kinematic = True

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        y0 = ball.transform.position.y
        _step_frames(60)
        y1 = ball.transform.position.y
        assert y1 == pytest.approx(y0, abs=0.1)

    def test_velocity_readable_during_fall(self, scene):
        Physics.set_gravity(Vector3(0, -9.81, 0))
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 20, 0))

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(30)

        vel = rb.velocity
        # Ball is falling, so velocity Y should be negative
        assert vel.y < 0, f"Velocity should be downward: {vel.y}"

    def test_add_force_impulse(self, scene):
        """An upward impulse should propel the ball up."""
        Physics.set_gravity(Vector3(0, 0, 0))  # disable gravity for a clean test
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 5, 0))

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)
        rb.add_force(Vector3(0, 100, 0), ForceMode.Impulse)
        _step_frames(30)

        assert ball.transform.position.y > 5.0, "Impulse should move ball up"
        Physics.set_gravity(Vector3(0, -9.81, 0))


class TestRigidbodyNumerics:
    def test_start_force_survives_deferred_body_creation(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        body, rigidbody = _make_ball(scene, pos=Vector3(0, 5, 0), mass=4.0)
        rigidbody.use_gravity = False
        rigidbody.drag = 0.0
        rigidbody.angular_drag = 0.0
        rigidbody.max_angular_velocity = 100.0

        class StartForceProbe(InxComponent):
            def start(self):
                rigidbody.add_force(Vector3(3, 0, 0), ForceMode.VelocityChange)
                rigidbody.add_torque(Vector3(0, 0, 1), ForceMode.VelocityChange)

        body.add_component(StartForceProbe)
        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()

        assert tuple(rigidbody.velocity) == pytest.approx((3, 0, 0), abs=1e-4)
        assert tuple(rigidbody.angular_velocity) == pytest.approx((0, 0, 1), abs=1e-4)

    @pytest.mark.parametrize(
        "mode,expected_light,expected_heavy",
        [
            (ForceMode.Force, 0.2, 0.02),
            (ForceMode.Acceleration, 0.2, 0.2),
            (ForceMode.Impulse, 10.0, 1.0),
            (ForceMode.VelocityChange, 10.0, 10.0),
        ],
    )
    def test_linear_force_modes_follow_mass_contract(self, scene, mode, expected_light, expected_heavy):
        Physics.set_gravity(Vector3(0, 0, 0))
        _, light = _make_ball(scene, pos=Vector3(-2, 5, 0), mass=1.0)
        _, heavy = _make_ball(scene, pos=Vector3(2, 5, 0), mass=10.0)
        for rigidbody in (light, heavy):
            rigidbody.use_gravity = False
            rigidbody.drag = 0.0
            rigidbody.max_linear_velocity = 100.0

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()

        force = Vector3(10, 0, 0)
        light.add_force(force, mode)
        heavy.add_force(force, mode)
        manager.step()

        assert light.velocity.x == pytest.approx(expected_light, abs=1e-4)
        assert heavy.velocity.x == pytest.approx(expected_heavy, abs=1e-4)

    def test_continuous_force_integrates_over_half_second(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        body = scene.create_game_object("ContinuousForceBox")
        body.transform.position = Vector3(0, 2, 0)
        body.transform.local_scale = Vector3(1.8, 0.7, 3.5)
        body.add_component("BoxCollider")
        rigidbody = body.add_component("Rigidbody")
        rigidbody.mass = 1.0
        rigidbody.use_gravity = False
        rigidbody.drag = 0.0
        rigidbody.max_linear_velocity = 500.0

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()
        start_z = body.transform.position.z
        for _ in range(25):
            rigidbody.add_force(Vector3(0, 0, 28), ForceMode.Force)
            manager.step()

        assert rigidbody.velocity.z == pytest.approx(14.0, abs=0.05)
        assert body.transform.position.z - start_z == pytest.approx(3.64, abs=0.15)

    @pytest.mark.parametrize(
        "mode,input_value,expected_velocity",
        [
            (ForceMode.Acceleration, 2.0, 0.04),
            (ForceMode.VelocityChange, 1.0, 1.0),
        ],
    )
    def test_angular_mass_independent_modes_survive_partial_rotation_constraints(
        self, scene, mode, input_value, expected_velocity
    ):
        Physics.set_gravity(Vector3(0, 0, 0))
        _, light = _make_ball(scene, pos=Vector3(-2, 5, 0), mass=1.0)
        _, heavy = _make_ball(scene, pos=Vector3(2, 5, 0), mass=10.0)
        for rigidbody in (light, heavy):
            rigidbody.use_gravity = False
            rigidbody.angular_drag = 0.0
            rigidbody.max_angular_velocity = 100.0
            rigidbody.constraints = int(RigidbodyConstraints.FreezeRotationX)

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()

        torque = Vector3(0, 0, input_value)
        light.add_torque(torque, mode)
        heavy.add_torque(torque, mode)
        manager.step()

        assert tuple(light.angular_velocity) == pytest.approx((0, 0, expected_velocity), abs=1e-4)
        assert tuple(heavy.angular_velocity) == pytest.approx((0, 0, expected_velocity), abs=1e-4)

    def test_angular_acceleration_inverts_coupled_allowed_inertia_subspace(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        body = scene.create_game_object("CoupledInertiaBox")
        body.transform.position = Vector3(0, 5, 0)
        body.transform.rotation = quatf(0.38268343, 0, 0, 0.92387953)
        rigidbody = body.add_component("Rigidbody")
        rigidbody.mass = 3.0
        rigidbody.use_gravity = False
        rigidbody.angular_drag = 0.0
        rigidbody.max_angular_velocity = 100.0
        rigidbody.constraints = int(RigidbodyConstraints.FreezeRotationX)
        collider = body.add_component("BoxCollider")
        collider.size = Vector3(1, 2, 4)

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()
        rigidbody.add_torque(Vector3(0, 1, 2), ForceMode.Acceleration)
        manager.step()

        fixed_delta = manager.get_fixed_time_step()
        assert tuple(rigidbody.angular_velocity) == pytest.approx(
            (0, fixed_delta, fixed_delta * 2.0), abs=1e-4
        )

    def test_drag_uses_configured_collision_substeps(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        _, rigidbody = _make_ball(scene, pos=Vector3(0, 5, 0))
        rigidbody.use_gravity = False
        rigidbody.drag = 5.0
        rigidbody.angular_drag = 4.0
        rigidbody.max_linear_velocity = 100.0
        rigidbody.max_angular_velocity = 100.0

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()
        rigidbody.velocity = Vector3(10, 0, 0)
        rigidbody.angular_velocity = Vector3(0, 10, 0)
        manager.step()

        fixed_delta = manager.get_fixed_time_step()
        substeps = EngineConfig.get().physics_collision_steps
        expected_linear = 10.0 * (1.0 - 5.0 * fixed_delta / substeps) ** substeps
        expected_angular = 10.0 * (1.0 - 4.0 * fixed_delta / substeps) ** substeps
        assert rigidbody.velocity.x == pytest.approx(expected_linear, abs=1e-4)
        assert rigidbody.angular_velocity.y == pytest.approx(expected_angular, abs=1e-4)

    def test_velocity_limits_apply_to_direct_assignment_and_impulses(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        _, rigidbody = _make_ball(scene, pos=Vector3(0, 5, 0))
        rigidbody.use_gravity = False
        rigidbody.drag = 0.0
        rigidbody.angular_drag = 0.0
        rigidbody.max_linear_velocity = 3.0
        rigidbody.max_angular_velocity = 2.0

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()
        rigidbody.velocity = Vector3(30, 0, 0)
        rigidbody.angular_velocity = Vector3(0, 20, 0)

        assert tuple(rigidbody.velocity) == pytest.approx((3, 0, 0), abs=1e-4)
        assert tuple(rigidbody.angular_velocity) == pytest.approx((0, 2, 0), abs=1e-4)

        rigidbody.add_force(Vector3(30, 0, 0), ForceMode.Impulse)
        rigidbody.add_torque(Vector3(0, 20, 0), ForceMode.Impulse)
        manager.step()
        assert tuple(rigidbody.velocity) == pytest.approx((3, 0, 0), abs=1e-4)
        assert tuple(rigidbody.angular_velocity) == pytest.approx((0, 2, 0), abs=1e-4)


class TestContinuousCollisionDetection:
    @pytest.mark.parametrize(
        "mode,should_cross_wall",
        [
            (CollisionDetectionMode.Discrete, True),
            (CollisionDetectionMode.Continuous, False),
            (CollisionDetectionMode.ContinuousDynamic, False),
        ],
    )
    def test_fast_sphere_against_thin_static_wall(self, scene, mode, should_cross_wall):
        Physics.set_gravity(Vector3(0, 0, 0))

        wall = scene.create_game_object("ThinWall")
        wall.transform.position = Vector3(0, 0, 0)
        wall.transform.local_scale = Vector3(0.1, 10, 10)
        wall.add_component("BoxCollider")

        projectile, rigidbody = _make_ball(scene, pos=Vector3(-4, 0, 0), radius=0.25)
        rigidbody.use_gravity = False
        rigidbody.drag = 0.0
        rigidbody.max_linear_velocity = 1000.0
        rigidbody.collision_detection_mode = mode
        rigidbody.velocity = Vector3(600, 0, 0)

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()

        if should_cross_wall:
            assert projectile.transform.position.x > 1.0
        else:
            assert projectile.transform.position.x < 0.0
            assert rigidbody.velocity.x < 1.0

    @pytest.mark.parametrize(
        "mode,should_cross",
        [
            (CollisionDetectionMode.Discrete, True),
            (CollisionDetectionMode.Continuous, True),
            (CollisionDetectionMode.ContinuousDynamic, False),
        ],
    )
    def test_fast_dynamic_spheres_respect_collision_mode(self, scene, mode, should_cross):
        Physics.set_gravity(Vector3(0, 0, 0))

        pair = []
        for side, velocity in ((-1, 600.0), (1, -600.0)):
            body = scene.create_game_object(f"DynamicPair{side}")
            body.transform.position = Vector3(side * 4.0, 0, 0)
            rigidbody = body.add_component("Rigidbody")
            rigidbody.use_gravity = False
            rigidbody.drag = 0.0
            rigidbody.max_linear_velocity = 1000.0
            rigidbody.collision_detection_mode = mode
            collider = body.add_component("SphereCollider")
            collider.radius = 0.25
            rigidbody.velocity = Vector3(velocity, 0, 0)
            pair.append((body, rigidbody))

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        manager.step()
        profile = manager.get_last_frame_profile()

        if should_cross:
            assert profile["dynamic_ccd_splits"] == 0
            assert pair[0][0].transform.position.x > 1.0
            assert pair[1][0].transform.position.x < -1.0
        else:
            assert profile["dynamic_ccd_splits"] >= 1
            assert pair[0][0].transform.position.x < 0.0
            assert pair[1][0].transform.position.x > 0.0
            assert pair[0][1].velocity.x < 1.0
            assert pair[1][1].velocity.x > -1.0


# ═══════════════════════════════════════════════════════════════════════════
# Raycasting with actual scene objects
# ═══════════════════════════════════════════════════════════════════════════

class TestRaycast:
    def test_raycast_hits_ground(self, scene):
        _make_ground(scene)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        hit = Physics.raycast(Vector3(0, 50, 0), Vector3(0, -1, 0), 100.0)
        assert hit is not None
        assert 0 < hit.distance < 50.0
        assert hit.normal.y == pytest.approx(1.0, abs=0.1)

    def test_raycast_miss(self, scene):
        _make_ground(scene)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        # Ray pointing away from all objects
        hit = Physics.raycast(Vector3(0, 50, 0), Vector3(0, 1, 0), 100.0)
        assert hit is None

    def test_raycast_all_returns_multiple(self, scene):
        _make_ground(scene)
        # Create floating box collider above ground
        box = scene.create_game_object("FloatingBox")
        box.transform.position = Vector3(0, 5, 0)
        box.add_component("BoxCollider")

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        hits = Physics.raycast_all(Vector3(0, 50, 0), Vector3(0, -1, 0), 100.0)
        assert len(hits) >= 2  # at least ground + floating box
        distances = [hit.distance for hit in hits]
        assert distances == sorted(distances)

    def test_static_non_convex_mesh_returns_triangle_identity(self, scene):
        mesh_object = scene.create_primitive(PrimitiveType.Cube, "RaycastTriangleMesh")
        primitive_collider = mesh_object.get_component("BoxCollider")
        assert mesh_object.remove_component(primitive_collider) is True
        mesh_collider = mesh_object.add_component("MeshCollider")
        assert mesh_collider.convex is False

        Physics.sync_transforms()
        hit = Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10.0)

        assert hit is not None
        assert hit.collider.component_id == mesh_collider.component_id
        assert hit.body_id != 0xFFFFFFFF
        assert hit.sub_shape_id >= 0
        assert hit.triangle_index is not None
        assert 0 <= hit.triangle_index < 12


# ═══════════════════════════════════════════════════════════════════════════
# Overlap queries
# ═══════════════════════════════════════════════════════════════════════════

class TestOverlapQueries:
    def test_overlap_sphere_finds_colliders(self, scene):
        for i in range(3):
            go = scene.create_game_object(f"Obj{i}")
            go.transform.position = Vector3(float(i), 0, 0)
            go.add_component("SphereCollider")

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        result = Physics.overlap_sphere(Vector3(0, 0, 0), 50.0)
        assert len(result) >= 3

    def test_overlap_box_finds_colliders(self, scene):
        go = scene.create_game_object("TestBox")
        go.add_component("BoxCollider")

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        result = Physics.overlap_box(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert len(result) >= 1

    def test_overlap_box_honors_orientation(self, scene):
        target = scene.create_game_object("OrientedQueryTarget")
        target.transform.position = Vector3(0, 3, 0)
        target.add_component("SphereCollider")
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        extents = Vector3(4, 0.25, 0.25)
        collider_id = target.get_component("SphereCollider").component_id
        assert collider_id not in [c.component_id for c in Physics.overlap_box(Vector3(0, 0, 0), extents)]
        quarter_turn_z = quatf(0.0, 0.0, 0.70710678, 0.70710678)
        assert collider_id in [
            c.component_id for c in Physics.overlap_box(Vector3(0, 0, 0), extents, quarter_turn_z)
        ]

    def test_overlap_capsule_finds_colliders(self, scene):
        target = scene.create_game_object("CapsuleQueryTarget")
        target.transform.position = Vector3(0, 2, 0)
        collider = target.add_component("SphereCollider")
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        assert collider.component_id in [
            c.component_id
            for c in Physics.overlap_capsule(Vector3(0, -2, 0), Vector3(0, 2, 0), 0.75)
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Shape casts
# ═══════════════════════════════════════════════════════════════════════════

class TestShapeCasts:
    def test_sphere_cast_hits_ground(self, scene):
        _make_ground(scene)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        hit = Physics.sphere_cast(Vector3(0, 50, 0), 1.0, Vector3(0, -1, 0), 100.0)
        assert hit is not None
        assert hit.distance < 60

    def test_box_cast_hits_ground(self, scene):
        _make_ground(scene)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        hit = Physics.box_cast(
            Vector3(0, 50, 0), Vector3(1, 1, 1), Vector3(0, -1, 0), max_distance=100.0
        )
        assert hit is not None

    def test_raycast_flushes_moved_static_collider_without_fixed_step(self, scene):
        target = scene.create_game_object("MovedStaticQueryTarget")
        target.transform.position = Vector3(0, 0, 0)
        target.add_component("BoxCollider")

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        sm.step()

        target.transform.position = Vector3(12, 0, 0)
        hit = Physics.raycast(Vector3(12, 5, 0), Vector3(0, -1, 0), 10.0)

        assert hit is not None
        assert hit.game_object.name == "MovedStaticQueryTarget"

    def test_capsule_cast_hits_ground(self, scene):
        _make_ground(scene)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        hit = Physics.capsule_cast(
            Vector3(0, 49, 0), Vector3(0, 51, 0), 0.5, Vector3(0, -1, 0), 100.0
        )
        assert hit is not None


class TestContactCallbacks:
    def test_trigger_enter_stay_exit_lifecycle(self, scene):
        class TriggerProbe(InxComponent):
            events = None

            def awake(self):
                self.events = []

            def on_trigger_enter(self, other):
                self.events.append("enter")

            def on_trigger_stay(self, other):
                self.events.append("stay")

            def on_trigger_exit(self, other):
                self.events.append("exit")

        trigger_object = scene.create_game_object("LifecycleTrigger")
        trigger = trigger_object.add_component("BoxCollider")
        trigger.size = Vector3(4, 4, 4)
        trigger.is_trigger = True

        mover = scene.create_game_object("LifecycleMover")
        mover.transform.position = Vector3(8, 0, 0)
        mover.add_component("SphereCollider")
        rigidbody = mover.add_component("Rigidbody")
        rigidbody.is_kinematic = True
        probe = mover.add_component(TriggerProbe)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        sm.step()

        mover.transform.position = Vector3(0, 0, 0)
        Physics.sync_transforms()
        sm.step()
        assert "enter" in probe.events

        sm.step()
        assert "stay" in probe.events

        mover.transform.position = Vector3(8, 0, 0)
        Physics.sync_transforms()
        sm.step()
        assert "exit" in probe.events

# ═══════════════════════════════════════════════════════════════════════════
# Layer collision filtering
# ═══════════════════════════════════════════════════════════════════════════

class TestLayerCollision:
    def test_ignore_layer_collision_round_trip(self, scene):
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        Physics.ignore_layer_collision(10, 11, True)
        assert Physics.get_ignore_layer_collision(10, 11) is True
        Physics.ignore_layer_collision(10, 11, False)
        assert Physics.get_ignore_layer_collision(10, 11) is False


class TestColliderCollisionIgnore:
    def test_exact_compound_member_can_be_ignored_and_restored(self, scene):
        compound = scene.create_game_object("CompoundGround")
        box = compound.add_component("BoxCollider")
        box.size = Vector3(2, 1, 2)
        sphere = compound.add_component("SphereCollider")
        sphere.center = Vector3(5, 0, 0)
        sphere.radius = 1.0

        box_ball, _ = _make_ball(scene, pos=Vector3(0, 3, 0))
        sphere_ball, sphere_body = _make_ball(scene, pos=Vector3(5, 3, 0))
        sphere_ball_collider = sphere_ball.get_component("SphereCollider")

        Physics.ignore_collision(sphere, sphere_ball_collider)
        assert Physics.get_ignore_collision(sphere_ball_collider, sphere) is True

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(120)

        assert box_ball.transform.position.y > 0.8
        assert sphere_ball.transform.position.y < -3.0

        sphere_body.position = Vector3(5, 3, 0)
        sphere_body.velocity = Vector3(0, 0, 0)
        Physics.ignore_collision(sphere_ball_collider, sphere, False)
        assert Physics.get_ignore_collision(sphere, sphere_ball_collider) is False
        _step_frames(120)
        assert sphere_ball.transform.position.y > 1.3

    def test_joint_and_game_ignore_own_their_policies_independently(self, scene):
        class CollisionProbe(InxComponent):
            events = None

            def awake(self):
                self.events = []

            def on_collision_enter(self, collision):
                self.events.append(collision)

        support = scene.create_game_object("IgnoreSupport")
        support.transform.position = Vector3(0, 3, 0)
        support_body = support.add_component("Rigidbody")
        support_body.is_kinematic = True
        support_collider = support.add_component("BoxCollider")

        body = scene.create_game_object("IgnoreDoor")
        body.transform.position = Vector3(0.75, 3, 0)
        rigidbody = body.add_component("Rigidbody")
        rigidbody.use_gravity = False
        body_collider = body.add_component("BoxCollider")
        hinge = body.add_component("HingeJoint")
        hinge.anchor = Vector3(-0.375, 0, 0)
        hinge.axis = Vector3(0, 0, 1)
        hinge.connected_body = support_body
        probe = body.add_component(CollisionProbe)

        Physics.ignore_collision(support_collider, body_collider)
        # The public policy is idempotent; a repeated authoring write cannot
        # create hidden reference-count ownership.
        Physics.ignore_collision(body_collider, support_collider)

        manager = SceneManager.instance()
        manager.play()
        manager.pause()
        _step_frames(3)
        assert probe.events == []

        hinge.enable_collision = True
        _step_frames(3)
        assert probe.events == []
        assert Physics.get_ignore_collision(support_collider, body_collider) is True

        Physics.ignore_collision(support_collider, body_collider, False)
        _step_frames(3)
        assert len(probe.events) == 1

        # Releasing an already released public policy remains an idempotent
        # state assignment rather than an ownership error.
        Physics.ignore_collision(support_collider, body_collider, False)

    def test_destroyed_collider_does_not_transfer_policy_to_replacement(self, scene):
        first = scene.create_game_object("DestroyIgnored")
        first_collider = first.add_component("BoxCollider")
        second = scene.create_game_object("IgnorePeer")
        second_collider = second.add_component("BoxCollider")
        Physics.ignore_collision(first_collider, second_collider)

        assert first.remove_component(first_collider) is True
        replacement = first.add_component("BoxCollider")
        assert Physics.get_ignore_collision(replacement, second_collider) is False


class TestRaycastBatch:
    @staticmethod
    def _output(capacity, fill=0):
        import numpy as np

        return {
            "hit": np.full(capacity, fill, dtype=np.uint8),
            "point": np.full((capacity, 3), fill, dtype=np.float32),
            "normal": np.full((capacity, 3), fill, dtype=np.float32),
            "distance": np.full(capacity, fill, dtype=np.float32),
            "body_id": np.full(capacity, fill, dtype=np.uint32),
            "sub_shape_id": np.full(capacity, fill, dtype=np.uint32),
            "triangle_index": np.full(capacity, fill, dtype=np.uint32),
            "collider_id": np.full(capacity, fill, dtype=np.uint64),
            "game_object_id": np.full(capacity, fill, dtype=np.uint64),
        }

    def test_reuses_numeric_outputs_and_writes_only_valid_prefix(self, scene):
        import numpy as np

        target = scene.create_game_object("BatchRayTarget")
        target_collider = target.add_component("BoxCollider")
        target_collider.size = Vector3(2, 1, 2)
        Physics.sync_transforms()

        origins = np.array([[0, 5, 0], [5, 5, 0]], dtype=np.float32)
        directions = np.array([[0, -1, 0], [0, -1, 0]], dtype=np.float32)
        output = self._output(4, fill=77)
        identities = {name: id(value) for name, value in output.items()}

        result = PublicPhysics.raycast_batch(origins, directions, output, max_distance=10)

        assert result is output
        assert {name: id(value) for name, value in output.items() if isinstance(value, np.ndarray)} == identities
        assert output["query_generation"] == int(PublicPhysics.query_generation)
        np.testing.assert_array_equal(output["hit"][:2], [1, 0])
        assert output["distance"][0] == pytest.approx(4.5, abs=1e-4)
        assert math.isinf(float(output["distance"][1]))
        np.testing.assert_allclose(output["point"][0], [0, 0.5, 0], atol=1e-4)
        np.testing.assert_allclose(output["normal"][0], [0, 1, 0], atol=1e-4)
        assert output["collider_id"][0] == target_collider.component_id
        assert output["game_object_id"][0] == target.id
        assert output["triangle_index"][0] == np.iinfo(np.uint32).max
        assert output["collider_id"][1] == 0
        assert output["game_object_id"][1] == 0
        assert output["triangle_index"][1] == np.iinfo(np.uint32).max
        for name, value in output.items():
            if isinstance(value, np.ndarray):
                assert np.all(value[2:] == 77)

    def test_rejects_insufficient_capacity_before_writing(self, scene):
        import numpy as np

        origins = np.zeros((2, 3), dtype=np.float32)
        directions = np.array([[0, -1, 0], [0, -1, 0]], dtype=np.float32)
        output = self._output(2, fill=91)
        output["normal"] = np.full((1, 3), 91, dtype=np.float32)

        with pytest.raises(ValueError, match="normal.*insufficient capacity"):
            PublicPhysics.raycast_batch(origins, directions, output)
        assert np.all(output["hit"] == 91)

    def test_python_allocation_does_not_scale_with_hit_count(self, scene):
        import gc
        import tracemalloc
        import numpy as np

        target = scene.create_game_object("BatchAllocationTarget")
        collider = target.add_component("BoxCollider")
        collider.size = Vector3(100, 1, 100)
        Physics.sync_transforms()

        count = 256
        origins = np.zeros((count, 3), dtype=np.float32)
        origins[:, 1] = 5
        directions = np.zeros((count, 3), dtype=np.float32)
        directions[:, 1] = -1
        output = self._output(count)
        PublicPhysics.raycast_batch(origins, directions, output)

        gc.collect()
        tracemalloc.start()
        PublicPhysics.raycast_batch(origins, directions, output)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"INFERNUX_RAYCAST_BATCH_PYTHON_PEAK_BYTES={peak}")

        # A result object per ray would exceed this bound by a wide margin.
        # The fixed pybind call frame and returned borrowed dict stay constant.
        assert peak < 32 * 1024

    def test_query_generation_changes_only_when_published_physics_state_changes(self, scene):
        import numpy as np

        target = scene.create_game_object("QueryGenerationTarget")
        target.add_component("BoxCollider")
        Physics.sync_transforms()
        initial = int(PublicPhysics.query_generation)

        # Repeating a query boundary with no authored change must keep the
        # same published-world token; consumers can safely retain hit buffers.
        Physics.sync_transforms()
        assert int(PublicPhysics.query_generation) == initial

        target.transform.position = Vector3(0, 2, 0)
        Physics.sync_transforms()
        moved = int(PublicPhysics.query_generation)
        assert moved > initial

        origins = np.array([[0, 5, 0]], dtype=np.float32)
        directions = np.array([[0, -1, 0]], dtype=np.float32)
        output = self._output(1)
        result = PublicPhysics.raycast_batch(origins, directions, output)
        assert int(PublicPhysics.query_generation) == moved
        assert result["query_generation"] == moved
        assert output["hit"][0] == 1

class TestIncrementalTransformSync:
    def test_mesh_cooking_cache_reuses_identical_geometry(self, scene):
        NativeMeshCollider.clear_cooking_cache()
        first = scene.create_primitive(PrimitiveType.Cube, "CachedMeshA")
        second = scene.create_primitive(PrimitiveType.Cube, "CachedMeshB")
        first_mesh = first.add_component("MeshCollider")
        second_mesh = second.add_component("MeshCollider")

        Physics.sync_transforms()

        stats = NativeMeshCollider.get_cooking_cache_stats()
        assert stats["misses"] == 1
        assert stats["hits"] >= 1
        assert stats["async_submissions"] == 1
        assert stats["pending"] == 0
        assert first_mesh.is_cooking is False
        assert second_mesh.is_cooking is False
        assert first_mesh.shape_error == ""
        assert second_mesh.shape_error == ""

    def test_mesh_geometry_change_invalidates_pending_cook(self, scene):
        NativeMeshCollider.clear_cooking_cache()
        game_object = scene.create_primitive(PrimitiveType.Cube, "ChangingMesh")
        renderer = game_object.get_component("MeshRenderer")
        mesh = game_object.add_component("MeshCollider")
        Physics.sync_transforms()

        renderer.set_primitive_mesh(PrimitiveType.Sphere)
        assert mesh.is_cooking is True
        assert NativeMeshCollider.get_cooking_cache_stats()["pending"] == 1

        # Returning to the cached cube invalidates this collider's sphere
        # revision without cancelling the shared immutable worker payload.
        renderer.set_primitive_mesh(PrimitiveType.Cube)
        assert mesh.is_cooking is False
        Physics.sync_transforms()

        stats = NativeMeshCollider.get_cooking_cache_stats()
        assert stats["pending"] == 0
        assert stats["async_submissions"] == 2
        assert mesh.is_cooking is False
        assert mesh.shape_error == ""


    def test_pose_readback_only_visits_active_bodies(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        rigidbodies = [
            _make_ball(scene, pos=Vector3(index * 3.0, 5, 0), radius=0.5)[1]
            for index in range(24)
        ]
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        sm.step()

        for rigidbody in rigidbodies:
            rigidbody.sleep()
        sm.step()
        assert sm.get_last_rigidbody_sync_candidate_count() == 0
        assert sm.get_last_interpolation_candidate_count() == 0

        rigidbodies[7].wake_up()
        sm.step()
        assert sm.get_last_rigidbody_sync_candidate_count() == 1
        assert sm.get_last_interpolation_candidate_count() == 1

    def test_only_changed_collider_is_considered(self, scene):
        objects = []
        for index in range(64):
            game_object = scene.create_game_object(f"StaticCollider{index}")
            game_object.transform.position = Vector3(float(index * 2), 0, 0)
            game_object.add_component("BoxCollider")
            objects.append(game_object)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        Physics.sync_transforms()

        sm.step()
        assert sm.get_last_collider_sync_candidate_count() == 0

        objects[17].transform.position = Vector3(34, 3, 0)
        sm.step()
        # A static-only world skips simulation. Its query boundary consumes
        # dirty poses on demand; a fixed step need not synchronize them.
        hit = Physics.raycast(Vector3(34, 6, 0), Vector3(0, -1, 0), 10)
        assert hit is not None
        assert hit.point.y == pytest.approx(3.5)
        assert sm.get_last_collider_sync_candidate_count() == 1

    def test_dynamic_scale_change_rebuilds_shape_through_dirty_actor(self, scene):
        Physics.set_gravity(Vector3(0, 0, 0))
        game_object, rigidbody = _make_ball(scene, pos=Vector3(0, 5, 0), radius=0.5)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        sm.step()

        collider_id = game_object.get_component("SphereCollider").component_id
        assert collider_id in {
            collider.component_id
            for collider in Physics.overlap_sphere(Vector3(0, 5, 0), 0.1)
        }
        initial = {
            collider.component_id
            for collider in Physics.overlap_sphere(Vector3(1.5, 5, 0), 0.1)
        }
        assert collider_id not in initial
        game_object.transform.local_scale = Vector3(4, 4, 4)
        sm.step()
        scaled = {
            collider.component_id
            for collider in Physics.overlap_sphere(Vector3(1.5, 5, 0), 0.1)
        }
        assert collider_id in scaled


class TestPrimitiveColliderAlignment:
    def test_cylinder_uses_y_as_its_height_axis(self, scene):
        cylinder = scene.create_primitive(PrimitiveType.Cylinder, "VerticalCylinder")
        cylinder.transform.position = Vector3(8, 1, 0)
        collider = cylinder.get_component("CylinderCollider")
        collider.height = 3.0
        collider.radius = 0.5

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        Physics.sync_transforms()
        sm.step()

        top_hit = Physics.raycast(Vector3(8, 4, 0), Vector3(0, -1, 0), 4.0)
        assert top_hit is not None
        assert top_hit.game_object.name == "VerticalCylinder"
        assert top_hit.distance == pytest.approx(1.5, abs=0.02)

        side_hit = Physics.raycast(Vector3(6, 1, 0), Vector3(1, 0, 0), 4.0)
        assert side_hit is not None
        assert side_hit.game_object.name == "VerticalCylinder"
        assert side_hit.distance == pytest.approx(1.5, abs=0.02)

    def test_sphere_and_capsule_native_shapes_match_builtin_mesh_dimensions(self, scene):
        sphere = scene.create_primitive(PrimitiveType.Sphere, "AlignedSphere")
        assert sphere.get_component("SphereCollider") is not None

        capsule = scene.create_primitive(PrimitiveType.Capsule, "AlignedCapsule")
        capsule.transform.position = Vector3(3, 0, 0)
        assert capsule.get_component("CapsuleCollider") is not None

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        Physics.sync_transforms()
        sm.step()

        sphere_hit = Physics.raycast(Vector3(-2, 0, 0), Vector3(1, 0, 0), 4.0)
        assert sphere_hit is not None
        assert sphere_hit.game_object.name == "AlignedSphere"
        assert sphere_hit.distance == pytest.approx(1.5, abs=0.02)

        capsule_top_hit = Physics.raycast(Vector3(3, 3, 0), Vector3(0, -1, 0), 4.0)
        assert capsule_top_hit is not None
        assert capsule_top_hit.game_object.name == "AlignedCapsule"
        assert capsule_top_hit.distance == pytest.approx(2.0, abs=0.02)

        capsule_side_hit = Physics.raycast(Vector3(1, 0, 0), Vector3(1, 0, 0), 4.0)
        assert capsule_side_hit is not None
        assert capsule_side_hit.game_object.name == "AlignedCapsule"
        assert capsule_side_hit.distance == pytest.approx(1.5, abs=0.02)


class TestCompoundCollider:
    def test_detached_collider_does_not_allocate_actor(self, scene):
        initial_count = Physics.get_actor_count()
        detached = NativeBoxCollider()
        assert detached.serialize()
        assert Physics.get_actor_count() == initial_count

    def test_compound_uses_one_generational_actor_slot(self, scene):
        initial_count = Physics.get_actor_count()
        compound = scene.create_game_object("ActorSlot")
        primary = compound.add_component("BoxCollider")
        secondary = compound.add_component("SphereCollider")

        assert Physics.get_actor_count() == initial_count + 1
        assert compound.remove_component(secondary) is True
        assert Physics.get_actor_count() == initial_count + 1
        assert compound.remove_component(primary) is True
        assert Physics.get_actor_count() == initial_count

    def test_non_primary_collider_geometry_rebuilds_shared_shape(self, scene):
        game_object = scene.create_game_object("Compound")
        game_object.add_component("BoxCollider")
        sphere = game_object.add_component("SphereCollider")
        sphere.radius = 0.5
        sphere.center = Vector3(5, 0, 0)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        sm.step()

        initial = Physics.raycast(Vector3(5, 5, 0), Vector3(0, -1, 0), 10.0)
        assert initial is not None
        assert initial.collider.component_id == sphere.component_id

        sphere.center = Vector3(10, 0, 0)
        moved = Physics.raycast(Vector3(10, 5, 0), Vector3(0, -1, 0), 10.0)
        assert moved is not None
        assert moved.collider.component_id == sphere.component_id

    def test_mixed_trigger_query_filters_per_subshape(self, scene):
        game_object = scene.create_game_object("MixedCompound")
        solid = game_object.add_component("BoxCollider")
        trigger = game_object.add_component("SphereCollider")
        trigger.center = Vector3(8, 0, 0)
        trigger.radius = 1.0
        trigger.is_trigger = True

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        sm.step()

        solid_hit = Physics.raycast(
            Vector3(0, 5, 0), Vector3(0, -1, 0), 10.0,
            query_triggers=False,
        )
        assert solid_hit is not None
        assert solid_hit.collider.component_id == solid.component_id

        trigger_hit = Physics.raycast(
            Vector3(8, 5, 0), Vector3(0, -1, 0), 10.0,
            query_triggers=False,
        )
        assert trigger_hit is None

    def test_mixed_trigger_preserves_solid_response(self, scene):
        compound = scene.create_game_object("MixedResponse")
        solid = compound.add_component("BoxCollider")
        solid.size = Vector3(4, 1, 4)
        trigger = compound.add_component("SphereCollider")
        trigger.center = Vector3(8, 2, 0)
        trigger.radius = 2.0
        trigger.is_trigger = True

        landing_ball, _ = _make_ball(scene, pos=Vector3(0, 5, 0))
        passing_ball, _ = _make_ball(scene, pos=Vector3(8, 6, 0))

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(120)

        assert landing_ball.transform.position.y > 0.0
        assert passing_ball.transform.position.y < 0.0

    def test_reenable_non_primary_collider_takes_actor_ownership(self, scene):
        compound = scene.create_game_object("ReenabledCompound")
        primary = compound.add_component("BoxCollider")
        secondary = compound.add_component("SphereCollider")

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        sm.step()

        primary.enabled = False
        secondary.enabled = False
        compound.transform.position = Vector3(12, 0, 0)
        secondary.enabled = True
        sm.step()

        hit = Physics.raycast(Vector3(12, 5, 0), Vector3(0, -1, 0), 10.0)
        assert hit is not None
        assert hit.collider.component_id == secondary.component_id

    def test_removing_secondary_collider_keeps_actor_body_alive(self, scene):
        compound = scene.create_game_object("RemovedCompoundChild")
        primary = compound.add_component("BoxCollider")
        secondary = compound.add_component("SphereCollider")
        secondary.center = Vector3(6, 0, 0)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        sm.step()

        assert compound.remove_component(secondary) is True
        sm.step()

        primary_hit = Physics.raycast(Vector3(0, 5, 0), Vector3(0, -1, 0), 10.0)
        removed_hit = Physics.raycast(Vector3(6, 5, 0), Vector3(0, -1, 0), 10.0)
        assert primary_hit is not None
        assert primary_hit.collider.component_id == primary.component_id
        assert removed_hit is None
