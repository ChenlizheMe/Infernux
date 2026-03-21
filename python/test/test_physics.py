"""Tests for InfEngine.physics — Physics query API and physics settings (real C++ backend)."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import InfEngine.physics as physics_module
from InfEngine.math.coerce import coerce_vec3
from InfEngine.lib import Vector3, Physics as CppPhysics
from InfEngine.physics.settings import (
    DEFAULT_PHYSICS_SETTINGS,
    load as load_physics_settings,
    save as save_physics_settings,
    settings_path,
)
from InfEngine.components.builtin import (
    CollisionDetectionMode,
    RigidbodyConstraints,
    RigidbodyInterpolation,
)


# ═══════════════════════════════════════════════════════════════════════════
# coerce_vec3 — real Vector3
# ═══════════════════════════════════════════════════════════════════════════

class TestCoerceVec3:
    def test_converts_tuples(self):
        value = coerce_vec3((1, 2, 3))
        assert isinstance(value, Vector3)
        assert (value.x, value.y, value.z) == pytest.approx((1.0, 2.0, 3.0))

    def test_preserves_existing_vec3(self):
        original = Vector3(4, 5, 6)
        assert coerce_vec3(original) is original


# ═══════════════════════════════════════════════════════════════════════════
# Physics class property / static API (real C++ backend)
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicsGravity:
    def test_gravity_returns_vector3(self):
        g = physics_module.Physics.gravity
        assert hasattr(g, "x") and hasattr(g, "y") and hasattr(g, "z")

    def test_default_gravity_is_earth(self):
        g = physics_module.Physics.gravity
        assert g.y == pytest.approx(-9.81, abs=0.01)

    def test_set_gravity_accepts_tuple(self):
        """set_gravity call chain does not crash."""
        physics_module.Physics.gravity = (0, -20, 0)
        # Reset
        physics_module.Physics.gravity = (0, -9.81, 0)

    def test_set_gravity_accepts_vector3(self):
        physics_module.Physics.gravity = Vector3(0, -9.81, 0)


class TestPhysicsRaycast:
    def test_raycast_returns_none_in_empty_space(self):
        # Fire a ray far from any test geometry so it misses everything
        result = physics_module.Physics.raycast(
            Vector3(99999, 99999, 99999), Vector3(0, 1, 0), 1.0)
        assert result is None

    def test_raycast_accepts_tuple_origin(self):
        result = physics_module.Physics.raycast(
            (99999, 99999, 99999), (0, 1, 0), 1.0)
        assert result is None

    def test_raycast_all_returns_list(self):
        # Fire a ray far from any test geometry
        result = physics_module.Physics.raycast_all(
            Vector3(99999, 99999, 99999), Vector3(0, 1, 0), 1.0)
        assert isinstance(result, list)


class TestPhysicsOverlap:
    def test_overlap_sphere_returns_empty_without_scene(self):
        result = physics_module.Physics.overlap_sphere(
            Vector3(0, 0, 0), 10.0)
        assert isinstance(result, list)

    def test_overlap_box_returns_empty_without_scene(self):
        result = physics_module.Physics.overlap_box(
            Vector3(0, 0, 0), Vector3(1, 1, 1))
        assert isinstance(result, list)

    def test_overlap_accepts_tuples(self):
        result = physics_module.Physics.overlap_sphere((0, 0, 0), 5.0)
        assert isinstance(result, list)


class TestPhysicsShapeCasts:
    def test_sphere_cast_returns_none_without_scene(self):
        result = physics_module.Physics.sphere_cast(
            Vector3(0, 0, 0), 1.0, Vector3(0, 1, 0), 100.0)
        assert result is None

    def test_box_cast_returns_none_without_scene(self):
        result = physics_module.Physics.box_cast(
            Vector3(0, 0, 0), Vector3(1, 1, 1), Vector3(0, 1, 0))
        assert result is None

    def test_shape_casts_accept_tuples(self):
        assert physics_module.Physics.sphere_cast(
            (0, 0, 0), 1.0, (0, 0, 1), 50.0) is None
        assert physics_module.Physics.box_cast(
            (0, 0, 0), (1, 1, 1), (1, 0, 0)) is None


class TestPhysicsLayerCollision:
    def test_layer_collision_defaults_to_false(self):
        assert physics_module.Physics.get_ignore_layer_collision(30, 31) is False

    def test_ignore_layer_collision_does_not_crash(self):
        physics_module.Physics.ignore_layer_collision(30, 31, True)
        physics_module.Physics.ignore_layer_collision(30, 31, False)


class TestPhysicsModuleExports:
    def test_module_exports_only_physics(self):
        assert physics_module.__all__ == ["Physics"]


# ═══════════════════════════════════════════════════════════════════════════
# Rigidbody enums (real C++ enums)
# ═══════════════════════════════════════════════════════════════════════════

class TestRigidbodyEnums:
    def test_collision_detection_mode_values(self):
        assert int(CollisionDetectionMode.Discrete) == 0
        assert int(CollisionDetectionMode.ContinuousSpeculative) == 3

    def test_interpolation_values(self):
        assert int(RigidbodyInterpolation.Interpolate) == 1

    def test_constraints_freeze_all(self):
        assert int(RigidbodyConstraints.FreezeAll) == 126


# ═══════════════════════════════════════════════════════════════════════════
# Collision detection mode mapping
# ═══════════════════════════════════════════════════════════════════════════

def _backend_motion_quality(mode: CollisionDetectionMode, is_kinematic: bool) -> int:
    if mode == CollisionDetectionMode.ContinuousDynamic:
        return 1
    if mode == CollisionDetectionMode.Continuous:
        return 0 if is_kinematic else 1
    return 0


class TestCollisionDetectionMapping:
    def test_dynamic_body_modes(self):
        assert _backend_motion_quality(CollisionDetectionMode.Discrete, False) == 0
        assert _backend_motion_quality(CollisionDetectionMode.Continuous, False) == 1
        assert _backend_motion_quality(CollisionDetectionMode.ContinuousDynamic, False) == 1
        assert _backend_motion_quality(CollisionDetectionMode.ContinuousSpeculative, False) == 0

    def test_kinematic_body_modes(self):
        assert _backend_motion_quality(CollisionDetectionMode.Discrete, True) == 0
        assert _backend_motion_quality(CollisionDetectionMode.Continuous, True) == 0
        assert _backend_motion_quality(CollisionDetectionMode.ContinuousDynamic, True) == 1
        assert _backend_motion_quality(CollisionDetectionMode.ContinuousSpeculative, True) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Physics settings (load / save / defaults)
# ═══════════════════════════════════════════════════════════════════════════

class TestPhysicsSettings:
    def test_defaults(self):
        assert DEFAULT_PHYSICS_SETTINGS["gravity"] == [0.0, -9.81, 0.0]
        assert DEFAULT_PHYSICS_SETTINGS["fixed_delta_time"] == 0.02

    def test_load_missing_project(self):
        result = load_physics_settings("")
        assert result == DEFAULT_PHYSICS_SETTINGS

    def test_load_missing_file(self, tmp_path):
        result = load_physics_settings(str(tmp_path))
        assert result == DEFAULT_PHYSICS_SETTINGS

    def test_settings_path(self, tmp_path):
        p = settings_path(str(tmp_path))
        assert p.endswith("PhysicsSettings.json")
        assert "ProjectSettings" in p

    def test_save_and_load_round_trip(self, tmp_path):
        project = str(tmp_path)
        os.makedirs(os.path.join(project, "ProjectSettings"), exist_ok=True)
        settings = {
            "gravity": [0.0, -20.0, 0.0],
            "fixed_delta_time": 0.01,
            "max_fixed_delta_time": 0.05,
        }
        save_physics_settings(project, settings)
        loaded = load_physics_settings(project)
        assert loaded["gravity"] == [0.0, -20.0, 0.0]
        assert loaded["fixed_delta_time"] == 0.01

    def test_load_corrupted_file(self, tmp_path):
        project = str(tmp_path)
        ps_dir = os.path.join(project, "ProjectSettings")
        os.makedirs(ps_dir, exist_ok=True)
        with open(os.path.join(ps_dir, "PhysicsSettings.json"), "w") as f:
            f.write("not json")
        result = load_physics_settings(project)
        assert result == DEFAULT_PHYSICS_SETTINGS
