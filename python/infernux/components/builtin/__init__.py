"""
Built-in component wrappers — Python InxComponent facades for C++ components.

Provides ``Light``, ``MeshRenderer``, ``Camera``, ``BoxCollider``,
``SphereCollider``, ``CapsuleCollider``, ``CylinderCollider``, and
``MeshCollider`` as first-class
InxComponent subclasses.  All state lives in the C++ component; CppProperty
descriptors delegate reads / writes transparently.

Usage::

    from infernux.components.builtin import Light, MeshRenderer, Camera
    from infernux.components.builtin import (
        BoxCollider,
        SphereCollider,
        CapsuleCollider,
        CylinderCollider,
        MeshCollider,
    )
    from infernux.components import InxComponent

    class MyScript(InxComponent):
        def start(self):
            light = self.game_object.get_component(Light)
            light.intensity = 2.0
"""

from .light import Light
from .mesh_renderer import MeshRenderer
from .line_renderer import LineRenderer
from .trail_renderer import TrailRenderer
from .skinned_mesh_renderer import SkinnedMeshRenderer
from .skeleton_instance import SkeletonInstance
from .camera import Camera
from .collider import Collider, PhysicsMaterialCombine
from .box_collider import BoxCollider
from .sphere_collider import SphereCollider
from .capsule_collider import CapsuleCollider
from .cylinder_collider import CylinderCollider
from .mesh_collider import MeshCollider
from .rigidbody import (
    Rigidbody,
    RigidbodyConstraints,
    CollisionDetectionMode,
    RigidbodyInterpolation,
)
from .hinge_joint import HingeJoint
from .six_dof_joint import SixDOFJoint
from infernux.lib import JointMotion
from .swing_twist_joint import SwingTwistJoint
from infernux.lib import JointMotorMode, JointSwingType
from .fixed_joint import FixedJoint
from .ragdoll import Ragdoll
from .distance_joint import DistanceJoint
from .slider_joint import SliderJoint
from .audio_source import AudioSource
from .audio_listener import AudioListener
from .sprite_renderer import SpriteRenderer

__all__ = [
    "Light",
    "MeshRenderer",
    "LineRenderer",
    "TrailRenderer",
    "SkinnedMeshRenderer",
    "SkeletonInstance",
    "Camera",
    "Collider",
    "PhysicsMaterialCombine",
    "BoxCollider",
    "SphereCollider",
    "CapsuleCollider",
    "CylinderCollider",
    "MeshCollider",
    "Rigidbody",
    "RigidbodyConstraints",
    "CollisionDetectionMode",
    "RigidbodyInterpolation",
    "HingeJoint",
    "SixDOFJoint",
    "JointMotion",
    "SwingTwistJoint",
    "JointMotorMode",
    "JointSwingType",
    "FixedJoint",
    "Ragdoll",
    "DistanceJoint",
    "SliderJoint",
    "AudioSource",
    "AudioListener",
    "SpriteRenderer",
]
