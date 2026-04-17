"""
Built-in component wrappers — Python InxComponent facades for C++ components.

Provides ``Light``, ``MeshRenderer``, ``Camera``, ``BoxCollider``,
``SphereCollider``, ``CapsuleCollider``, and ``MeshCollider`` as first-class
InxComponent subclasses.  All state lives in the C++ component; CppProperty
descriptors delegate reads / writes transparently.

Usage::

    from Infernux.components.builtin import Light, MeshRenderer, Camera
    from Infernux.components.builtin import BoxCollider, SphereCollider, CapsuleCollider, MeshCollider
    from Infernux.components import InxComponent

    class MyScript(InxComponent):
        def start(self):
            light = self.game_object.get_component(Light)
            light.intensity = 2.0
"""

from .light import Light
from .mesh_renderer import MeshRenderer
from .camera import Camera
from .collider import Collider
from .box_collider import BoxCollider
from .sphere_collider import SphereCollider
from .capsule_collider import CapsuleCollider
from .mesh_collider import MeshCollider
from .rigidbody import (
    Rigidbody,
    RigidbodyConstraints,
    CollisionDetectionMode,
    RigidbodyInterpolation,
)
from .audio_source import AudioSource
from .audio_listener import AudioListener
from .sprite_renderer import SpriteRenderer

__all__ = [
    "Light",
    "MeshRenderer",
    "Camera",
    "Collider",
    "BoxCollider",
    "SphereCollider",
    "CapsuleCollider",
    "MeshCollider",
    "Rigidbody",
    "RigidbodyConstraints",
    "CollisionDetectionMode",
    "RigidbodyInterpolation",
    "AudioSource",
    "AudioListener",
    "SpriteRenderer",
]
