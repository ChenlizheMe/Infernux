"""Public Python package for the Infernux engine."""

# Windows can resolve an incorrectly cased package directory. Fail before any
# native types or registries are loaded instead of creating a second engine.
if __name__ != "infernux":
    raise ImportError("The engine package is named 'infernux'; use lowercase imports.")

import importlib

from infernux.version import ENGINE_VERSION

__version__ = ENGINE_VERSION

# ── Runtime API (used by game scripts) ─────────────────────────────
from infernux.engine import release_engine, run_headless, Engine, LogLevel
from infernux.application import Application
from infernux.screen import Insets, Rect, Screen
from infernux.acceptance import RuntimeAcceptance, RuntimeAcceptanceManifest, RuntimeAcceptanceTest
from infernux.math import Vector2, Vector3, vec4f, quatf, vector2, vector3, vector4, quaternion
from infernux import components as _components_module
from infernux.components import serialized_field
from infernux.components import *
from infernux import core
from infernux.core import *
from infernux.lib import (
    AudioEngine,
    Component,
    GameObject,
    LineAlignment,
    LineColorKey,
    LineCurveWrapMode,
    LineGradientMode,
    LineTextureMode,
    LineWidthKey,
    PrimitiveType,
    Space,
    Transform,
)
from infernux.debug import Debug, debug, log, log_warning, log_error, log_exception
from infernux import scene
from infernux.scene import GameObjectQuery, LayerMask, SceneManager
from infernux.timing import Time
from infernux.mathf import Mathf
from infernux.ui import (
    UICanvas, UIFrame, UIGroup, UIProgressBar, UISlider, UIText, UIImage,
    UIRawImage, UIButton, UIEvent, UIEvent1,
)
from infernux.coroutine import (
    Coroutine,
    WaitForSeconds,
    WaitForSecondsRealtime,
    WaitForEndOfFrame,
    WaitForFrames,
    WaitForFixedUpdate,
    WaitUntil,
    WaitWhile,
)
from infernux.batch import batch_read, batch_write, create_batch_handle
from infernux.instantiate import Instantiate, Destroy
from infernux.lifecycle import InxPreload, PreloadContext


def __getattr__(name: str):
    """Lazily expose optional JIT helpers without loading them at startup.

    This keeps Numba out of ordinary star-import paths while still supporting:

        from infernux import compute
        from infernux import jit
    """
    if name == "RenderStack":
        from infernux.renderstack import RenderStack
        globals()[name] = RenderStack
        return RenderStack
    if name in {"jit", "compute"}:
        return importlib.import_module(f"infernux.{name}")
    if name in {"buffer", "Buffer"}:
        compute_module = importlib.import_module("infernux.compute")
        value = getattr(compute_module, name)
        globals()[name] = value
        return value
    if name in {
        "warmup",
        "JIT_AVAILABLE",
    }:
        jit_module = importlib.import_module("infernux.jit")
        return getattr(jit_module, name)
    if name in {
        "components",
        "editor",
        "input",
        "lifecycle",
        "physics",
        "rendergraph",
        "renderstack",
        "resources",
        "ui",
    }:
        module = importlib.import_module(f"infernux.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'infernux' has no attribute {name!r}")


# ── Public game-scripting API surface ──────────────────────────────
# Curated list: only symbols commonly needed in game scripts.
# Internal / advanced helpers stay accessible via their submodules
# (e.g. ``from infernux.debug import debug``).
__all__ = [
    "__version__",
    "editor",
    "RenderStack",
    # Engine
    "Engine",
    "Application",
    "Screen",
    "Rect",
    "Insets",
    "RuntimeAcceptance",
    "RuntimeAcceptanceManifest",
    "RuntimeAcceptanceTest",
    "LogLevel",
    "release_engine",
    "run_headless",
    # Math
    "Vector2",
    "Vector3",
    "vec4f",
    "quatf",
    "vector2",
    "vector3",
    "vector4",
    "quaternion",
    # Game Objects
    "GameObject",
    "Transform",
    "Component",
    "Space",
    "PrimitiveType",
    "LineAlignment",
    "LineColorKey",
    "LineCurveWrapMode",
    "LineGradientMode",
    "LineTextureMode",
    "LineWidthKey",
    # Components — user-facing
    "InxComponent",
    "serialized_field",
    "int_field",
    "list_field",
    "component_field",
    "component_list_field",
    "hide_field",
    "InspectorSpace",
    "FieldType",
    "Color",
    "AnimationCurve",
    "Keyframe",
    "Gradient",
    "GradientKey",
    "GameObjectRef",
    "MaterialRef",
    "ComponentRef",
    "PrefabRef",
    "SerializableObject",
    "DataAsset",
    # Builtin components
    "Light",
    "MeshRenderer",
    "LineRenderer",
    "SkinnedMeshRenderer",
    "Camera",
    "Collider",
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
    "SliderJoint",
    "AudioSource",
    "AudioListener",
    "AudioEngine",
    "SpriteRenderer",
    "SpiritAnimator",
    "SkeletalAnimator",
    "RuntimeAcceptanceRunner",
    # UI components
    "UICanvas",
    "UIFrame",
    "UIGroup",
    "UIProgressBar",
    "UISlider",
    "UIText",
    "UIImage",
    "UIRawImage",
    "UIButton",
    "UIEvent",
    "UIEvent1",
    # Explicit early-import lifecycle used by project and plugin scripts
    "InxPreload",
    "PreloadContext",
    # Decorators
    "require_component",
    "disallow_multiple",
    "execute_in_edit_mode",
    "add_component_menu",
    "icon",
    "help_url",
    "RequireComponent",
    "DisallowMultipleComponent",
    "ExecuteInEditMode",
    "AddComponentMenu",
    "HelpURL",
    "Icon",
    "DrivenTransformProperties",
    "drives_transform",
    "DrivesTransform",
    # Core assets
    "Material",
    "Texture",
    "RenderTexture",
    "Mesh",
    "Shader",
    "AudioClip",
    "AnimationClip",
    "AnimationFrame",
    "AnimStateMachine",
    "AnimState",
    "AnimTransition",
    "AnimCondition",
    "AnimParameter",
    "AssetFile",
    "AssetManager",
    "SandboxPath",
    "TextureRef",
    "RenderTextureRef",
    "ShaderRef",
    "AudioClipRef",
    "AnimationClipRef",
    "AnimStateMachineRef",
    "RenderEffectRef",
    "DataAssetRef",
    # Debug — class only (use Debug.log / Debug.log_warning / …)
    "Debug",
    # Submodules
    "core",
    "components",
    "lifecycle",
    "physics",
    "rendergraph",
    "renderstack",
    "resources",
    "scene",
    "input",
    "ui",
    # Scene
    "GameObjectQuery",
    "LayerMask",
    "SceneManager",
    # Timing & math utilities
    "Time",
    "Mathf",
    # Coroutines
    "Coroutine",
    "WaitForSeconds",
    "WaitForSecondsRealtime",
    "WaitForEndOfFrame",
    "WaitForFrames",
    "WaitForFixedUpdate",
    "WaitUntil",
    "WaitWhile",
    # Batch processing
    "batch_read",
    "batch_write",
    # Object lifecycle
    "Instantiate",
    "Destroy",
]
