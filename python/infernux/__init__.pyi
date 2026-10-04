from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Type, TypeVar

_SerializedValue = TypeVar("_SerializedValue")
__version__: str

# Engine
from infernux.lib import AudioEngine as AudioEngine
from infernux.engine import release_engine as release_engine
from infernux.engine import run_headless as run_headless
from infernux.engine import Engine as Engine
from infernux.engine import LogLevel as LogLevel
from infernux.application import Application as Application
from infernux.screen import Insets as Insets
from infernux.screen import Rect as Rect
from infernux.screen import Screen as Screen
from infernux.acceptance import RuntimeAcceptance as RuntimeAcceptance
from infernux.acceptance import RuntimeAcceptanceManifest as RuntimeAcceptanceManifest
from infernux.acceptance import RuntimeAcceptanceTest as RuntimeAcceptanceTest
# Math
from infernux.math import Vector2 as Vector2
from infernux.math import Vector3 as Vector3
from infernux.math import vec4f as vec4f
from infernux.math import quatf as quatf
from infernux.math import vector2 as vector2
from infernux.math import vector3 as vector3
from infernux.math import vector4 as vector4
from infernux.math import quaternion as quaternion
from infernux.compute import Buffer as Buffer
from infernux.compute import buffer as buffer
# Game Objects
from infernux.lib import GameObject as GameObject
from infernux.lib import Transform as Transform
from infernux.lib import Component as Component
from infernux.lib import Space as Space
from infernux.lib import PrimitiveType as PrimitiveType
from infernux.lib import LineAlignment as LineAlignment
from infernux.lib import LineTextureMode as LineTextureMode
from infernux.lib import LineCurveWrapMode as LineCurveWrapMode
from infernux.lib import LineGradientMode as LineGradientMode
from infernux.lib import LineWidthKey as LineWidthKey
from infernux.lib import LineColorKey as LineColorKey
# Components — user-facing
from infernux.components import InxComponent as InxComponent
from infernux.components import Color as Color
from infernux.instantiate import Instantiate as Instantiate
from infernux.instantiate import Destroy as Destroy
from infernux.components import int_field as int_field
from infernux.components import list_field as list_field
from infernux.components import component_field as component_field
from infernux.components import component_list_field as component_list_field
from infernux.components import hide_field as hide_field
from infernux.components import InspectorSpace as InspectorSpace
from infernux.components import FieldType as FieldType
from infernux.components import AnimationCurve as AnimationCurve
from infernux.components import Keyframe as Keyframe
from infernux.components import Gradient as Gradient
from infernux.components import GradientKey as GradientKey
from infernux.components import GameObjectRef as GameObjectRef
from infernux.components import MaterialRef as MaterialRef
from infernux.components import ComponentRef as ComponentRef
from infernux.components import PrefabRef as PrefabRef
from infernux.components import SerializableObject as SerializableObject
from infernux.core import DataAsset as DataAsset
from infernux.core.render_texture import RenderTexture as RenderTexture
from infernux.ui import UICanvas as UICanvas
from infernux.ui import UIFrame as UIFrame
from infernux.ui import UIGroup as UIGroup
from infernux.ui import UIProgressBar as UIProgressBar
from infernux.ui import UISlider as UISlider
from infernux.ui import UIText as UIText
from infernux.ui import UIImage as UIImage
from infernux.ui import UIRawImage as UIRawImage
from infernux.ui import UIButton as UIButton
from infernux.ui import UIEvent as UIEvent
from infernux.ui import UIEvent1 as UIEvent1

def serialized_field(
    default: _SerializedValue = ...,
    *,
    default_factory: Optional[Callable[[], _SerializedValue]] = ...,
    field_type: Optional[FieldType] = ...,
    element_type: Optional[FieldType] = ...,
    element_class: Optional[Type] = ...,
    serializable_class: Optional[Type] = ...,
    component_type: Optional[str] = ...,
    asset_type: Optional[str] = ...,
    range: Optional[Tuple[float, float]] = ...,
    tooltip: str = ...,
    display_name_key: str = ...,
    enum_labels: Optional[List[str]] = ...,
    readonly: bool = ...,
    header: str = ...,
    space: float = ...,
    group: str = ...,
    info_text: str = ...,
    multiline: bool = ...,
    slider: bool = ...,
    drag_speed: Optional[float] = ...,
    required_component: Optional[str] = ...,
    visible_when: Optional[Callable] = ...,
    hdr: bool = ...,
    curve_non_negative: bool = ...,
    hidden: bool = ...,
) -> _SerializedValue: ...
# Builtin components
from infernux.components import Light as Light
from infernux.components import MeshRenderer as MeshRenderer
from infernux.components import LineRenderer as LineRenderer
from infernux.components import SkinnedMeshRenderer as SkinnedMeshRenderer
from infernux.components import Camera as Camera
from infernux.components import Collider as Collider
from infernux.components import BoxCollider as BoxCollider
from infernux.components import SphereCollider as SphereCollider
from infernux.components import CapsuleCollider as CapsuleCollider
from infernux.components import CylinderCollider as CylinderCollider
from infernux.components import MeshCollider as MeshCollider
from infernux.components import Rigidbody as Rigidbody
from infernux.components import RigidbodyConstraints as RigidbodyConstraints
from infernux.components import CollisionDetectionMode as CollisionDetectionMode
from infernux.components import RigidbodyInterpolation as RigidbodyInterpolation
from infernux.components import HingeJoint as HingeJoint
from infernux.components import SliderJoint as SliderJoint
from infernux.components import AudioSource as AudioSource
from infernux.components import AudioListener as AudioListener
from infernux.components import SpriteRenderer as SpriteRenderer
from infernux.components import SpiritAnimator as SpiritAnimator
from infernux.components import SkeletalAnimator as SkeletalAnimator
from infernux.components import RuntimeAcceptanceRunner as RuntimeAcceptanceRunner
from infernux.lifecycle import InxPreload as InxPreload
from infernux.lifecycle import PreloadContext as PreloadContext
# Decorators
from infernux.components import require_component as require_component
from infernux.components import disallow_multiple as disallow_multiple
from infernux.components import execute_in_edit_mode as execute_in_edit_mode
from infernux.components import add_component_menu as add_component_menu
from infernux.components import icon as icon
from infernux.components import help_url as help_url
from infernux.components import RequireComponent as RequireComponent
from infernux.components import DisallowMultipleComponent as DisallowMultipleComponent
from infernux.components import ExecuteInEditMode as ExecuteInEditMode
from infernux.components import AddComponentMenu as AddComponentMenu
from infernux.components import HelpURL as HelpURL
from infernux.components import Icon as Icon
from infernux.components import DrivenTransformProperties as DrivenTransformProperties
from infernux.components import drives_transform as drives_transform
from infernux.components import DrivesTransform as DrivesTransform
# Core assets
from infernux.core import Material as Material
from infernux.core import Texture as Texture
from infernux.core import Mesh as Mesh
from infernux.core import Shader as Shader
from infernux.core import AudioClip as AudioClip
from infernux.core import AnimationClip as AnimationClip
from infernux.core import AnimationFrame as AnimationFrame
from infernux.core import AnimStateMachine as AnimStateMachine
from infernux.core import AnimState as AnimState
from infernux.core import AnimTransition as AnimTransition
from infernux.core import AnimCondition as AnimCondition
from infernux.core import AnimParameter as AnimParameter
from infernux.core import AssetFile as AssetFile
from infernux.core import AssetManager as AssetManager
from infernux.core import SandboxPath as SandboxPath
from infernux.core import TextureRef as TextureRef
from infernux.core import RenderTextureRef as RenderTextureRef
from infernux.core import ShaderRef as ShaderRef
from infernux.core import AudioClipRef as AudioClipRef
from infernux.core import AnimationClipRef as AnimationClipRef
from infernux.core import AnimStateMachineRef as AnimStateMachineRef
from infernux.core import RenderEffectRef as RenderEffectRef
from infernux.core import DataAssetRef as DataAssetRef
# Debug — class only (use Debug.log / Debug.log_warning / …)
from infernux.debug import Debug as Debug
# Submodules
from infernux import core as core
from infernux import editor as editor
from infernux import components as components
from infernux import lifecycle as lifecycle
from infernux import physics as physics
from infernux import rendergraph as rendergraph
from infernux import renderstack as renderstack
from infernux.renderstack import RenderStack as RenderStack
from infernux import resources as resources
from infernux import scene as scene
from infernux import input as input
from infernux import ui as ui
# Scene
from infernux.scene import GameObjectQuery as GameObjectQuery
from infernux.scene import LayerMask as LayerMask
from infernux.scene import SceneManager as SceneManager
# Timing & math utilities
from infernux.timing import Time as Time
from infernux.mathf import Mathf as Mathf
# Coroutines
from infernux.coroutine import (
    Coroutine as Coroutine,
    WaitForSeconds as WaitForSeconds,
    WaitForSecondsRealtime as WaitForSecondsRealtime,
    WaitForEndOfFrame as WaitForEndOfFrame,
    WaitForFrames as WaitForFrames,
    WaitForFixedUpdate as WaitForFixedUpdate,
    WaitUntil as WaitUntil,
    WaitWhile as WaitWhile,
)
# Batch processing
from infernux.batch import batch_read as batch_read
from infernux.batch import batch_write as batch_write
# JIT helpers (lazy-loaded via __getattr__ at runtime)
from infernux import jit as jit
from infernux import compute as compute
from infernux.jit import JIT_AVAILABLE as JIT_AVAILABLE
from infernux.jit import warmup as warmup
