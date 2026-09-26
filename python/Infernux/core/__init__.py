"""
Infernux Core Module

Provides Pythonic wrappers around the C++ engine core, establishing a clean
boundary between the C++ execution engine and the Python business logic layer.

Design Principles:
    - C++ handles: physics, rendering, memory management, resource I/O
    - Python handles: business logic, render graph topology, component scripting
    - All resource lifecycle managed via context managers or explicit acquire/release
    - Clear, minimal, self-documenting API surface

Usage::

    from Infernux.core import Material, Texture, Mesh, Shader, ResourceManager

    # Context-managed resource lifecycle
    with Material.create("MyMaterial") as mat:
        mat.set_color("baseColor", 1.0, 0.0, 0.0)
        mat.set_float("metallic", 0.8)
        renderer.material = mat

    # Shader hot-reload
    Shader.reload("pbr_lit")

    # Render pipeline topology
    from Infernux.rendergraph import RenderGraph, Format
"""

from .material import Material
from .texture import Texture
from .render_texture import RenderTexture
from .mesh import Mesh
from .shader import Shader
from .audio_clip import AudioClip
from .physic_material import PhysicMaterial
from .data_asset import DataAsset
from .animation_clip import AnimationClip, AnimationFrame
from .animation_clip3d import AnimationClip3D, ImportedFloatCurve
from .anim_state_machine import (
    AnimStateMachine,
    AnimState,
    AnimTransition,
    AnimCondition,
    AnimParameter,
)
from .assets import AssetFile, AssetManager
from .sandbox_files import SandboxPath
from .parallel_backend import (
    ParallelBackend,
    ParallelBufferView,
    ParallelCapabilities,
    ParallelTaskState,
)
from .asset_types import (
    TextureCompression, TextureCompressionQuality, TextureFormat, TextureImportSettings,
    TextureType, WrapMode, FilterMode, SpriteFrame,
    ShaderAssetInfo, FontAssetInfo, asset_category_from_extension,
    AudioImportSettings, AudioCompressionFormat,
    MeshImportSettings,
    read_meta_file, write_meta_fields, write_meta_fields_async,
    read_texture_import_settings, write_texture_import_settings,
    read_audio_import_settings, write_audio_import_settings,
    read_mesh_import_settings, write_mesh_import_settings,
)
from .asset_ref import (
    TextureRef,
    RenderTextureRef,
    ShaderRef,
    AudioClipRef,
    AnimationClipRef,
    AnimationClip3DRef,
    AnimStateMachineRef,
    PhysicMaterialRef,
    ParticleGraphRef,
    RenderEffectRef,
    DataAssetRef,
)
from .asset_reference_types import (
    AssetReferenceType,
    AssetTypeRegistry,
    asset_type_registry,
)

__all__ = [
    "Material",
    "Texture",
    "RenderTexture",
    "Mesh",
    "Shader",
    "AudioClip",
    "PhysicMaterial",
    "DataAsset",
    "AnimationClip",
    "AnimationFrame",
    "AnimationClip3D",
    "ImportedFloatCurve",
    "AnimStateMachine",
    "AnimState",
    "AnimTransition",
    "AnimCondition",
    "AnimParameter",
    "AssetFile",
    "AssetManager",
    "SandboxPath",
    "ParallelBackend",
    "ParallelBufferView",
    "ParallelCapabilities",
    "ParallelTaskState",
    "TextureImportSettings",
    "TextureCompression",
    "TextureCompressionQuality",
    "TextureFormat",
    "TextureType",
    "WrapMode",
    "FilterMode",
    "SpriteFrame",
    "ShaderAssetInfo",
    "FontAssetInfo",
    "AudioImportSettings",
    "AudioCompressionFormat",
    "MeshImportSettings",
    "TextureRef",
    "RenderTextureRef",
    "ShaderRef",
    "AudioClipRef",
    "AnimationClipRef",
    "AnimationClip3DRef",
    "AnimStateMachineRef",
    "PhysicMaterialRef",
    "ParticleGraphRef",
    "RenderEffectRef",
    "DataAssetRef",
    "AssetReferenceType",
    "AssetTypeRegistry",
    "asset_type_registry",
]
