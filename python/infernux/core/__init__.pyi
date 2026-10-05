"""Type stubs for infernux.core."""

from __future__ import annotations

from .material import Material as Material
from .texture import Texture as Texture
from .render_texture import RenderTexture as RenderTexture
from .mesh import Mesh as Mesh
from .shader import Shader as Shader
from .audio_clip import AudioClip as AudioClip
from .physic_material import PhysicMaterial as PhysicMaterial
from .animation_clip import AnimationClip as AnimationClip, AnimationFrame as AnimationFrame
from .animation_clip3d import AnimationClip3D as AnimationClip3D, ImportedFloatCurve as ImportedFloatCurve
from .anim_state_machine import (
    AnimStateMachine as AnimStateMachine,
    AnimState as AnimState,
    AnimTransition as AnimTransition,
    AnimCondition as AnimCondition,
    AnimParameter as AnimParameter,
)
from .data_asset import DataAsset as DataAsset
from .assets import AssetFile as AssetFile
from .assets import AssetManager as AssetManager
from .sandbox_files import SandboxPath as SandboxPath
from .parallel_backend import (
    ParallelBackend as ParallelBackend,
    ParallelBufferView as ParallelBufferView,
    ParallelCapabilities as ParallelCapabilities,
    ParallelTaskState as ParallelTaskState,
)
from .asset_types import (
    TextureCompression as TextureCompression,
    TextureCompressionQuality as TextureCompressionQuality,
    TextureFormat as TextureFormat,
    TextureImportSettings as TextureImportSettings,
    TextureType as TextureType,
    WrapMode as WrapMode,
    FilterMode as FilterMode,
    SpriteFrame as SpriteFrame,
    ShaderAssetInfo as ShaderAssetInfo,
    FontAssetInfo as FontAssetInfo,
    AudioImportSettings as AudioImportSettings,
    AudioCompressionFormat as AudioCompressionFormat,
    MeshImportSettings as MeshImportSettings,
    asset_category_from_extension as asset_category_from_extension,
    read_meta_file as read_meta_file,
    write_meta_fields as write_meta_fields,
    write_meta_fields_async as write_meta_fields_async,
    read_texture_import_settings as read_texture_import_settings,
    write_texture_import_settings as write_texture_import_settings,
    read_audio_import_settings as read_audio_import_settings,
    write_audio_import_settings as write_audio_import_settings,
    read_mesh_import_settings as read_mesh_import_settings,
    write_mesh_import_settings as write_mesh_import_settings,
)
from .asset_ref import (
    TextureRef as TextureRef,
    RenderTextureRef as RenderTextureRef,
    ShaderRef as ShaderRef,
    AudioClipRef as AudioClipRef,
    AnimationClipRef as AnimationClipRef,
    AnimationClip3DRef as AnimationClip3DRef,
    AnimStateMachineRef as AnimStateMachineRef,
    PhysicMaterialRef as PhysicMaterialRef,
    ParticleGraphRef as ParticleGraphRef,
    RenderEffectRef as RenderEffectRef,
    DataAssetRef as DataAssetRef,
)
from .asset_reference_types import AssetReferenceType as AssetReferenceType, AssetTypeRegistry as AssetTypeRegistry, asset_type_registry as asset_type_registry

__all__ = [
    "Material",
    "Texture",
    "RenderTexture",
    "Mesh",
    "Shader",
    "AudioClip",
    "PhysicMaterial",
    "AnimationClip",
    "AnimationFrame",
    "AnimationClip3D",
    "ImportedFloatCurve",
    "AnimStateMachine",
    "AnimState",
    "AnimTransition",
    "AnimCondition",
    "AnimParameter",
    "DataAsset",
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
