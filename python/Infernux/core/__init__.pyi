"""Type stubs for Infernux.core."""

from __future__ import annotations

from .material import Material as Material
from .texture import Texture as Texture
from .shader import Shader as Shader
from .audio_clip import AudioClip as AudioClip
from .animation_clip import AnimationClip as AnimationClip
from .anim_state_machine import (
    AnimStateMachine as AnimStateMachine,
    AnimState as AnimState,
    AnimTransition as AnimTransition,
)
from .assets import AssetManager as AssetManager
from .asset_types import (
    TextureImportSettings as TextureImportSettings,
    TextureType as TextureType,
    WrapMode as WrapMode,
    FilterMode as FilterMode,
    SpriteFrame as SpriteFrame,
    ShaderAssetInfo as ShaderAssetInfo,
    FontAssetInfo as FontAssetInfo,
    asset_category_from_extension as asset_category_from_extension,
    AudioImportSettings as AudioImportSettings,
    AudioCompressionFormat as AudioCompressionFormat,
    MeshImportSettings as MeshImportSettings,
    read_meta_file as read_meta_file,
    write_meta_fields as write_meta_fields,
    read_texture_import_settings as read_texture_import_settings,
    write_texture_import_settings as write_texture_import_settings,
    read_audio_import_settings as read_audio_import_settings,
    write_audio_import_settings as write_audio_import_settings,
    read_mesh_import_settings as read_mesh_import_settings,
    write_mesh_import_settings as write_mesh_import_settings,
)
from .asset_ref import (
    TextureRef as TextureRef,
    ShaderRef as ShaderRef,
    AudioClipRef as AudioClipRef,
    AnimationClipRef as AnimationClipRef,
    AnimStateMachineRef as AnimStateMachineRef,
)

__all__ = [
    "Material",
    "Texture",
    "Shader",
    "AudioClip",
    "AnimationClip",
    "AnimStateMachine",
    "AnimState",
    "AnimTransition",
    "AssetManager",
    "TextureImportSettings",
    "TextureType",
    "WrapMode",
    "FilterMode",
    "SpriteFrame",
    "ShaderAssetInfo",
    "FontAssetInfo",
    "asset_category_from_extension",
    "AudioImportSettings",
    "AudioCompressionFormat",
    "MeshImportSettings",
    "read_meta_file",
    "write_meta_fields",
    "read_texture_import_settings",
    "write_texture_import_settings",
    "read_audio_import_settings",
    "write_audio_import_settings",
    "read_mesh_import_settings",
    "write_mesh_import_settings",
    "TextureRef",
    "ShaderRef",
    "AudioClipRef",
    "AnimationClipRef",
    "AnimStateMachineRef",
]
