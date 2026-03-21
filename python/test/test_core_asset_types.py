"""Tests for InfEngine.core.asset_types — enums, dataclasses, meta file helpers."""

from __future__ import annotations

import json
import os

import pytest

from InfEngine.core.asset_types import (
    AudioCompressionFormat,
    AudioImportSettings,
    FilterMode,
    FontAssetInfo,
    MeshImportSettings,
    ShaderAssetInfo,
    TextureImportSettings,
    TextureType,
    WrapMode,
    _python_type_to_meta_tag,
    asset_category_from_extension,
    IMAGE_EXTENSIONS,
    SHADER_EXTENSIONS,
    MATERIAL_EXTENSIONS,
    AUDIO_EXTENSIONS,
    FONT_EXTENSIONS,
    MESH_EXTENSIONS,
    PREFAB_EXTENSIONS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class TestTextureType:
    def test_values(self):
        assert int(TextureType.DEFAULT) == 0
        assert int(TextureType.NORMAL_MAP) == 1
        assert int(TextureType.UI) == 2


class TestWrapMode:
    def test_from_string(self):
        assert WrapMode.from_string("repeat") == WrapMode.REPEAT
        assert WrapMode.from_string("clamp") == WrapMode.CLAMP
        assert WrapMode.from_string("mirror") == WrapMode.MIRROR
        assert WrapMode.from_string("unknown") == WrapMode.REPEAT

    def test_to_string(self):
        assert WrapMode.REPEAT.to_string() == "repeat"
        assert WrapMode.CLAMP.to_string() == "clamp"
        assert WrapMode.MIRROR.to_string() == "mirror"


class TestFilterMode:
    def test_from_string(self):
        assert FilterMode.from_string("point") == FilterMode.POINT
        assert FilterMode.from_string("nearest") == FilterMode.POINT
        assert FilterMode.from_string("bilinear") == FilterMode.BILINEAR
        assert FilterMode.from_string("linear") == FilterMode.BILINEAR
        assert FilterMode.from_string("trilinear") == FilterMode.TRILINEAR
        assert FilterMode.from_string("unknown") == FilterMode.BILINEAR

    def test_to_string(self):
        assert FilterMode.POINT.to_string() == "point"
        assert FilterMode.BILINEAR.to_string() == "linear"
        assert FilterMode.TRILINEAR.to_string() == "trilinear"


class TestAudioCompressionFormat:
    def test_values(self):
        assert int(AudioCompressionFormat.PCM) == 0
        assert int(AudioCompressionFormat.VORBIS) == 1
        assert int(AudioCompressionFormat.ADPCM) == 2


# ═══════════════════════════════════════════════════════════════════════════
# TextureImportSettings
# ═══════════════════════════════════════════════════════════════════════════

class TestTextureImportSettings:
    def test_defaults(self):
        s = TextureImportSettings()
        assert s.texture_type == TextureType.DEFAULT
        assert s.wrap_mode == WrapMode.REPEAT
        assert s.filter_mode == FilterMode.BILINEAR
        assert s.generate_mipmaps is True
        assert s.srgb is True
        assert s.max_size == 2048
        assert s.aniso_level == 1

    def test_to_dict_round_trip(self):
        s = TextureImportSettings(
            texture_type=TextureType.NORMAL_MAP,
            wrap_mode=WrapMode.CLAMP,
            filter_mode=FilterMode.TRILINEAR,
            generate_mipmaps=False,
            srgb=False,
            max_size=1024,
            aniso_level=4,
        )
        d = s.to_dict()
        s2 = TextureImportSettings.from_dict(d)
        assert s == s2

    def test_copy(self):
        s = TextureImportSettings(max_size=512)
        c = s.copy()
        assert s == c
        c.max_size = 256
        assert s.max_size == 512

    def test_sync_derived_fields_normal_map(self):
        s = TextureImportSettings(srgb=True, texture_type=TextureType.NORMAL_MAP)
        s._sync_derived_fields()
        assert s.srgb is False

    def test_sync_derived_fields_default_preserves_srgb(self):
        s = TextureImportSettings(srgb=True, texture_type=TextureType.DEFAULT)
        s._sync_derived_fields()
        assert s.srgb is True

    def test_equality_false_for_different(self):
        s1 = TextureImportSettings()
        s2 = TextureImportSettings(max_size=512)
        assert s1 != s2

    def test_equality_not_implemented_for_other(self):
        assert TextureImportSettings().__eq__("not a settings") is NotImplemented


# ═══════════════════════════════════════════════════════════════════════════
# AudioImportSettings
# ═══════════════════════════════════════════════════════════════════════════

class TestAudioImportSettings:
    def test_defaults(self):
        s = AudioImportSettings()
        assert s.force_mono is False
        assert s.quality == 1.0
        assert s.compression_format == AudioCompressionFormat.PCM

    def test_to_dict_round_trip(self):
        s = AudioImportSettings(force_mono=True, quality=0.5,
                                compression_format=AudioCompressionFormat.VORBIS)
        d = s.to_dict()
        s2 = AudioImportSettings.from_dict(d)
        assert s == s2

    def test_copy(self):
        s = AudioImportSettings(force_mono=True)
        c = s.copy()
        assert s == c
        c.force_mono = False
        assert s.force_mono is True


# ═══════════════════════════════════════════════════════════════════════════
# MeshImportSettings
# ═══════════════════════════════════════════════════════════════════════════

class TestMeshImportSettings:
    def test_defaults(self):
        s = MeshImportSettings()
        assert s.scale_factor == 0.01
        assert s.generate_normals is True

    def test_to_dict_round_trip(self):
        s = MeshImportSettings(scale_factor=1.0, flip_uvs=True)
        s2 = MeshImportSettings.from_dict(s.to_dict())
        assert s == s2

    def test_copy(self):
        s = MeshImportSettings(optimize_mesh=False)
        c = s.copy()
        assert s == c
        c.optimize_mesh = True
        assert s.optimize_mesh is False


# ═══════════════════════════════════════════════════════════════════════════
# ShaderAssetInfo / FontAssetInfo
# ═══════════════════════════════════════════════════════════════════════════

class TestShaderAssetInfo:
    def test_from_path_vertex(self):
        info = ShaderAssetInfo.from_path("shaders/test.vert", guid="abc")
        assert info.shader_type == "vertex"
        assert info.guid == "abc"

    def test_from_path_fragment(self):
        assert ShaderAssetInfo.from_path("test.frag").shader_type == "fragment"

    def test_from_path_unknown(self):
        assert ShaderAssetInfo.from_path("test.txt").shader_type == "unknown"


class TestFontAssetInfo:
    def test_from_path_ttf(self):
        info = FontAssetInfo.from_path("fonts/arial.ttf", guid="def")
        assert info.font_type == "truetype"

    def test_from_path_otf(self):
        assert FontAssetInfo.from_path("font.otf").font_type == "opentype"

    def test_from_path_unknown(self):
        assert FontAssetInfo.from_path("font.woff").font_type == "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Extension to asset category mapping
# ═══════════════════════════════════════════════════════════════════════════

class TestAssetCategory:
    def test_material(self):
        assert asset_category_from_extension(".mat") == "material"

    def test_texture(self):
        assert asset_category_from_extension(".png") == "texture"
        assert asset_category_from_extension(".jpg") == "texture"

    def test_shader(self):
        assert asset_category_from_extension(".vert") == "shader"
        assert asset_category_from_extension(".frag") == "shader"

    def test_audio(self):
        assert asset_category_from_extension(".wav") == "audio"

    def test_font(self):
        assert asset_category_from_extension(".ttf") == "font"

    def test_mesh(self):
        assert asset_category_from_extension(".fbx") == "mesh"
        assert asset_category_from_extension(".gltf") == "mesh"

    def test_prefab(self):
        assert asset_category_from_extension(".prefab") == "prefab"

    def test_unknown(self):
        assert asset_category_from_extension(".xyz") is None

    def test_case_insensitive(self):
        assert asset_category_from_extension(".PNG") == "texture"

    def test_extension_sets_are_frozensed(self):
        assert isinstance(IMAGE_EXTENSIONS, frozenset)
        assert isinstance(SHADER_EXTENSIONS, frozenset)
        assert isinstance(MESH_EXTENSIONS, frozenset)


# ═══════════════════════════════════════════════════════════════════════════
# _python_type_to_meta_tag helper
# ═══════════════════════════════════════════════════════════════════════════

class TestPythonTypeToMetaTag:
    def test_bool(self):
        assert _python_type_to_meta_tag(True) == "bool"

    def test_int(self):
        assert _python_type_to_meta_tag(42) == "int"

    def test_float(self):
        assert _python_type_to_meta_tag(3.14) == "float"

    def test_string(self):
        assert _python_type_to_meta_tag("hello") == "string"

    def test_none_defaults_to_string(self):
        assert _python_type_to_meta_tag(None) == "string"
