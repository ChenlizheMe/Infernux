"""Tests for InfEngine.core.texture — Texture wrapper (real C++ backend)."""

from __future__ import annotations

import pytest

from InfEngine.core.texture import Texture
from InfEngine.lib import TextureLoader


# ═══════════════════════════════════════════════════════════════════════════
# Texture creation via real C++ TextureLoader
# ═══════════════════════════════════════════════════════════════════════════

class TestTextureSolidColor:
    def test_create_solid_color(self):
        tex = Texture.solid_color(4, 4, 255, 0, 0, 255)
        assert tex is not None
        assert tex.width == 4
        assert tex.height == 4

    def test_create_solid_color_channels(self):
        tex = Texture.solid_color(2, 2, 128, 64, 32, 255)
        assert tex is not None
        assert tex.channels == 4


class TestTextureCheckerboard:
    def test_create_checkerboard(self):
        tex = Texture.checkerboard(8, 8)
        assert tex is not None
        assert tex.width == 8
        assert tex.height == 8


# ═══════════════════════════════════════════════════════════════════════════
# Texture.__init__ validation
# ═══════════════════════════════════════════════════════════════════════════

class TestTextureInit:
    def test_none_raises(self):
        with pytest.raises(ValueError):
            Texture(None)

    def test_wraps_native(self):
        native = TextureLoader.create_solid_color(8, 8, 255, 255, 255, 255)
        tex = Texture(native)
        assert tex.width == 8
        assert tex.height == 8


# ═══════════════════════════════════════════════════════════════════════════
# Texture properties
# ═══════════════════════════════════════════════════════════════════════════

class TestTextureProperties:
    def test_width_height_channels(self):
        tex = Texture.solid_color(16, 32, 0, 0, 0, 255)
        assert tex.width == 16
        assert tex.height == 32
        assert tex.channels == 4

    def test_name_is_string(self):
        tex = Texture.solid_color(4, 4, 255, 0, 0, 255)
        assert isinstance(tex.name, str)
