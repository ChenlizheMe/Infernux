"""
Texture wrapper.

Wraps the C++ TextureData and TextureLoader with context manager support
and a clean API for texture loading, creation, and pixel access.

Usage::

    # Load from file
    tex = Texture.load("textures/albedo.png")
    print(tex.width, tex.height, tex.channels)

    # Create procedural textures
    tex = Texture.solid_color(64, 64, r=255, g=0, b=0)
    tex = Texture.checkerboard(256, 256, cell_size=32)

    # Access pixel data as bytes (for NumPy/PIL interop)
    pixels = tex.pixels_as_bytes()
    import numpy as np
    arr = np.frombuffer(pixels, dtype=np.uint8).reshape(tex.height, tex.width, tex.channels)

    # Context manager
    with Texture.load("textures/normal.png") as tex:
        # use tex
        pass
"""

from __future__ import annotations

from typing import Optional

from Infernux.lib import TextureLoader, TextureData


class Texture:
    """Pythonic wrapper around C++ TextureData.

    Provides:
    - Context manager for scoped lifecycle
    - Factory methods for loading and procedural generation
    - Easy pixel data access for external tooling
    """

    def __init__(self, native: "TextureData"):
        if native is None:
            raise ValueError("Cannot wrap a None TextureData")
        self._native = native
        self._guid = ""

    # ==========================================================================
    # Factory Methods
    # ==========================================================================

    @staticmethod
    def load(file_path: str) -> Optional["Texture"]:
        """Load a texture from an image file (PNG, JPG, BMP, TGA)."""
        if TextureLoader is None:
            return None
        if hasattr(TextureLoader, "load_from_file"):
            native = TextureLoader.load_from_file(file_path)
        elif hasattr(TextureLoader, "load"):
            native = TextureLoader.load(file_path)
        else:
            raise AttributeError("TextureLoader has neither load_from_file() nor load()")
        if native and native.width > 0:
            return Texture(native)
        return None

    @staticmethod
    def from_memory(data: bytes, width: int, height: int, channels: int = 4,
                    name: str = "memory_texture") -> Optional["Texture"]:
        """Create a texture from raw pixel data in memory."""
        if TextureLoader is None:
            return None
        # The current native binding decodes image bytes directly and only
        # accepts (data, name).
        native = TextureLoader.load_from_memory(data, name)
        if native:
            return Texture(native)
        return None

    @staticmethod
    def solid_color(width: int, height: int, r: int = 255, g: int = 255,
                    b: int = 255, a: int = 255) -> Optional["Texture"]:
        """Create a solid color texture."""
        if TextureLoader is None:
            return None
        native = TextureLoader.create_solid_color(width, height, r, g, b, a)
        if native:
            return Texture(native)
        return None

    @staticmethod
    def checkerboard(width: int, height: int, cell_size: int = 32) -> Optional["Texture"]:
        """Create a checkerboard pattern texture."""
        if TextureLoader is None:
            return None
        native = TextureLoader.create_checkerboard(width, height, cell_size)
        if native:
            return Texture(native)
        return None

    @staticmethod
    def from_native(native: "TextureData") -> "Texture":
        """Wrap an existing C++ TextureData."""
        return Texture(native)

    # ==========================================================================
    # Context Manager
    # ==========================================================================

    def __enter__(self) -> "Texture":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def native(self) -> "TextureData":
        """Access the underlying C++ TextureData."""
        return self._native

    @property
    def guid(self) -> str:
        """Asset GUID when this texture originates from AssetManager/AssetDatabase."""
        return self._guid

    @property
    def width(self) -> int:
        return self._native.width

    @property
    def height(self) -> int:
        return self._native.height

    @property
    def channels(self) -> int:
        return self._native.channels

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def source_path(self) -> str:
        return self._native.source_path

    @property
    def size(self) -> tuple:
        """Return (width, height) tuple."""
        return (self.width, self.height)

    # ==========================================================================
    # Pixel data access for external tooling
    # ==========================================================================

    def pixels_as_bytes(self) -> bytes:
        """Get raw pixel data as bytes.

        Layout: row-major, RGBA (or RGB) depending on channel count.
        Suitable for conversion to NumPy arrays or PIL images.
        """
        return self._native.pixels_as_bytes()

    def pixels_as_list(self) -> list:
        """Get pixel data as a flat list of integers [0-255]."""
        return self._native.pixels_as_list()

    def to_numpy(self):
        """Convert pixel data to a NumPy array (H, W, C).

        Requires NumPy to be installed.

        Returns:
            numpy.ndarray with shape (height, width, channels), dtype=uint8
        """
        import numpy as np
        data = self.pixels_as_bytes()
        return np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, self.channels
        )

    # ==========================================================================
    # Dunder methods
    # ==========================================================================

    def __repr__(self):
        return f"Texture(name='{self.name}', size={self.width}x{self.height}, ch={self.channels})"
