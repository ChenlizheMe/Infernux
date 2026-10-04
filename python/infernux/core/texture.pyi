"""Type stubs for infernux.core.texture."""

from __future__ import annotations

from typing import Optional, Tuple

from infernux.lib._Infernux import TextureData


class Texture:
    """Pythonic wrapper around C++ TextureData.

    Example::

        tex = Texture.load("textures/albedo.png")
        print(tex.width, tex.height, tex.channels)
        pixels = tex.pixels_as_bytes()

        import numpy as np
        arr = np.frombuffer(pixels, dtype=np.uint8).reshape(
            tex.height, tex.width, tex.channels
        )
    """

    def __init__(self, native: TextureData) -> None:
        """Wrap an existing C++ TextureData."""
        ...

    # Factory methods
    @staticmethod
    def load(file_path: str) -> Optional[Texture]:
        """Load a texture from an image file (PNG, JPG, BMP, TGA)."""
        ...
    @staticmethod
    def decode(data: bytes, name: str = ...) -> Optional[Texture]:
        """Decode PNG/JPEG/BMP/TGA bytes into a texture."""
        ...
    @staticmethod
    def solid_color(
        width: int,
        height: int,
        r: int = ...,
        g: int = ...,
        b: int = ...,
        a: int = ...,
    ) -> Optional[Texture]:
        """Create a solid color texture."""
        ...
    @staticmethod
    def checkerboard(
        width: int, height: int, cell_size: int = ...
    ) -> Optional[Texture]:
        """Create a checkerboard pattern texture."""
        ...
    @staticmethod
    def from_native(native: TextureData) -> Texture:
        """Wrap an existing C++ TextureData."""
        ...

    # Context manager
    def __enter__(self) -> Texture:
        """Enter context manager for resource management."""
        ...
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        """Exit context manager."""
        ...

    # Properties
    @property
    def native(self) -> TextureData:
        """The underlying C++ TextureData object."""
        ...
    @property
    def width(self) -> int:
        """Width of the texture in pixels."""
        ...
    @property
    def height(self) -> int:
        """Height of the texture in pixels."""
        ...
    @property
    def channels(self) -> int:
        """Number of color channels (e.g. 3 for RGB, 4 for RGBA)."""
        ...
    @property
    def name(self) -> str:
        """The display name of the texture."""
        ...
    @property
    def guid(self) -> str:
        """Asset GUID when this texture originates from AssetManager."""
        ...
    @property
    def source_path(self) -> str:
        """The file path the texture was loaded from."""
        ...
    @property
    def size(self) -> Tuple[int, int]:
        """``(width, height)`` tuple."""
        ...

    # Pixel data access
    def pixels_as_bytes(self) -> bytes:
        """Get raw pixel data as bytes (row-major, RGBA or RGB)."""
        ...
    def pixels_as_list(self) -> list:
        """Get pixel data as a flat list of integers ``[0-255]``."""
        ...
    def to_numpy(self) -> "numpy.ndarray":  # type: ignore[name-defined]
        """Convert pixel data to a NumPy array ``(H, W, C)``, dtype ``uint8``."""
        ...

    def __repr__(self) -> str: ...
