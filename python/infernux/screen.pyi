from __future__ import annotations

from typing import Optional, Tuple

class Rect:
    x: float
    y: float
    width: float
    height: float
    def __init__(self, x: float, y: float, width: float, height: float) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __getitem__(self, index): ...

class RectOffset:
    left: int
    right: int
    top: int
    bottom: int
    def __init__(self, left: int = ..., right: int = ..., top: int = ..., bottom: int = ...) -> None: ...
    @classmethod
    def _from_int32(cls, values: tuple[int, int, int, int] | list[int]) -> RectOffset: ...
    @property
    def horizontal(self) -> int: ...
    @property
    def vertical(self) -> int: ...
    def add(self, rect: Rect) -> Rect: ...
    def remove(self, rect: Rect) -> Rect: ...

class Insets:
    left: int
    top: int
    right: int
    bottom: int

class Screen:
    revision: int
    size: Tuple[int, int]
    framebuffer_size: Tuple[int, int]
    pixel_ratio: float
    safe_area: Rect
    safe_insets: Insets
    keyboard_inset: Optional[int]
    focused: bool
    occluded: bool

__all__: list[str]
