"""Platform-independent window, framebuffer, safe-area, and keyboard metrics."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as _integer_index
from typing import Optional, Tuple

from infernux.lib import InputManager as _NativeInputManager


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def __iter__(self):
        return iter((self.x, self.y, self.width, self.height))

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return (self.x, self.y, self.width, self.height)[index]


@dataclass(slots=True, eq=False)
class RectOffset:
    """Mutable integer edges around a rectangle, following Unity's value API.

    This is a reference object, distinct from the immutable OS ``Insets``.
    Negative edges are valid. Tween playback captures a separate start value.
    """

    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0

    @classmethod
    def _from_int32(cls, values):
        """Construct from four already validated native-width channels."""
        result = object.__new__(cls)
        object.__setattr__(result, "left", values[0])
        object.__setattr__(result, "right", values[1])
        object.__setattr__(result, "top", values[2])
        object.__setattr__(result, "bottom", values[3])
        return result

    def __setattr__(self, name, value):
        if name in ("left", "right", "top", "bottom"):
            value = _integer_index(value)
            if not -(2**31) <= value < 2**31:
                raise OverflowError("RectOffset edges must fit a signed 32-bit integer")
        object.__setattr__(self, name, value)

    @property
    def horizontal(self) -> int:
        return self.left + self.right

    @property
    def vertical(self) -> int:
        return self.top + self.bottom

    def add(self, rect: Rect) -> Rect:
        return Rect(rect.x-self.left, rect.y-self.top,
                    rect.width+self.horizontal, rect.height+self.vertical)

    def remove(self, rect: Rect) -> Rect:
        return Rect(rect.x+self.left, rect.y+self.top,
                    rect.width-self.horizontal, rect.height-self.vertical)


@dataclass(frozen=True, slots=True)
class Insets:
    left: int
    top: int
    right: int
    bottom: int


class _ScreenMeta(type):
    @staticmethod
    def _state():
        return _NativeInputManager.instance().screen_state

    @property
    def revision(cls) -> int:
        return int(cls._state().revision)

    @property
    def size(cls) -> Tuple[int, int]:
        state = cls._state()
        return (int(state.logical_width), int(state.logical_height))

    @property
    def framebuffer_size(cls) -> Tuple[int, int]:
        state = cls._state()
        return (int(state.framebuffer_width), int(state.framebuffer_height))

    @property
    def pixel_ratio(cls) -> float:
        return float(cls._state().pixel_ratio)

    @property
    def safe_area(cls) -> Rect:
        state = cls._state()
        return Rect(
            int(state.safe_area_x),
            int(state.safe_area_y),
            int(state.safe_area_width),
            int(state.safe_area_height),
        )

    @property
    def safe_insets(cls) -> Insets:
        state = cls._state()
        width = int(state.logical_width)
        height = int(state.logical_height)
        x = int(state.safe_area_x)
        y = int(state.safe_area_y)
        safe_width = int(state.safe_area_width)
        safe_height = int(state.safe_area_height)
        return Insets(
            x,
            y,
            max(0, width - x - safe_width),
            max(0, height - y - safe_height),
        )

    @property
    def keyboard_inset(cls) -> Optional[int]:
        state = cls._state()
        if not bool(state.keyboard_inset_known):
            return None
        return int(state.keyboard_inset)

    @property
    def focused(cls) -> bool:
        return bool(cls._state().focused)

    @property
    def occluded(cls) -> bool:
        return bool(cls._state().occluded)


class Screen(metaclass=_ScreenMeta):
    """Live game-window metrics in logical units.

    ``safe_area`` is derived from SDL WindowInsets on Android and CSS
    ``safe-area-inset-*`` on Web. ``keyboard_inset`` is ``None`` when the
    platform cannot report it reliably; no fixed keyboard height is guessed.
    """


__all__ = ["Insets", "Rect", "RectOffset", "Screen"]
