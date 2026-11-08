"""Runtime animation curves. No Editor, renderer, or native-window dependency.

The public surface follows Unity's AnimationCurve/Keyframe contract, with
snake_case members and upper-case enum constants. Keys have value semantics;
curves are mutable and deliberately unhashable. ``revision`` lets compiled
consumers invalidate their own snapshots after a curve changes.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from enum import IntEnum, IntFlag
import math
import operator
import struct
from typing import Any, Iterable, MutableSequence


class WrapMode(IntEnum):
    DEFAULT = 0
    ONCE = 1
    CLAMP = 1
    LOOP = 2
    PING_PONG = 4
    CLAMP_FOREVER = 8


class WeightedMode(IntFlag):
    NONE = 0
    IN = 1
    OUT = 2
    BOTH = 3

    @classmethod
    def _missing_(cls, value):
        # IntFlag normally folds negative inputs into its known bit mask.
        # Curve keys retain the signed enum value, including unused bits.
        if isinstance(value, int) and value < 0:
            result = int.__new__(cls, value)
            result._name_, result._value_ = None, value
            return result
        return super()._missing_(value)


_WRAP_NAMES = {
    "default": WrapMode.DEFAULT, "once": WrapMode.ONCE,
    "clamp": WrapMode.CLAMP_FOREVER, "clamp_forever": WrapMode.CLAMP_FOREVER,
    "repeat": WrapMode.LOOP, "loop": WrapMode.LOOP, "ping_pong": WrapMode.PING_PONG,
}
_WRAP_DOCUMENT_NAMES = {0: "default", 2: "repeat", 4: "ping_pong", 8: "clamp"}
_KEY_FLOAT_FIELDS = ("time", "value", "in_tangent", "out_tangent", "in_weight", "out_weight")
_FLOAT32 = struct.Struct("f")
_FLOAT32_BITS = struct.Struct("I")


def _float32(value) -> float:
    try:
        return _FLOAT32.unpack(_FLOAT32.pack(value))[0]
    except OverflowError:
        return -math.inf if value < 0 else math.inf


def _encode_number(value):
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return f"nan:{_FLOAT32_BITS.unpack(_FLOAT32.pack(value))[0]:08x}"
    return "infinity" if value > 0 else "-infinity"


def _decode_number(value):
    if not isinstance(value, str):
        return value
    if value in ("infinity", "-infinity", "nan"):
        return float(value)
    if value.startswith("nan:"):
        result = _FLOAT32.unpack(_FLOAT32_BITS.pack(int(value[4:], 16)))[0]
        if not math.isnan(result):
            raise ValueError("encoded curve NaN must contain NaN bits")
        return result
    return value


def _wrap_mode(value: Any) -> WrapMode:
    if isinstance(value, str):
        try:
            value = _WRAP_NAMES[value]
        except KeyError:
            raise ValueError(f"unknown curve wrap mode: {value!r}") from None
    if isinstance(value, bool):
        raise ValueError("curve wrap mode must be a WrapMode")
    value = operator.index(value)
    return WrapMode(value) if value in (0, 2, 4) else WrapMode.CLAMP_FOREVER


@dataclass(init=False)
class Keyframe:
    time: float
    value: float
    in_tangent: float
    out_tangent: float
    in_weight: float
    out_weight: float
    weighted_mode: WeightedMode
    tangent_mode: int

    def __init__(self, time: float = 0.0, value: float = 0.0,
                 in_tangent: float = 0.0, out_tangent: float = 0.0,
                 in_weight: float | None = None, out_weight: float | None = None,
                 weighted_mode: WeightedMode | None = None, *, tangent_mode: int = 0):
        self.time, self.value = time, value
        self.in_tangent, self.out_tangent = in_tangent, out_tangent
        self.in_weight = 0.0 if in_weight is None else in_weight
        self.out_weight = 0.0 if out_weight is None else out_weight
        self.weighted_mode = (
            WeightedMode.BOTH if in_weight is not None or out_weight is not None else WeightedMode.NONE
        ) if weighted_mode is None else weighted_mode
        # Legacy authoring metadata, not an interpolation selector.
        self.tangent_mode = tangent_mode

    def __setattr__(self, name, value):
        if name in _KEY_FLOAT_FIELDS:
            value = _float32(value)
        elif name == "weighted_mode":
            raw = operator.index(value)
            value = WeightedMode(((raw + 0x80000000) & 0xffffffff) - 0x80000000)
        elif name == "tangent_mode":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("tangent_mode must be an integer")
        object.__setattr__(self, name, value)

    def __copy__(self):
        key = object.__new__(type(self))
        object.__setattr__(key, "__dict__", self.__dict__.copy())
        return key

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return (self.weighted_mode == other.weighted_mode and self.tangent_mode == other.tangent_mode
                and all(a == b or (math.isnan(a) and math.isnan(b))
                        for a, b in ((getattr(self, name), getattr(other, name)) for name in _KEY_FLOAT_FIELDS)))

    def to_dict(self) -> dict[str, Any]:
        # Strict JSON preserves nonfinite fields and quiet NaN payload bits.
        result = {name: _encode_number(getattr(self, name)) for name in _KEY_FLOAT_FIELDS}
        result["weighted_mode"] = int(self.weighted_mode)
        if self.tangent_mode:
            result["tangent_mode"] = self.tangent_mode
        return result

    @classmethod
    def from_dict(cls, value: Any) -> Keyframe:
        required = {"time", "value", "in_tangent", "out_tangent"}
        optional = {"in_weight", "out_weight", "weighted_mode", "tangent_mode"}
        if type(value) is not dict or not required <= value.keys() or value.keys() - required - optional:
            raise ValueError("curve key requires time, value, in_tangent and out_tangent")
        data = dict(value)
        for name in _KEY_FLOAT_FIELDS:
            if name in data:
                data[name] = _decode_number(data[name])
        return cls(**data)


def _copy_key(key: Keyframe | dict) -> Keyframe:
    return key.__copy__() if isinstance(key, Keyframe) else Keyframe.from_dict(key)


def _curve_keys(source):
    keys = (_copy_key(key) for key in source)
    return tuple(sorted((key for key in keys if math.isfinite(key.time) and math.isfinite(key.value)),
                        key=lambda key: key.time))


class AnimationCurve:
    """Mutable, unbounded key collection with copied key reads and writes.

    ``AnimationCurve(k0, k1)`` and ``AnimationCurve(keys=[k0, k1])`` are
    equivalent. The existing ``AnimationCurve(keys, pre_wrap, post_wrap)``
    spelling remains readable for authored Infernux assets/scripts.
    """
    __hash__ = None

    def __init__(self, *keyframes: Keyframe, keys: Iterable[Keyframe] | None = None,
                 pre_wrap_mode: WrapMode = WrapMode.CLAMP_FOREVER,
                 post_wrap_mode: WrapMode = WrapMode.CLAMP_FOREVER,
                 pre_wrap: str | None = None, post_wrap: str | None = None):
        if keyframes and not isinstance(keyframes[0], Keyframe):
            if keys is not None or len(keyframes) > 3:
                raise TypeError("provide keyframes or keys, not both")
            keys = keyframes[0]
            if len(keyframes) > 1:
                pre_wrap = keyframes[1]
            if len(keyframes) > 2:
                post_wrap = keyframes[2]
        elif keyframes:
            if keys is not None:
                raise TypeError("provide keyframes or keys, not both")
            keys = keyframes
        self._revision = 0
        self._compiled_cache = None
        self._hash_cache = None
        self._pre_wrap_mode = _wrap_mode(pre_wrap_mode if pre_wrap is None else pre_wrap)
        self._post_wrap_mode = _wrap_mode(post_wrap_mode if post_wrap is None else post_wrap)
        self._keys = _curve_keys(() if keys is None else keys)
        self._times = tuple(k.time for k in self._keys)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def keys(self) -> list[Keyframe]:
        return [key.__copy__() for key in self._keys]

    @keys.setter
    def keys(self, keys: Iterable[Keyframe]) -> None:
        updated = _curve_keys(keys)
        self._keys = updated
        self._times = tuple(k.time for k in updated)
        self._revision += 1

    def set_keys(self, keys: Iterable[Keyframe]) -> None:
        self.keys = keys

    def get_keys(self, keys: MutableSequence[Keyframe] | None = None) -> list[Keyframe] | None:
        if keys is None:
            return self.keys
        if len(keys) < self.length:
            raise ValueError("destination must have space for all curve keys")
        for i, key in enumerate(self._keys):
            keys[i] = key.__copy__()

    @property
    def length(self) -> int:
        return len(self._keys)

    def __len__(self) -> int:
        return self.length

    def _index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.length:
            raise IndexError("curve key index out of range")
        return index

    def __getitem__(self, index: int) -> Keyframe:
        return self._keys[self._index(index)].__copy__()

    @property
    def pre_wrap_mode(self) -> WrapMode:
        return self._pre_wrap_mode

    @pre_wrap_mode.setter
    def pre_wrap_mode(self, value: WrapMode) -> None:
        self._pre_wrap_mode = _wrap_mode(value)
        self._revision += 1

    @property
    def post_wrap_mode(self) -> WrapMode:
        return self._post_wrap_mode

    @post_wrap_mode.setter
    def post_wrap_mode(self, value: WrapMode) -> None:
        self._post_wrap_mode = _wrap_mode(value)
        self._revision += 1

    @property
    def pre_wrap(self) -> str:
        return _WRAP_DOCUMENT_NAMES[int(self.pre_wrap_mode)]

    @pre_wrap.setter
    def pre_wrap(self, value: str) -> None:
        self.pre_wrap_mode = _wrap_mode(value)

    @property
    def post_wrap(self) -> str:
        return _WRAP_DOCUMENT_NAMES[int(self.post_wrap_mode)]

    @post_wrap.setter
    def post_wrap(self, value: str) -> None:
        self.post_wrap_mode = _wrap_mode(value)

    def _native_snapshot(self):
        revision = self._revision
        cached = self._compiled_cache
        if cached is not None and cached[0] == revision:
            return cached[1]
        from infernux.lib import _Infernux as native
        keys = self._keys
        rows = [(k.time, k.value, k.in_tangent, k.out_tangent, k.in_weight,
                 k.out_weight, int(k.weighted_mode) & 0xffffffff) for k in keys]
        snapshot = native._CurveSnapshot(rows, int(self.pre_wrap_mode), int(self.post_wrap_mode),
                                         [key.tangent_mode for key in keys])
        if self.revision == revision:
            # Publish the version and immutable data as one object. Another
            # thread may retain the previous snapshot while the GIL is released.
            self._compiled_cache = revision, snapshot
        return snapshot

    def evaluate(self, time: float) -> float:
        """Sample in C++; non-finite times follow Unity's evaluation behavior."""
        return self._native_snapshot().evaluate(time)

    def evaluate_many(self, times, *, time_scale: float = 1.0):
        """Sample in one native call; large jobs release the GIL during computation."""
        return self._native_snapshot().evaluate_many(times, time_scale)

    def evaluate_into(self, times, values, *, time_scale: float = 1.0) -> None:
        """Write to a contiguous float64 buffer; no Python callbacks run in the loop."""
        self._native_snapshot().evaluate_into(times, values, time_scale)

    def __deepcopy__(self, memo):
        # Native snapshots are immutable, but need not be serialized or copied.
        result = AnimationCurve()
        result.copy_from(self)
        memo[id(self)] = result
        return result

    def add_key(self, time: Keyframe | float | None = None, value: float | None = None,
                *, key: Keyframe | None = None) -> int:
        if key is not None:
            if time is not None or value is not None:
                raise TypeError("add_key accepts a Keyframe or time, value")
            time = key
        smooth = not isinstance(time, Keyframe)
        if not smooth and value is not None:
            raise TypeError("add_key accepts a Keyframe or time, value")
        key = Keyframe(time, value) if smooth else time.__copy__()
        return self._insert_key(key, smooth)

    def _insert_key(self, key: Keyframe, smooth: bool) -> int:
        if math.isnan(key.time):
            return -1
        index = bisect_left(self._times, key.time)
        if index < self.length and self._times[index] == key.time:
            return -1
        if smooth:
            key.in_weight = key.out_weight = 1.0 / 3.0
        self._keys = self._keys[:index] + (key,) + self._keys[index:]
        self._times = self._times[:index] + (key.time,) + self._times[index:]
        self._revision += 1
        if smooth:
            for i in range(max(0, index-1), min(self.length, index+2)):
                self.smooth_tangents(i, 0.0)
        return index

    def move_key(self, index: int, key: Keyframe) -> int:
        self._index(index)
        key = _copy_key(key)
        self._remove_key(index)
        return self._insert_key(key, False)

    def remove_key(self, index: int) -> None:
        self._index(index)
        self._remove_key(index)

    def _remove_key(self, index: int) -> None:
        self._keys = self._keys[:index] + self._keys[index+1:]
        self._times = self._times[:index] + self._times[index+1:]
        self._revision += 1

    def clear_keys(self) -> None:
        self._keys = self._times = ()
        self._revision += 1

    def smooth_tangents(self, index: int, weight: float) -> None:
        self._index(index)
        weight = _float32(weight)
        if self.length == 1:
            return
        from infernux.lib._Infernux import _curve_smoothed_tangent
        keys = self._keys
        key = keys[index].__copy__()
        left, right = keys[max(0, index-1)], keys[min(index+1, len(keys)-1)]
        side = -1 if index == 0 else 1 if index+1 == len(keys) else 0
        tangent = _curve_smoothed_tangent((left.time, left.value, key.time, key.value, right.time, right.value),
                                          weight, side)
        key.in_tangent = key.out_tangent = tangent
        if index:
            key.in_weight = 1.0 / 3.0
        if index + 1 < len(keys):
            key.out_weight = 1.0 / 3.0
        self._keys = keys[:index] + (key,) + keys[index+1:]
        self._revision += 1

    def copy_from(self, other: AnimationCurve) -> None:
        if not isinstance(other, AnimationCurve):
            raise TypeError("copy_from requires an AnimationCurve")
        self._replace_stored_keys(other._keys)
        self.pre_wrap_mode, self.post_wrap_mode = other.pre_wrap_mode, other.post_wrap_mode

    def _replace_stored_keys(self, keys):
        # Runtime copies retain keys accepted by insertion, including infinity
        # times and nonfinite values. The public keys setter has different rules.
        self._keys = tuple(keys)
        self._times = tuple(key.time for key in self._keys)
        self._revision += 1

    def equals(self, other: object) -> bool:
        if not isinstance(other, AnimationCurve):
            return False
        if self is other:
            return True
        left, right = self._compiled_cache, other._compiled_cache
        if (left is not None and right is not None
                and left[0] == self._revision and right[0] == other._revision):
            return left[1].equals(right[1])
        if (len(self._keys) != len(other._keys) or self._pre_wrap_mode != other._pre_wrap_mode
                or self._post_wrap_mode != other._post_wrap_mode):
            return False
        return self._native_snapshot().equals(other._native_snapshot())

    def __eq__(self, other: object) -> bool:
        return self.equals(other)

    def get_hash_code(self) -> int:
        """Content hash for explicit snapshots; never use a mutable curve as a dict key."""
        revision = self._revision
        cached = self._hash_cache
        if cached is not None and cached[0] == revision:
            return cached[1]
        result = self._native_snapshot().get_hash_code()
        if self._revision == revision:
            self._hash_cache = revision, result
        return result

    @staticmethod
    def linear(time_start: float, value_start: float, time_end: float, value_end: float) -> AnimationCurve:
        a, b = Keyframe(time_start, value_start), Keyframe(time_end, value_end)
        if a.time == b.time:
            return AnimationCurve(a)
        a.out_tangent = b.in_tangent = (b.value-a.value)/(b.time-a.time)
        return AnimationCurve(a, b)

    @staticmethod
    def constant(time_start: float, time_end: float, value: float) -> AnimationCurve:
        return AnimationCurve.linear(time_start, value, time_end, value)

    @staticmethod
    def ease_in_out(time_start: float, value_start: float, time_end: float, value_end: float) -> AnimationCurve:
        a, b = Keyframe(time_start, value_start), Keyframe(time_end, value_end)
        return AnimationCurve(a) if a.time == b.time else AnimationCurve(a, b)

    def to_dict(self) -> dict[str, Any]:
        return dict(keys=[key.to_dict() for key in self._keys], pre_wrap=self.pre_wrap, post_wrap=self.post_wrap)

    @classmethod
    def from_dict(cls, value: Any) -> AnimationCurve:
        if type(value) is not dict or set(value) != {"keys", "pre_wrap", "post_wrap"}:
            raise ValueError("curve requires keys, pre_wrap and post_wrap")
        if type(value["keys"]) is not list:
            raise ValueError("curve keys must be an array")
        keys = [Keyframe.from_dict(key) for key in value["keys"]]
        if any(math.isnan(key.time) for key in keys):
            raise ValueError("stored curve key times cannot be NaN")
        result = cls(pre_wrap=value["pre_wrap"], post_wrap=value["post_wrap"])
        result._replace_stored_keys(sorted(keys, key=lambda key: key.time))
        return result


__all__ = ["AnimationCurve", "Keyframe", "WrapMode", "WeightedMode"]
