"""GPU author declarations for statically compiled ``*.frag.py`` assets.

Shader source is parsed, never imported as a game component. Mathematical
operations use GPU types and Slang semantics; they do not run in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._language import INTRINSICS, TYPES


class fullscreen:
    """Fragment program paired with the engine's fullscreen triangle."""


class fragment_input:
    """Raster position in pixel coordinates and interpolated fullscreen UV."""

    position: vec4
    uv: vec2


class Color2D:
    """A typed sampled RGBA image and the sampler supplied by its graph port."""


@dataclass(frozen=True)
class _Input:
    semantic: str
    kind: type
    domain: str | None


def input(semantic: str, kind: type, *, domain: str | None = None):
    return _Input(semantic, kind, domain)


class _Value:
    def __init__(self, *components):
        self.components = components


@dataclass(frozen=True)
class _Parameter:
    kind: str
    default: object
    range: tuple[float, float] | None = None
    label: str | None = None
    hdr: bool = False


class _Parameters:
    @staticmethod
    def float(default=0.0, *, range=None, label=None):
        return _Parameter("float", default, range, label)

    @staticmethod
    def int(default=0, *, range=None, label=None):
        return _Parameter("int", default, range, label)

    @staticmethod
    def uint(default=0, *, range=None, label=None):
        return _Parameter("uint", default, range, label)

    @staticmethod
    def vec2(default=(0.0, 0.0), *, label=None):
        return _Parameter("vec2", default, label=label)

    @staticmethod
    def vec3(default=(0.0, 0.0, 0.0), *, label=None):
        return _Parameter("vec3", default, label=label)

    @staticmethod
    def vec4(default=(0.0, 0.0, 0.0, 0.0), *, label=None):
        return _Parameter("vec4", default, label=label)

    @staticmethod
    def mat3(default=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), *, label=None):
        return _Parameter("mat3", default, label=label)

    @staticmethod
    def mat4(default=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
                      (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)), *, label=None):
        return _Parameter("mat4", default, label=label)

    @staticmethod
    def bool(default=False, *, label=None):
        return _Parameter("bool", default, label=label)

    @staticmethod
    def color(default=(1.0, 1.0, 1.0, 1.0), *, label=None, hdr=False):
        return _Parameter("color", default, label=label, hdr=hdr)


parameter = _Parameters()


def _gpu_intrinsic(name):
    def intrinsic(*args):
        raise RuntimeError(f"shader.{name} executes in a compiled GPU program")
    intrinsic.__name__ = name
    return intrinsic


for _name in TYPES:
    globals()[_name] = type(_name, (_Value,), {"__module__": __name__})
for _name in INTRINSICS:
    globals()[_name] = _gpu_intrinsic(_name)
for _name in ("sample", "sample_level", "sample_grad", "load"):
    globals()[_name] = _gpu_intrinsic(_name)

__all__ = ["fullscreen", "fragment_input", "parameter", "Color2D", "input", "sample", "sample_level", "sample_grad", "load", *TYPES, *INTRINSICS]
