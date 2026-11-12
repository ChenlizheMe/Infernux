"""GPU author declarations for statically compiled ``*.frag.py`` assets.

Shader source is parsed, never imported as a game component. Mathematical
operations use GPU types and Slang semantics; they do not run in Python.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._language import INTRINSICS, TYPES


class _ShaderAsset:
    shader_id: str
    hidden: bool
    surface_type: str
    render_queue: int
    cull_mode: str
    depth_write: bool | None
    depth_test: str
    blend: str
    pass_tag: str
    stencil: str
    alpha_clip: float | None
    cast_shadows: bool
    receive_shadows: bool


class fullscreen(_ShaderAsset):
    """Fragment program paired with the engine's fullscreen triangle."""


class frag(_ShaderAsset):
    """Mesh fragment program paired with the standard instanced/skinned vertex path."""


class surface(_ShaderAsset):
    """Material surface evaluated by the engine's shared lighting and geometry passes."""


class shadingmodel(_ShaderAsset):
    """A typed lighting response shared by surfaces referring to this GPU asset."""


class vert(_ShaderAsset):
    """Local-space deformation shared by all geometry passes of a material."""


class VertexData:
    """Authored local-space mesh values, before skinning and object transforms."""

    position_os: vec3
    normal_os: vec3
    tangent_os: vec4
    color: vec3
    uv0: vec2
    uv1: vec2


class VertexContext:
    """Time and object state at the geometry sample being evaluated."""

    time: float
    delta_time: float
    object_id: uvec2
    layer_mask: uint
    model_matrix: mat4


class ShadingContext:
    """Lighting geometry independent of mesh UVs, vertex colors and tangent streams."""

    position_ws: vec3
    geometric_normal_ws: vec3
    view_depth: float
    pixel_position: vec2
    layer_mask: uint
    receives_shadows: bool


class LightData:
    """Incident light; color, distance attenuation and visibility are separate."""

    direction: vec3
    color: vec3
    attenuation: float
    shadow: float


def ref(path: str):
    """Reference a statically compiled GPU asset relative to this author file."""
    return path


class mesh_input:
    """Interpolated mesh data in the current geometry view."""

    position: vec4
    position_ws: vec3
    normal_ws: vec3
    tangent_ws: vec4
    color: vec3
    uv0: vec2
    uv1: vec2
    view_depth: float
    line_color: vec4
    object_id: uvec2
    object_flags: uint
    layer_mask: uint


class SurfaceContext(mesh_input):
    """Geometry and object identity supplied to a material surface."""


class fragment_input:
    """Raster position in pixel coordinates and interpolated fullscreen UV."""

    position: vec4
    uv: vec2


class SurfaceData:
    """Linear material properties shared by visible and auxiliary geometry passes."""

    albedo: vec3
    normal_ws: vec3
    metallic: float
    smoothness: float
    occlusion: float
    emission: vec3
    alpha: float
    specular_highlights: float
    shading_param0: float
    shading_param1: float


class Color2D:
    """A typed sampled RGBA image and the sampler supplied by its graph port."""


class Depth2D:
    """A depth attachment read as a scalar through its graph port."""


class UInt2D:
    """An unsigned integer attachment loaded as uvec4 at pixel coordinates."""


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
    def texture_2d(default="white", *, label=None):
        """A material image using the referenced asset's import format and sampler."""
        return _Parameter("texture_2d", default, label=label)

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
material = _Parameters()


def _gpu_intrinsic(name):
    def intrinsic(*args):
        raise RuntimeError(f"shader.{name} executes in a compiled GPU program")
    intrinsic.__name__ = name
    return intrinsic


for _name in TYPES:
    globals()[_name] = type(_name, (_Value,), {"__module__": __name__})
for _name in INTRINSICS:
    globals()[_name] = _gpu_intrinsic(_name)
for _name in ("sample", "sample_level", "sample_grad", "load", "dimensions"):
    globals()[_name] = _gpu_intrinsic(_name)


class _Lighting:
    pbr = staticmethod(_gpu_intrinsic("lighting.pbr"))
    pbr_indirect = staticmethod(_gpu_intrinsic("lighting.pbr_indirect"))
    main_light = staticmethod(_gpu_intrinsic("lighting.main_light"))
    view_direction = staticmethod(_gpu_intrinsic("lighting.view_direction"))


lighting = _Lighting()

__all__ = ["fullscreen", "frag", "surface", "shadingmodel", "vert", "VertexData", "VertexContext", "ref", "lighting", "ShadingContext", "LightData", "fragment_input", "mesh_input", "SurfaceContext", "SurfaceData", "parameter", "material", "Color2D", "Depth2D", "UInt2D", "input", "sample", "sample_level", "sample_grad", "load", "dimensions", *TYPES, *INTRINSICS]
