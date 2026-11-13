"""GPU author declarations for statically compiled ``*.frag.py`` assets.

Shader source is parsed, never imported as a game component. Mathematical
operations use GPU types and Slang semantics; they do not run in Python.

``frame_time()`` and ``frame_delta_time()`` read the engine-owned render clock
in seconds. Vertex deformation uses ``VertexContext.time`` for the geometry
sample, including the previous sample used to calculate motion vectors.
``view()`` returns the current view's engine-owned camera and output settings.
``camera()`` supplies its matrices in canonical clip coordinates (+Y up).
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
    """Local-space deformation shared by all geometry passes of a material.

    With a raw ``frag``, an optional ``project(mesh_input, ProjectionContext)``
    method returns clip position. Surfaces retain the engine's projection for
    their lighting, depth, shadow and motion passes.
    """


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


class ProjectionContext:
    """Current view matrices with a target-independent projection (+Y up)."""

    view_matrix: mat4
    projection_matrix: mat4


class CameraContext:
    """Camera matrices shared by geometry and fullscreen passes.

    Clip X/Y span [-w, w], with +Y up; depth uses the engine's [0, w]
    convention. Normalized image UVs start at the top left. Matrices include
    the current view's temporal projection jitter.
    """

    view_matrix: mat4
    projection_matrix: mat4
    view_projection_matrix: mat4
    previous_view_projection_matrix: mat4
    inverse_view_projection_matrix: mat4
    projection_params: vec4
    depth_params: vec4


class FragmentOutput:
    """Fragment color and normalized device depth, written together.

    Return this instead of ``vec4`` when a mesh or fullscreen fragment computes
    depth. The render pass still controls depth testing and attachment writes.
    Camera projection and reconstruction helpers use the same depth convention.
    """

    color: vec4
    depth: float


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


class ViewContext:
    """Camera-local values read from the shared frame constant buffer."""

    size: vec2
    pixel_size: vec2
    camera_position_ws: vec3
    near_clip: float
    far_clip: float
    depth_params: vec4
    dithering: bool
    stop_nans: bool
    opaque_output: bool


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


class History2D:
    """A previous-view color image with validity owned by its temporal graph resource."""

    valid: bool


class Depth2D:
    """Unfiltered scalar depth from a depth plane or a resolved R32 image."""


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
for _name in ("sample", "sample_level", "sample_bias", "sample_grad", "load", "dimensions", "clip", "frame_time", "frame_delta_time", "view", "camera"):
    globals()[_name] = _gpu_intrinsic(_name)


class _Lighting:
    """Scene lighting and shared Surface responses.

    ``shade(surface, context, model_id)`` evaluates a built-in GBuffer model
    using the same source as its Surface. Fullscreen programs use tiled lights.
    """

    shade = staticmethod(_gpu_intrinsic("lighting.shade"))
    pbr = staticmethod(_gpu_intrinsic("lighting.pbr"))
    pbr_indirect = staticmethod(_gpu_intrinsic("lighting.pbr_indirect"))
    main_light = staticmethod(_gpu_intrinsic("lighting.main_light"))
    view_direction = staticmethod(_gpu_intrinsic("lighting.view_direction"))
    direct = staticmethod(_gpu_intrinsic("lighting.direct"))
    ambient = staticmethod(_gpu_intrinsic("lighting.ambient"))


lighting = _Lighting()

__all__ = ["fullscreen", "frag", "surface", "shadingmodel", "vert", "VertexData", "VertexContext", "ProjectionContext", "CameraContext", "ViewContext", "FragmentOutput", "ref", "lighting", "ShadingContext", "LightData", "fragment_input", "mesh_input", "SurfaceContext", "SurfaceData", "parameter", "material", "Color2D", "History2D", "Depth2D", "UInt2D", "input", "sample", "sample_level", "sample_bias", "sample_grad", "load", "dimensions", "clip", "frame_time", "frame_delta_time", "view", "camera", *TYPES, *INTRINSICS]
