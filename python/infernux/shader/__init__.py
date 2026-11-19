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
from typing import Generic, TypeVar

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


class particle(_ShaderAsset):
    """Particle fragment shared by billboard, mesh and ribbon geometry."""

class screen_ui(_ShaderAsset):
    """Screen-space UI fragment; its canvas owns vertex generation."""


class world_ui(_ShaderAsset):
    """World-space UI fragment; its canvas owns projection and billboarding."""


class UIContext:
    """Fragment position, UV, vertex color, primary sample and material color."""
    position: vec4
    uv: vec2
    color: vec4
    texture_color: vec4
    material_color: vec4


class WorldUIContext(UIContext):
    """World UI also supplies the element's local position."""
    local_position: vec2


class ParticleContext:
    """Immutable geometry and lifetime values of one rendered particle."""

    position: vec4
    position_ws: vec3
    normal_ws: vec3
    tangent_ws: vec4
    color: vec3
    uv0: vec2
    view_depth: float
    line_color: vec4
    motion_vector: vec2
    flipbook_grid: vec2
    local_uv: vec2
    next_uv: vec2
    frame_blend: float
    normalized_age: float
    particle_id: uint
    alpha: float
    layer_mask: uint


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
    """Lighting geometry. An omitted shadow plane uses the geometric normal."""

    position_ws: vec3
    geometric_normal_ws: vec3
    shadow_plane_ws: vec3
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


class multisample_input(fragment_input):
    """Per-sample raster input for a fullscreen fragment.

    Reading ``sample_index`` selects per-sample invocation; ordinary fragments
    retain pixel-rate shading. The pipeline's actual sample count defines the
    valid indices. Devices must support per-sample fragment inputs.
    """

    sample_index: uint


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


class ThinSheetData:
    """Parallel smooth optical interfaces on a transparent Surface.

    A Surface opts in by defining thin_sheet(self, ctx) with this return type.
    Thickness uses world units. Absorption is the nonnegative RGB coefficient
    per world unit. The Surface's alpha and emission define coverage and light.
    The initial capture uses the geometric normal for perfect reflection.
    """

    ior: float
    thickness: float
    absorption: vec3


class RaySurfaceHit:
    """The closest accepted world ray hit and its supported authored surface.

    ``available`` is true for a resolved supported surface or a definite miss.
    Unprepared world data and possible hits with unavailable surface data
    leave the query unavailable.

    Built-in Lit/Unlit and static PBR ``shader.surface`` authors use their
    actual material values, typed ShaderLib functions and explicit
    ``sample_level`` reads from the same published ray texture pages. The
    compiler assigns their layouts; no separate ray material is required.
    Raster derivatives, frame/view-dependent expressions, custom lighting
    models and unsupported parameter types retain a located secondary-surface
    authoring diagnostic. A possible hit on one leaves ``available`` false.
    """

    available: bool
    found: bool
    distance: float
    position_ws: vec3
    surface_id: uint
    surface: SurfaceData


class Color2D:
    """A typed sampled RGBA image and the sampler supplied by its graph port."""


class Color3D:
    """A sampled RGBA volume, addressed with three-dimensional coordinates."""


class Color2DMS:
    """Multisampled RGBA values loaded with pixel coordinates and a sample index."""


class Depth2DMS:
    """Unresolved depth samples loaded with pixel coordinates and a sample index."""


class UInt2DMS:
    """Unsigned integer samples; identities are never filtered or averaged."""


class History2D:
    """A previous-view color image with validity owned by its temporal graph resource."""

    valid: bool


class Depth2D:
    """Unfiltered scalar depth from a depth plane or a resolved R32 image."""


class UInt2D:
    """An unsigned integer attachment loaded as uvec4 at pixel coordinates."""


_Element = TypeVar("_Element")


class Buffer(Generic[_Element]):
    """A read-only graph buffer of tightly packed 32-bit scalar or vector elements.

    Indexing reads an element; ``len`` reports the logical view's element count.
    Compute owns writes, allocation, and the producer dependency. Shader authors
    do not declare descriptor slots, offsets, or backend alignment.
    """


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


class float(_Value):
    """Scalar floating-point value."""


class int(_Value):
    """Signed scalar integer."""


class uint(_Value):
    """Unsigned scalar integer."""


class bool(_Value):
    """Scalar boolean."""


class f32(_Value):
    """32-bit floating-point value."""


class i32(_Value):
    """32-bit signed integer."""


class u32(_Value):
    """32-bit unsigned integer."""


class vec2(_Value):
    """Two floating-point components."""


class vec3(_Value):
    """Three floating-point components."""


class vec4(_Value):
    """Four floating-point components."""


class ivec2(_Value):
    """Two signed integer components."""


class ivec3(_Value):
    """Three signed integer components."""


class ivec4(_Value):
    """Four signed integer components."""


class uvec2(_Value):
    """Two unsigned integer components."""


class uvec3(_Value):
    """Three unsigned integer components."""


class uvec4(_Value):
    """Four unsigned integer components."""


class bvec2(_Value):
    """Two boolean components."""


class bvec3(_Value):
    """Three boolean components."""


class bvec4(_Value):
    """Four boolean components."""


class mat2(_Value):
    """2 by 2 floating-point matrix."""


class mat3(_Value):
    """3 by 3 floating-point matrix."""


class mat4(_Value):
    """4 by 4 floating-point matrix."""


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
    intrinsic.__doc__ = {
        "inverse": "Inverse of a nonsingular float mat2, mat3 or mat4; singular results are undefined.",
        "matrix_comp_mult": "Elementwise product of float matrices with the same row/column shape (also a * b).",
        "outer_product": "Outer product of float vectors: result[i][j] = column[i] * row[j].",
    }.get(name)
    return intrinsic


for _name in INTRINSICS:
    globals()[_name] = _gpu_intrinsic(_name)
for _name in ("sample", "sample_level", "sample_bias", "sample_grad", "load", "dimensions", "sample_count", "clip", "frame_time", "frame_delta_time", "view", "camera"):
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
    particle_pbr = staticmethod(_gpu_intrinsic("lighting.particle_pbr"))
    six_way_smoke = staticmethod(_gpu_intrinsic("lighting.six_way_smoke"))


lighting = _Lighting()


class _RayQueries:
    """Fullscreen queries using shared world geometry and secondary surface evaluators."""

    @staticmethod
    def trace_closest(origin_ws: vec3, direction_ws: vec3, min_distance: float, max_distance: float,
                      layer_mask: uint, lod: float) -> RaySurfaceHit:
        """Trace along a unit direction over the given world-space distance interval.

        The explicit texture LOD applies to alpha testing and surface evaluation.
        Read the surface properties, position and surface ID when ``found`` is true.
        A miss or unavailable surface keeps ``distance`` at ``max_distance``.
        Blended draws belong to the transparent pass and do not block this query.
        """
        raise RuntimeError("shader.ray.trace_closest executes in a compiled GPU program")

    @staticmethod
    def trace_visibility(origin_ws: vec3, direction_ws: vec3, min_distance: float, max_distance: float,
                         layer_mask: uint, lod: float) -> bool:
        """Return true for a clear path through the opaque ray geometry.

        An opaque draw with unavailable surface data conservatively blocks the path.
        """
        raise RuntimeError("shader.ray.trace_visibility executes in a compiled GPU program")


ray = _RayQueries()

__all__ = ["fullscreen", "frag", "surface", "particle", "ParticleContext", "shadingmodel", "vert", "VertexData", "VertexContext", "ProjectionContext", "CameraContext", "ViewContext", "FragmentOutput", "ref", "lighting", "ShadingContext", "LightData", "fragment_input", "multisample_input", "mesh_input", "SurfaceContext", "SurfaceData", "ThinSheetData", "ray", "RaySurfaceHit", "parameter", "material", "Color2D", "History2D", "Depth2D", "UInt2D", "Color3D", "Color2DMS", "Depth2DMS", "UInt2DMS", "Buffer", "input", "sample", "sample_level", "sample_bias", "sample_grad", "load", "dimensions", "sample_count", "clip", "frame_time", "frame_delta_time", "view", "camera", *TYPES, *INTRINSICS]
