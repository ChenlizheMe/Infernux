"""Camera-local graph authoring for indirect lighting and whole-scene renderers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property

from infernux.rendergraph.graph import Format, RenderGraph, TextureHandle
from infernux.renderstack.pass_result import PassResult


class RayTracingQuality(IntEnum):
    LIGHTWEIGHT = 0
    STANDARD = 1
    HIGH = 2


class RayTracingWorld:
    """GPU resources of the world bound by the renderer. Each is requested on use."""

    def __init__(self, graph: RenderGraph):
        self._graph = graph

    @cached_property
    def geometry(self):
        return self._graph._create_world_ray_geometry()

    @cached_property
    def meshes(self):
        return self._graph._create_world_ray_meshes()

    @cached_property
    def surface_records(self):
        return self._graph._create_world_surface_records()

    @cached_property
    def materials(self):
        """Geometry, meshes, reflected material data and texture pages for secondary hits."""
        return self._graph._create_world_ray_lit()


class RayTracingView:
    """Per-view graph inputs. Shaders access native camera matrices with shader.camera()."""

    def __init__(self, ctx: RayTracingContext):
        self._ctx = ctx

    @cached_property
    def lights(self):
        surface = self._ctx.surface
        if surface is not None and surface.has("light_list"):
            return surface.sample("light_list")
        return self._ctx.graph.create_view_light_list(f"{self._ctx.name}/lights")

    def history(self, name: str, *, format: Format, size_divisor: int = 0, respond_to_lighting: bool = False):
        """Allocate per-camera history. A filter that remeasures changing light can retain its prior estimate."""
        read, write = self._ctx.graph.create_temporal_history(
            f"{self._ctx.name}/{name}", format=format, size_divisor=size_divisor)
        read._depends_on_ray_world = write._depends_on_ray_world = True
        read._responds_to_ray_lighting = write._responds_to_ray_lighting = respond_to_lighting
        return read, write


class RayTracingSignal:
    """One GPU lighting estimate. Accessing radiance declares its linear texture once."""

    def __init__(self, resolve: Callable[[], TextureHandle]):
        self._resolve = resolve

    @cached_property
    def radiance(self) -> TextureHandle:
        """Linear RGB and positive alpha for a valid sample, including valid black."""
        return self._resolve()


@dataclass
class RayTracingResult:
    raw: PassResult
    denoised: PassResult
    scene_color: TextureHandle | None = None
    # The default tracer can fuse reconstruction into material lighting. A
    # replacement that publishes HDR or linear layers acquires no default passes.
    _packets: dict[str, TextureHandle] = field(default_factory=dict, repr=False)
    _primary: TextureHandle | None = field(default=None, repr=False)
    _after_sky: Callable | None = field(default=None, repr=False)

    def after_sky(self, surface: PassResult | None = None) -> bool:
        """Finish world-cache updates and return whether transparency was drawn."""
        return self._after_sky(surface) if self._after_sky is not None else False


@dataclass
class RayTracingContext:
    """Inputs requested by a pipeline, resolved for the native view at execution.

    Surface and original geometry are optional for a whole-scene renderer.
    Shaders read the current camera through shader.camera(). The graph owns all
    buffers and per-view histories. No per-frame Python shading is involved.
    """

    graph: RenderGraph
    surface: PassResult | None = None
    geometry: PassResult | None = None
    diffuse: bool = True
    specular: bool = True
    transparency: bool = False
    quality: RayTracingQuality = RayTracingQuality.STANDARD
    reconstruct: bool = True
    name: str = "ray"
    gi_intensity: float = 1.0
    ssao_strength: float = 0.0
    gi_saturation: float = 1.0
    ssao_contrast: float = 1.0

    @cached_property
    def world(self) -> RayTracingWorld:
        return RayTracingWorld(self.graph)

    @cached_property
    def view(self) -> RayTracingView:
        return RayTracingView(self)

    @cached_property
    def ambient_visibility(self) -> TextureHandle | None:
        """Screen-space visibility shared by diffuse reconstruction and lighting."""
        if not self.diffuse:
            return None
        source = self.surface
        if self.geometry is not None and (source is None or not source.has("ambient_visibility")):
            source = self.geometry
        return source.sample("ambient_visibility") if source is not None else None

    def _publish_layers(self, stage, layers):
        textures = {}
        providers = {}
        for name, value in (layers or {}).items():
            if isinstance(value, RayTracingSignal):
                providers[name] = lambda signal=value: signal.radiance
            elif callable(value):
                providers[name] = value
            else:
                textures[name] = value

        return self.graph.publish_pass_result(self.graph._scoped_name(f"{self.name}/{stage}"), textures, providers=providers)

    def shade_deferred(self, result: RayTracingResult, target: TextureHandle, *, name: str = "DeferredLightingPass") -> None:
        """Write to the pipeline's own HDR target using the material's deferred shading model.

        Built-in estimates resolve directly in this pass. A custom tracer may
        supply linear lighting or replace the entire scene, including the sky.
        """
        from infernux.renderstack._pipeline_common import bind_deferred_surface, deferred_surface_inputs, draw_deferred_lighting

        if result.scene_color is not None:
            if target is not result.scene_color:
                with self.graph.add_pass(name) as p:
                    p.write_color(target)
                    p.set_texture("source_tex", result.scene_color)
                    p.fullscreen_quad("infernux/internal/fullscreen_copy")
            return

        surface = deferred_surface_inputs(self.surface)
        enabled = [key for key, active in (("diffuse_irradiance", self.diffuse), ("specular_radiance", self.specular)) if active]
        packed = {key: result._packets[key] for key in enabled if key in result._packets}
        linear = {key: result.denoised.sample(key) for key in enabled if key not in packed}
        visibility = self.ambient_visibility
        with self.graph.add_pass(name) as p:
            p.write_color(target)
            p.set_clear(color=(0.0, 0.0, 0.0, 1.0))
            if not enabled:
                draw_deferred_lighting(p, surface)
                return
            bind_deferred_surface(p, surface)
            p.set_texture("ambient_visibility", visibility or surface["surface_frame"])
            p.set_param("diffuse_intensity", self.gi_intensity)
            p.set_param("diffuse_saturation", self.gi_saturation)
            p.set_param("ssao_strength", self.ssao_strength if visibility is not None else 0.0)
            p.set_param("ssao_contrast", max(0.25, self.ssao_contrast))
            if not self.specular and not packed:
                p.set_texture("diffuse_irradiance", linear["diffuse_irradiance"])
                p.fullscreen_quad("infernux/internal/deferred_linear_lighting")
                return

            primary = result._primary or surface["surface"]
            p.set_texture("primary_tex", primary)
            for key, port in (("diffuse_irradiance", "diffuse"), ("specular_radiance", "specular")):
                p.set_texture(f"ray_{port}", packed.get(key, primary))
                p.set_texture(key, linear.get(key, surface["surface_frame"]))
                p.set_param(f"{port}_mode", 2 if key in packed else int(key in linear))
            p.fullscreen_quad("infernux/internal/deferred_ray_lighting")

    def publish(
        self, *, raw: Mapping[str, TextureHandle | RayTracingSignal | Callable[[], TextureHandle]] | None = None,
        denoised: Mapping[str, TextureHandle | RayTracingSignal] | None = None,
        scene_color: TextureHandle | None = None,
    ) -> RayTracingResult:
        """Return linear lighting layers, or an entire linear HDR scene image.

        diffuse_irradiance is incident irradiance before albedo and 1/pi.
        specular_radiance is outgoing reflected radiance including the material
        response. Denoised layers and scene_color use the full camera pixel grid.
        Raw guides can be supplied as zero-argument texture providers. A consumer
        materializes a guide once by sampling its name on the raw pass result.
        """
        if scene_color is None:
            for requested, name in ((self.diffuse, "diffuse_irradiance"), (self.specular, "specular_radiance")):
                if requested and name not in (denoised or {}):
                    raise ValueError(f"{self.name} did not publish requested lighting layer {name!r}")
        else:
            if not self.graph._owns_texture(scene_color):
                raise ValueError(f"{self.name}: scene_color belongs to another render graph")
            _require_camera_signal("scene_color", scene_color)
        for name, value in (denoised or {}).items():
            if isinstance(value, TextureHandle):
                _require_camera_signal(name, value)
        return RayTracingResult(self._publish_layers("raw", raw), self._publish_layers("denoised", denoised), scene_color)


def _require_camera_signal(name, texture):
    if (texture.format not in (Format.RGBA16_SFLOAT, Format.RGBA32_SFLOAT) or texture.samples != 1
            or texture.size is not None or texture.size_divisor != 0):
        raise ValueError(f"{name} requires a single-sample, camera-sized linear RGBA floating-point texture")


class RayTracer:
    """Subclass build to author any world representation, estimator, or neural renderer."""

    def build(self, ctx: RayTracingContext) -> RayTracingResult:
        raise NotImplementedError
