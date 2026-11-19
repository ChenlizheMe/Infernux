"""Built-in software ray transport and per-view reconstruction stages."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from infernux.lib import _Infernux
from infernux.rendergraph.graph import Format, TextureHandle
from infernux.renderstack._pipeline_common import (
    COLOR_TEXTURE, DEPTH_TEXTURE, SURFACE_RECORDS_BUFFER,
    bind_deferred_material, deferred_surface_inputs, draw_deferred_geometry, transparent_queue_range,
)
from infernux.renderstack.pass_result import PassResult
from infernux.renderstack.ray_tracer import RayTracer, RayTracingContext, RayTracingQuality, RayTracingSignal


_DIFFUSE_DIVISOR = {RayTracingQuality.LIGHTWEIGHT: 4, RayTracingQuality.STANDARD: 2, RayTracingQuality.HIGH: 0}


@dataclass(frozen=True)
class RaySurfaceGuides:
    """Current material surface, primary identity, previous primary and motion replay."""

    surface: PassResult
    primary: TextureHandle
    previous_primary: TextureHandle | None
    motion: TextureHandle | None


class _PackedSignal(RayTracingSignal):
    def __init__(self, packet, resolve, *, local_estimate=None):
        super().__init__(resolve)
        self.packet = packet
        self.local_estimate = packet if local_estimate is None else local_estimate


def _linear_raw(graph, packet):
    # The packed estimator keeps the FP32 exponent. Bright samples must stay
    # finite when a project fetches them as linear values for its own filter.
    output = graph.create_texture(f"{packet.name}/raw", format=Format.RGBA32_SFLOAT,
                                  size_divisor=packet.size_divisor, samples=1)
    with graph.add_pass(f"Decode {packet.name}") as p:
        p.set_texture("signal", packet)
        p.write_color(output)
        p.fullscreen_quad("infernux/internal/ray_raw_signal")
    return output


def _bind_transport(render_pass, ctx, guides, specular):
    bind_deferred_material(render_pass, deferred_surface_inputs(guides.surface))
    render_pass.set_texture("primary", guides.primary)
    render_pass.set_buffer("tail_cache", ctx.graph._world_ray_tail_read())
    _bind_world(render_pass, ctx.world.materials)
    render_pass.set_param("specular", specular)


def _hit_distance(graph, packet):
    """Distance in world units along the fresh secondary ray, zero without a finite hit."""
    output = graph.create_texture(f"{packet.name}/hit_distance", format=Format.R32_SFLOAT,
                                  size_divisor=packet.size_divisor, samples=1)
    with graph.add_pass(f"Decode {packet.name} hit distance") as p:
        p.set_texture("signal", packet)
        p.write_color(output)
        p.fullscreen_quad("infernux/internal/ray_hit_distance")
    return output


def _filter_diffuse(ctx, guides, source, target, stride):
    frame = guides.surface.sample("surface_frame")
    visibility = ctx.ambient_visibility
    with ctx.graph.add_pass(f"Ray Diffuse Spatial {stride}") as p:
        p.set_textures({"signal": source, "primary": guides.primary, "frame": frame,
                        "ambient_visibility": visibility or frame})
        p.set_param("ssao_strength", ctx.ssao_strength)
        p.write_color(target)
        p.set_param("stride", stride)
        p.fullscreen_quad("infernux/internal/ray_diffuse_filter")


def _linear_resolve(graph, packet, guides, specular):
    name = "specular_radiance" if specular else "diffuse_irradiance"
    output = graph.create_texture(f"{packet.name}/resolved", format=Format.RGBA32_SFLOAT, samples=1)
    with graph.add_pass(f"Resolve {name}") as p:
        p.set_textures({"signal": packet, "primary": guides.primary, "frame": guides.surface.sample("surface_frame"),
                        "surface_tex": guides.surface.sample("surface"), "depth_tex": guides.surface.sample(DEPTH_TEXTURE)})
        p.set_param("specular", specular)
        p.write_color(output)
        p.fullscreen_quad("infernux/internal/resolve_ray_signal")
    return output


class DefaultRayTracer(RayTracer):
    """Raster primary surfaces, software world queries and temporal reconstruction."""

    def prepare_guides(self, ctx: RayTracingContext) -> RaySurfaceGuides:
        surface = ctx.surface
        if surface is None:
            raise ValueError("DefaultRayTracer requires a material surface result")
        geometry = ctx.geometry or surface
        if any(surface.sample(key) is not geometry.sample(key) for key in (DEPTH_TEXTURE, "surface")):
            raise ValueError("DefaultRayTracer requires depth and surface identity from the same geometry")
        previous = geometry.sample("ray_previous_primary_surface") if ctx.reconstruct else None
        primary = geometry.sample("ray_primary_surface")
        motion = geometry.sample("ray_surface_motion") if ctx.reconstruct else None
        return RaySurfaceGuides(surface, primary, previous, motion)

    def trace(self, ctx, guides, *, specular, divisor):
        """Generate one lobe's samples and its hit witnesses on the native trace grid."""
        graph = ctx.graph
        name = "ray_specular" if specular else "ray_diffuse"
        stem = f"{ctx.name}/{name}"
        output = graph.create_texture(f"{stem}_packet", format=Format.RGBA32_UINT, size_divisor=divisor, samples=1)
        label = "Ray Specular Radiance" if specular else "Ray Diffuse Irradiance"
        if ctx.reconstruct:
            # These keys generate paths in the current scene, including after a light changes.
            # Only the temporal selection recurs. The spatial result is consumed once.
            previous, current = graph.create_temporal_history(
                f"{stem}/paths", format=Format.RGBA32_UINT, size_divisor=divisor)
            estimates = graph.create_texture(f"{stem}/temporal_estimate", format=Format.RGBA32_UINT,
                                             size_divisor=divisor, samples=1)
            with graph.add_pass(f"{label} Path Candidates") as p:
                _bind_transport(p, ctx, guides, specular)
                p.set_textures({"motion": guides.motion, "previous_surface": guides.previous_primary,
                                "previous_paths": previous})
                p.write_color(estimates)
                p.write_color(current, slot=1)
                p.fullscreen_quad("infernux/internal/ray_path_temporal")
            with graph.add_pass(label) as p:
                _bind_transport(p, ctx, guides, specular)
                p.set_textures({"paths": current, "estimates": estimates})
                p.write_color(output)
                p.fullscreen_quad("infernux/internal/ray_path_spatial")
        else:
            with graph.add_pass(label) as p:
                _bind_transport(p, ctx, guides, specular)
                p.set_param("temporal", False)
                p.set_param("resolution_divisor", divisor or 1)
                p.write_color(output)
                p.fullscreen_quad("infernux/internal/ray_indirect")
        return _PackedSignal(output, partial(_linear_raw, graph, output),
                             local_estimate=estimates if ctx.reconstruct and specular else output)

    def reconstruct_diffuse(self, ctx, raw, guides):
        """Return full-resolution irradiance. Custom filters read raw.radiance on demand."""
        return self._reconstruct(ctx, raw, guides, specular=False)

    def reconstruct_specular(self, ctx, raw, guides):
        """Return outgoing reflected radiance, with positive alpha on supported surfaces."""
        return self._reconstruct(ctx, raw, guides, specular=True)

    def _reconstruct(self, ctx, raw, guides, *, specular):
        graph = ctx.graph
        packet = raw.packet
        if ctx.reconstruct:
            name = "specular" if specular else "diffuse"
            divisor = packet.size_divisor
            if not specular:
                filtered = graph.create_texture(f"{ctx.name}/{name}/filtered", format=Format.RGBA32_UINT,
                                                size_divisor=divisor, samples=1)
                _filter_diffuse(ctx, guides, packet, filtered, 1)
                packet = filtered
            previous, output = ctx.view.history(name, format=Format.RGBA32_UINT, size_divisor=divisor,
                                                respond_to_lighting=not specular)
            label = "Ray Specular Reconstruction" if specular else "Ray Diffuse Reconstruction"
            with graph.add_pass(label) as p:
                p.set_textures({"signal": packet, "local_estimate": raw.local_estimate if specular else packet,
                                "primary": guides.primary, "motion": guides.motion,
                                "previous_signal": previous, "previous_primary": guides.previous_primary})
                p.set_buffer("geometry", ctx.world.geometry)
                p.write_color(output)
                p.set_param("specular", specular)
                p.fullscreen_quad("infernux/internal/ray_reconstruct")
            packet = output
            if not specular:
                intermediate = graph.create_texture(f"{ctx.name}/{name}/intermediate", format=Format.RGBA32_UINT,
                                                    size_divisor=divisor, samples=1)
                _filter_diffuse(ctx, guides, packet, intermediate, 2)
                _filter_diffuse(ctx, guides, intermediate, filtered, 4)
                packet = filtered
        return _PackedSignal(packet, partial(_linear_resolve, graph, packet, guides, specular))

    def build(self, ctx: RayTracingContext):
        lobes = []
        if ctx.diffuse:
            lobes.append(("diffuse_irradiance", False, _DIFFUSE_DIVISOR[ctx.quality], self.reconstruct_diffuse))
        if ctx.specular:
            lobes.append(("specular_radiance", True, 0, self.reconstruct_specular))
        guides = self.prepare_guides(ctx) if lobes else None
        raw = {name: self.trace(ctx, guides, specular=specular, divisor=divisor)
               for name, specular, divisor, _ in lobes}
        denoised = {name: reconstruct(ctx, raw[name], guides) for name, _, _, reconstruct in lobes}
        for name, specular, _, _ in lobes:
            sample = raw[name]
            if isinstance(sample, _PackedSignal):
                guide = "specular_hit_distance" if specular else "diffuse_hit_distance"
                raw[guide] = partial(_hit_distance, ctx.graph, sample.packet)
        frame = ctx.publish(raw=raw, denoised=denoised)
        frame._packets = {name: signal.packet for name, signal in denoised.items() if isinstance(signal, _PackedSignal)}
        if frame._packets:
            frame._primary = guides.primary
        if ctx.transparency:
            frame._after_sky = partial(self.finish_transparency, ctx, guides)
        elif guides is not None:
            _update_ray_tail(ctx, primary=guides.primary, thin=None)
        return frame

    def finish_transparency(self, ctx, guides, surface):
        surface = surface or ctx.surface
        capture = _capture_thin_transparency(ctx.graph, surface.sample(DEPTH_TEXTURE))
        _update_ray_tail(ctx, primary=guides.primary if guides else None, thin=capture)
        _draw_thin_transparency(ctx.graph, capture, surface)
        return True


def _bind_world(render_pass, world):
    for key in ("world_ray_geometry", "world_ray_meshes", "ray_lit_materials"):
        render_pass.set_buffer(key, world[key])
    for key in ("ray_color_pages", "ray_data_pages"):
        render_pass.set_texture(key, world[key])


def draw_surface_replay(context):
    graph = context.graph
    stem = f"_result/{context.source}/ray_surface"
    replay = graph.create_texture(f"{stem}_witness", format=Format.RGBA32_UINT, samples=1)
    depth = graph.create_texture(f"{stem}_depth", format=Format.D32_SFLOAT, samples=1)
    with graph.add_pass(f"{context.source}/Ray Surface Replay") as render_pass:
        render_pass.write_color(replay)
        draw_deferred_geometry(render_pass, depth, context.queue_range, context.sort_mode, "ray_surface")
    context.publish("ray_surface_depth", depth)
    return replay


def provide_ray_primary_surface(context):
    graph = context.graph
    stem = f"_result/{context.source}/ray_primary"
    result = graph.get_texture(f"{stem}/write")
    if result is None:
        result = graph.create_texture(f"{stem}/witness", format=Format.RGBA32_UINT, samples=1)
    with graph.add_pass(f"{context.source}/Ray Surface Witness") as render_pass:
        render_pass.write_color(result)
        render_pass.set_textures({
            "gbuffer_surface": context.sample("surface"),
            "gbuffer_depth": context.sample(DEPTH_TEXTURE),
            "gbuffer_normal": context.sample("surface_frame"),
            "replay_surface": context.sample("ray_surface_motion"),
            "replay_depth": context.sample("ray_surface_depth"),
        })
        render_pass.fullscreen_quad("infernux/internal/ray_primary_surface")
    return result


def _capture_thin_transparency(graph, opaque_depth):
    layer = graph.create_texture("_ray_thin_layer", format=Format.RGBA32_UINT, samples=1)
    optics = graph.create_texture("_ray_thin_optics", format=Format.RGBA32_UINT, samples=1)
    depth = graph.create_texture("_ray_thin_depth", format=Format.D32_SFLOAT, samples=1)
    with graph.add_copy_pass("Thin Surface Depth") as render_pass:
        render_pass.copy_texture(opaque_depth, depth)
    with graph.add_pass("Thin Surface Capture") as render_pass:
        render_pass.write_color(layer, slot=0)
        render_pass.write_color(optics, slot=1)
        render_pass.write_depth(depth)
        render_pass.set_clear(color=(0.0, 0.0, 0.0, 0.0))
        render_pass.draw_renderers(
            queue_range=transparent_queue_range(), sort_mode="front_to_back",
            material_pass="thin_surface", material_filter="thin_sheet",
        )
    return layer, optics


def _update_ray_tail(ctx, *, primary, thin):
    graph, gbuffer, world = ctx.graph, ctx.surface, ctx.world.materials
    tail = graph._create_world_ray_tail()
    count = _Infernux._ray_tail_batch_size()
    requests = tuple(graph.create_texture(f"_ray_tail_request_{index}", format=Format.RGBA32_UINT,
                                          size=(count, 2), samples=1) for index in range(2))
    samples = graph.create_texture("_ray_tail_samples", format=Format.RGBA32_SFLOAT,
                                    size=(count, 1), samples=1)
    inputs = deferred_surface_inputs(gbuffer)
    primary = primary if primary is not None else thin[0]
    with graph.add_pass("Ray Tail Requests") as render_pass:
        render_pass._ray_tail_batch = 2
        bind_deferred_material(render_pass, inputs)
        render_pass.set_texture("primary", primary)
        render_pass.set_texture("thin_layer", thin[0] if thin else primary)
        render_pass.set_buffer("requests", tail["ray_tail_requests"])
        _bind_world(render_pass, world)
        for index, target in enumerate(requests):
            render_pass.write_color(target, slot=index)
        render_pass.set_param("diffuse_enabled", ctx.diffuse)
        render_pass.set_param("specular_enabled", ctx.specular)
        render_pass.set_param("temporal", ctx.reconstruct)
        render_pass.set_param("thin_enabled", bool(thin))
        render_pass.fullscreen_quad("infernux/internal/ray_tail_request")
    with graph.add_compute_pass("Ray Tail Prepare") as render_pass:
        render_pass._action = "ray_tail_prepare"
        render_pass._ray_tail_batch = 1
        render_pass._side_effect = True
        render_pass.set_textures({f"request_{index}": target for index, target in enumerate(requests)})
        render_pass.read_buffer(tail["ray_tail_read"])
        render_pass.read_buffer(tail["ray_tail_control"])
        render_pass.write_buffer(tail["ray_tail_write"])
        render_pass.write_buffer(tail["ray_tail_requests"])
    with graph.add_pass("Ray Tail Estimate") as render_pass:
        render_pass._ray_tail_batch = 2
        render_pass.set_buffer("requests", tail["ray_tail_requests"])
        render_pass.set_buffer("cache", tail["ray_tail_read"])
        render_pass.set_buffer("surface_records", inputs[SURFACE_RECORDS_BUFFER])
        _bind_world(render_pass, world)
        render_pass.write_color(samples)
        render_pass.fullscreen_quad("infernux/internal/ray_tail_estimate")
    with graph.add_compute_pass("Ray Tail Merge") as render_pass:
        render_pass._action = "ray_tail_merge"
        render_pass._ray_tail_batch = 1
        render_pass._side_effect = True
        render_pass.set_texture("tail_samples", samples)
        render_pass.read_buffer(tail["ray_tail_requests"])
        render_pass.read_buffer(tail["ray_tail_control"])
        render_pass.write_buffer(tail["ray_tail_write"])


def _draw_thin_transparency(graph, capture, surface):
    layer, optics = capture
    color, depth = surface.sample(COLOR_TEXTURE), surface.sample(DEPTH_TEXTURE)
    reflection = graph.create_texture("_ray_thin_reflection", format=Format.RGBA16_SFLOAT, samples=1)
    background = graph.create_texture("_ray_thin_background", format=Format.RGBA16_SFLOAT, samples=1)
    world = graph._create_world_ray_lit()
    with graph.add_pass("Thin Surface Reflection") as render_pass:
        render_pass.set_textures({"layer": layer, "optics": optics})
        render_pass.set_buffer("surface_records", surface.sample(SURFACE_RECORDS_BUFFER))
        render_pass.set_buffer("tail_cache", graph._world_ray_tail_read())
        _bind_world(render_pass, world)
        render_pass.write_color(reflection)
        render_pass.fullscreen_quad("infernux/internal/ray_thin_reflection")
    with graph.add_pass("Transparent Behind Thin Surface") as render_pass:
        render_pass.read(depth)
        render_pass.write_color(color)
        render_pass.set_texture("transparent_layer", layer)
        render_pass.draw_renderers(
            queue_range=transparent_queue_range(), sort_mode="back_to_front",
            material_pass="transparent_behind", material_filter="ordinary_transparent",
        )
    with graph.add_pass("Thin Surface Background") as render_pass:
        render_pass.set_texture("source_tex", color)
        render_pass.write_color(background)
        render_pass.fullscreen_quad("infernux/internal/fullscreen_copy")
    with graph.add_pass("Thin Surface Composite") as render_pass:
        render_pass.set_textures({"background": background, "layer": layer, "optics": optics, "reflection": reflection})
        render_pass.write_color(color)
        render_pass.fullscreen_quad("infernux/internal/thin_composite")
    with graph.add_pass("Transparent In Front Of Thin Surface") as render_pass:
        render_pass.read(depth)
        render_pass.write_color(color)
        render_pass.set_texture("transparent_layer", layer)
        render_pass.draw_renderers(
            queue_range=transparent_queue_range(), sort_mode="back_to_front",
            material_pass="transparent_front", material_filter="ordinary_transparent",
        )
