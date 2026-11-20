"""Directional irradiance observations and geometry-guided reconstruction."""

from functools import partial

from infernux.rendergraph.graph import Format
from infernux.renderstack._pipeline_common import bind_deferred_surface, bind_ray_world, deferred_surface_inputs
from infernux.renderstack.ray_tracer import RayTracingQuality, RayTracingSignal


# Trace-grid and reconstruction divisors. Each trace site emits one direction.
_GRIDS = {
    RayTracingQuality.LIGHTWEIGHT: (8, 4),
    RayTracingQuality.STANDARD: (4, 4),
    RayTracingQuality.HIGH: (4, 2),
}


def _pair(graph, name, divisor):
    return tuple(graph.create_texture(f"{name}/{key}", format=Format.RGBA32_UINT,
                                      size_divisor=divisor, samples=1) for key in ("energy", "directions"))


def _write_moments(render_pass, moments):
    energy, directions = moments
    render_pass.write_color(energy)
    render_pass.write_color(directions, slot=1)


def _resolve(ctx, guides, moments, *, raw):
    energy, directions = moments
    stage = "raw" if raw else "resolved"
    output = ctx.graph.create_texture(f"{ctx.name}/diffuse/{stage}", format=Format.RGBA32_SFLOAT if raw else Format.RGBA16_SFLOAT,
                                      size_divisor=energy.size_divisor if raw else 0, samples=1)
    with ctx.graph.add_pass(f"Ray Diffuse {stage.title()} Irradiance") as p:
        p.set_textures({"coefficients": energy, "directions": directions, "primary": guides.primary,
                        "frame": guides.surface.sample("surface_frame")})
        p.set_param("raw", raw)
        p.write_color(output)
        p.fullscreen_quad("infernux/internal/ray_diffuse_resolve")
    return output


class DiffuseObservation(RayTracingSignal):
    def __init__(self, ctx, guides, moments):
        super().__init__(partial(_resolve, ctx, guides, moments, raw=True))
        self.moments = moments
        # The directional packet retains the distance along the first fresh ray.
        self.witness = moments[1]


def trace_grid(quality):
    return _GRIDS[quality][0]


def trace_diffuse(ctx, guides, divisor):
    graph = ctx.graph
    stem = f"{ctx.name}/diffuse"
    screen_radiance = graph.create_texture(f"{stem}/screen_radiance", format=Format.RGBA16_SFLOAT,
                                           size_divisor=2, samples=1)
    screen_depth = graph.create_texture(f"{stem}/screen_depth", format=Format.R32_SFLOAT,
                                        size_divisor=2, samples=1)
    with graph.add_pass("Ray Diffuse Screen Proxy") as p:
        bind_deferred_surface(p, deferred_surface_inputs(guides.surface))
        p.write_color(screen_radiance)
        p.write_color(screen_depth, slot=1)
        p.fullscreen_quad("infernux/internal/ray_screen_radiance")
    moments = _pair(graph, f"{stem}/observation", divisor)
    with graph.add_pass("Ray Diffuse Irradiance") as p:
        bind_ray_world(p, ctx.world.materials)
        p.set_textures({"primary": guides.primary, "screen_radiance": screen_radiance, "screen_depth": screen_depth})
        p.set_buffer("surface_records", guides.surface.sample("surface_records"))
        p.set_buffer("tail_cache", graph._world_ray_tail_read())
        p.set_param("divisor", divisor or 1)
        p.set_param("temporal", ctx.reconstruct)
        _write_moments(p, moments)
        p.fullscreen_quad("infernux/internal/ray_diffuse_observation")
    return DiffuseObservation(ctx, guides, moments)


def reconstruct_diffuse(ctx, guides, observation):
    moments = observation.moments
    if ctx.reconstruct:
        graph = ctx.graph
        divisor = _GRIDS[ctx.quality][1]
        history = [ctx.view.history(f"diffuse/{key}", format=Format.RGBA32_UINT, size_divisor=divisor,
                                    respond_to_lighting=True) for key in ("energy", "directions")]
        (old_energy, old_axes), filtered = zip(*history)
        reprojection = _pair(graph, f"{ctx.name}/diffuse/reprojected", divisor)
        visibility = ctx.ambient_visibility or guides.surface.sample("surface_frame")
        with graph.add_pass("Ray Diffuse Reconstruction") as p:
            p.set_textures({"fresh_dc": moments[0], "fresh_axes": moments[1], "old_dc": old_energy,
                            "old_axes": old_axes, "primary": guides.primary, "previous_primary": guides.previous_primary,
                            "motion": guides.motion, "ambient_visibility": visibility})
            p.set_buffer("geometry", ctx.world.geometry)
            p.set_param("ssao_strength", ctx.ssao_strength)
            _write_moments(p, reprojection)
            p.fullscreen_quad("infernux/internal/ray_diffuse_history")
        with graph.add_pass("Ray Diffuse Recurrent Blur") as p:
            p.set_textures({"coefficients": reprojection[0], "directions": reprojection[1],
                            "primary": guides.primary, "visibility": visibility})
            p.set_param("ssao_strength", ctx.ssao_strength)
            _write_moments(p, filtered)
            p.fullscreen_quad("infernux/internal/ray_diffuse_blur")
        moments = filtered
    return RayTracingSignal(partial(_resolve, ctx, guides, moments, raw=False))
