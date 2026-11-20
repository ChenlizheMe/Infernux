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


_CHANNELS = ("red", "green", "blue")


def _sh_textures(graph, name, divisor):
    return tuple(graph.create_texture(f"{name}/{channel}", format=Format.RGBA16_SFLOAT,
                                      size_divisor=divisor, samples=1) for channel in _CHANNELS)


def _sh_inputs(textures):
    return dict(zip(("red_sh", "green_sh", "blue_sh"), textures))


def _write_slots(render_pass, textures):
    for slot, texture in enumerate(textures):
        render_pass.write_color(texture, slot=slot)


def _read_observations(ctx, guides, observations, *, full_resolution=False):
    stage = "resolved" if full_resolution else "raw"
    output = ctx.graph.create_texture(f"{ctx.name}/diffuse/{stage}", format=Format.RGBA32_SFLOAT,
                                      size_divisor=0 if full_resolution else observations.size_divisor, samples=1)
    with ctx.graph.add_pass(f"Ray Diffuse {stage.title()} Irradiance") as p:
        p.set_textures({"observations": observations, "primary": guides.primary,
                        "frame": guides.surface.sample("surface_frame")})
        p.set_param("full_resolution", full_resolution)
        p.set_param("temporal", ctx.reconstruct)
        p.write_color(output)
        p.fullscreen_quad("infernux/internal/ray_diffuse_raw_irradiance")
    return output


def _resolve(ctx, guides, coefficients, state):
    graph = ctx.graph
    output = graph.create_texture(f"{ctx.name}/diffuse/resolved", format=Format.RGBA16_SFLOAT, samples=1)
    with graph.add_pass("Ray Diffuse Resolved Irradiance") as p:
        p.set_textures({**_sh_inputs(coefficients), "state": state, "primary": guides.primary,
                        "frame": guides.surface.sample("surface_frame")})
        p.write_color(output)
        p.fullscreen_quad("infernux/internal/ray_diffuse_resolve")
    return output


class DiffuseObservation(RayTracingSignal):
    def __init__(self, ctx, guides, packet):
        super().__init__(partial(_read_observations, ctx, guides, packet))
        self.witness = packet


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
    packet = graph.create_texture(f"{stem}/observation", format=Format.RGBA32_UINT,
                                  size_divisor=divisor, samples=1)
    with graph.add_pass("Ray Diffuse Irradiance") as p:
        bind_ray_world(p, ctx.world.materials)
        p.set_textures({"primary": guides.primary, "screen_radiance": screen_radiance, "screen_depth": screen_depth})
        p.set_buffer("surface_records", guides.surface.sample("surface_records"))
        p.set_buffer("tail_cache", graph._world_ray_tail_read())
        p.set_param("divisor", divisor or 1)
        p.set_param("temporal", ctx.reconstruct)
        p.write_color(packet)
        p.fullscreen_quad("infernux/internal/ray_diffuse_observation")
    return DiffuseObservation(ctx, guides, packet)


def reconstruct_diffuse(ctx, guides, observation):
    if not ctx.reconstruct:
        return RayTracingSignal(partial(_read_observations, ctx, guides, observation.witness, full_resolution=True))

    graph = ctx.graph
    divisor = _GRIDS[ctx.quality][1]
    stem = f"{ctx.name}/diffuse"
    old, filtered = zip(*(ctx.view.history(f"diffuse/{key}", format=Format.RGBA16_SFLOAT, size_divisor=divisor,
                                           respond_to_lighting=True) for key in _CHANNELS))
    old_state, state = ctx.view.history("diffuse/state", format=Format.RG16_SFLOAT, size_divisor=divisor,
                                       respond_to_lighting=True)
    fresh = _sh_textures(graph, f"{stem}/gathered", divisor)
    fresh_state = graph.create_texture(f"{stem}/gathered/state", format=Format.RG16_SFLOAT,
                                       size_divisor=divisor, samples=1)
    reprojection = _sh_textures(graph, f"{stem}/reprojected", divisor)
    visibility = ctx.ambient_visibility or guides.surface.sample("surface_frame")
    with graph.add_pass("Ray Diffuse SH Gather") as p:
        p.set_textures({"observations": observation.witness, "primary": guides.primary,
                        "ambient_visibility": visibility})
        p.set_param("ssao_strength", ctx.ssao_strength)
        p.set_param("divisor", divisor)
        _write_slots(p, (*fresh, fresh_state))
        p.fullscreen_quad("infernux/internal/ray_diffuse_gather")
    with graph.add_pass("Ray Diffuse Reconstruction") as p:
        p.set_textures({**dict(zip(("fresh_red", "fresh_green", "fresh_blue"), fresh)),
                        **dict(zip(("old_red", "old_green", "old_blue"), old)),
                        "fresh_state": fresh_state, "old_state": old_state, "primary": guides.primary,
                        "previous_primary": guides.previous_primary, "motion": guides.motion})
        p.set_buffer("geometry", ctx.world.geometry)
        _write_slots(p, (*reprojection, state))
        p.fullscreen_quad("infernux/internal/ray_diffuse_history")
    with graph.add_pass("Ray Diffuse Recurrent Blur") as p:
        p.set_textures({**_sh_inputs(reprojection), "state": state, "primary": guides.primary,
                        "visibility": visibility})
        p.set_param("ssao_strength", ctx.ssao_strength)
        _write_slots(p, filtered)
        p.fullscreen_quad("infernux/internal/ray_diffuse_blur")
    return RayTracingSignal(partial(_resolve, ctx, guides, filtered, state))
