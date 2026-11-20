"""Directional irradiance observations and geometry-guided reconstruction."""

from functools import partial

from infernux.rendergraph.graph import Format
from infernux.renderstack._pipeline_common import bind_ray_world
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
    def __init__(self, ctx, guides, packet, reprojection, visibility, half_geometry):
        super().__init__(partial(_read_observations, ctx, guides, packet))
        self.witness = packet
        self.reprojection = reprojection
        self.visibility = visibility
        self.half_geometry = half_geometry


def trace_grid(quality):
    return _GRIDS[quality][0]


def _add_diffuse_guide(graph, name, depth, frame, *, temporal=True):
    output = graph.create_texture(name, format=Format.RG32_UINT, size_divisor=2, samples=1)
    with graph.add_pass("Ray Diffuse Half Geometry") as p:
        p.set_textures({"depth": depth, "frame": frame})
        p.set_param("temporal", temporal)
        p.write_color(output)
        p.fullscreen_quad("infernux/internal/ray_diffuse_guide")
    return output


def provide_diffuse_guide(context):
    return _add_diffuse_guide(context.graph, f"_result/{context.source}/ray_diffuse_guide",
                              context.sample("depth"), context.sample("surface_frame"))


def provide_surface_reprojection(context):
    graph = context.graph
    output = graph.create_texture(f"_result/{context.source}/ray_surface_reprojection",
                                  format=Format.RG32_UINT, samples=1)
    with graph.add_pass("Ray Diffuse Surface Reprojection") as p:
        p.set_textures({"primary": context.sample("ray_primary_surface"),
                        "previous_primary": context.sample("ray_previous_primary_surface"),
                        "motion": context.sample("ray_surface_motion")})
        p.write_color(output)
        p.fullscreen_quad("infernux/internal/ray_diffuse_reprojection")
    return output


def provide_ambient_visibility(context):
    graph = context.graph
    stem = f"_result/{context.source}/gtao"
    guide = context.sample("ray_diffuse_guide")
    reprojection = context.has("ray_surface_reprojection")
    if reprojection:
        motion = context.sample("ray_surface_reprojection")
    else:
        motion = graph.get_pass_result(context.source).sample("motion")
    half = graph.create_texture(f"{stem}/half", format=Format.R16_SFLOAT, size_divisor=2, samples=1)
    full = graph.create_texture(f"{stem}/full", format=Format.R16_SFLOAT, samples=1)
    previous, current = graph.create_temporal_history(f"{stem}/history", format=Format.R8_UNORM)
    with graph.add_pass(f"{context.source}/GTAO") as p:
        p.set_texture("guide", guide)
        p.write_color(half)
        p.fullscreen_quad("infernux/internal/gtao")
    with graph.add_pass(f"{context.source}/GTAO Upsample") as p:
        p.set_textures({"visibility": half, "guide": guide, "depth": context.sample("depth"),
                        "frame": context.sample("surface_frame")})
        p.write_color(full)
        p.fullscreen_quad("infernux/internal/gtao_upsample")
    with graph.add_pass(f"{context.source}/GTAO Temporal") as p:
        p.set_textures({"visibility": full, "history": previous, "reprojection": motion if reprojection else guide,
                        "motion": full if reprojection else motion})
        p.set_param("use_reprojection", reprojection)
        p.write_color(current)
        p.fullscreen_quad("infernux/internal/gtao_temporal")
    return current


def trace_diffuse(ctx, guides, divisor):
    graph = ctx.graph
    stem = f"{ctx.name}/diffuse"
    previous_radiance, _ = graph._scene_radiance_history()
    geometry = ctx.geometry or guides.surface
    if ctx.reconstruct:
        screen_depth = geometry.sample("ray_diffuse_guide")
        reprojection = geometry.sample("ray_surface_reprojection")
        visibility = ctx.ambient_visibility
    else:
        reprojection = guides.primary
        screen_depth = _add_diffuse_guide(graph, f"{stem}/guide", guides.surface.sample("depth"),
                                          guides.surface.sample("surface_frame"), temporal=False)
        visibility = None
    packet = graph.create_texture(f"{stem}/observation", format=Format.RGBA32_UINT,
                                  size_divisor=divisor, samples=1)
    with graph.add_pass("Ray Diffuse Irradiance") as p:
        bind_ray_world(p, ctx.world.materials)
        p.set_textures({"primary": guides.primary, "screen_depth": screen_depth,
                        "previous_radiance": previous_radiance, "reprojection": reprojection})
        p.set_buffer("surface_records", guides.surface.sample("surface_records"))
        p.set_buffer("tail_cache", graph._world_ray_tail_read())
        p.set_param("divisor", divisor or 1)
        p.set_param("temporal", ctx.reconstruct)
        p.write_color(packet)
        p.fullscreen_quad("infernux/internal/ray_diffuse_observation")
    return DiffuseObservation(ctx, guides, packet, reprojection, visibility, screen_depth)


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
    reprojection = _sh_textures(graph, f"{stem}/reprojected", divisor)
    reprojected_state = graph.create_texture(f"{stem}/reprojected/state", format=Format.RG16_SFLOAT,
                                             size_divisor=divisor, samples=1)
    filled = _sh_textures(graph, f"{stem}/filled", divisor)
    visibility = observation.visibility or guides.surface.sample("surface_frame")
    with graph.add_pass("Ray Diffuse SH Gather") as p:
        p.set_textures({"observations": observation.witness, "primary": guides.primary,
                        "guide": observation.half_geometry, "ambient_visibility": visibility})
        p.set_param("use_visibility", observation.visibility is not None)
        p.set_param("divisor", divisor)
        _write_slots(p, fresh)
        p.fullscreen_quad("infernux/internal/ray_diffuse_gather")
    with graph.add_pass("Ray Diffuse Reconstruction") as p:
        p.set_textures({**dict(zip(("fresh_red", "fresh_green", "fresh_blue"), fresh)),
                        **dict(zip(("old_red", "old_green", "old_blue"), old)),
                        "old_state": old_state, "primary": guides.primary, "reprojection": observation.reprojection})
        p.set_buffer("geometry", ctx.world.geometry)
        _write_slots(p, (*reprojection, reprojected_state))
        p.fullscreen_quad("infernux/internal/ray_diffuse_history")
    with graph.add_pass("Ray Diffuse Fill Disocclusion") as p:
        p.set_textures({**_sh_inputs(reprojection), "state": reprojected_state, "old_state": old_state,
                        "primary": guides.primary})
        _write_slots(p, (*filled, state))
        p.fullscreen_quad("infernux/internal/ray_diffuse_fill")
    with graph.add_pass("Ray Diffuse Recurrent Blur") as p:
        p.set_textures({**_sh_inputs(filled), "state": state, "primary": guides.primary,
                        "guide": observation.half_geometry, "visibility": visibility})
        p.set_param("use_visibility", observation.visibility is not None)
        _write_slots(p, filtered)
        p.fullscreen_quad("infernux/internal/ray_diffuse_blur")
    return RayTracingSignal(partial(_resolve, ctx, guides, filtered, state))
