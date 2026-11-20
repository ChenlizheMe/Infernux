import infernux.shader as shader
from infernux.shader.lib.deferred_surface import deferred_raster_context, load_deferred_surface
from infernux.shader.lib.ray_signal import ray_grid_pixel


class RayScreenRadiance(shader.fullscreen):
    shader_id = "infernux/internal/ray_screen_radiance"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])

    def main(self, ctx: shader.fragment_input) -> tuple[shader.vec4, shader.vec4]:
        cell = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.depth_tex, 0))
        extent = shader.max(shader.ivec2(full / 2), shader.ivec2(1))
        lo = shader.ivec2(shader.floor(shader.vec2(cell * full) / shader.vec2(extent)))
        hi = shader.ivec2(shader.ceil(shader.vec2((cell + 1) * full) / shader.vec2(extent)))
        closest = 0.0
        for y in range(lo.y, hi.y):
            for x in range(lo.x, hi.x):
                closest = shader.max(closest, shader.load(self.depth_tex, shader.ivec2(x, y), 0))

        pixel = ray_grid_pixel(cell, extent, full)
        depth = shader.load(self.depth_tex, pixel, 0)
        packed = shader.load(self.surface_tex, pixel, 0)
        surface = load_deferred_surface(self.albedo_tex, self.normal_tex, self.emission_tex, self.surface_tex, pixel)
        radiance = shader.vec4(0.0)
        # The camera-view sample approximates outgoing diffuse radiance. Emitters
        # and view-dependent surfaces keep their world estimator and MIS weights.
        diffuse = (surface.metallic < 0.15 and surface.smoothness < 0.6
                   and shader.dot(surface.emission, surface.emission) == 0.0)
        if depth > 0.0 and packed.x != 0 and diffuse:
            metadata = self.surface_records[packed.x * 3]
            raster = shader.vec2(pixel) + shader.vec2(0.5)
            context = deferred_raster_context(raster / shader.vec2(full), raster, depth, surface.normal_ws,
                                              metadata, self.depth_tex)
            surface.specular_highlights = 0.0
            # This pass supplies current direct radiance. Mark the empty indirect
            # inputs as valid so PBR doesn't inject an unoccluded environment probe.
            no_indirect = shader.vec4(0.0, 0.0, 0.0, 1.0)
            rgb = shader.lighting._shade_ray(surface, context, metadata.y, no_indirect, no_indirect)
            radiance = shader.vec4(rgb, 1.0)
        return (radiance, shader.vec4(closest, 0.0, 0.0, 1.0))
