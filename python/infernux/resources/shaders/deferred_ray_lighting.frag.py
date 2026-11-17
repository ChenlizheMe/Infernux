import infernux.shader as shader
from infernux.shader.lib.deferred_surface import decode_deferred_surface, deferred_shading_context
from infernux.shader.lib.ray_signal import resolve_ray_spatial


class DeferredRayLighting(shader.fullscreen):
    shader_id = "infernux/internal/deferred_ray_lighting"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    primary_tex = shader.input("ray_primary_surface", shader.UInt2D)
    ray_diffuse = shader.input("ray_diffuse", shader.UInt2D)
    ray_specular = shader.input("ray_specular", shader.UInt2D)
    diffuse_enabled = shader.parameter.bool(default=False)
    specular_enabled = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        depth = shader.load(self.depth_tex, pixel, 0)
        packed = shader.load(self.surface_tex, pixel, 0)
        base = shader.load(self.albedo_tex, pixel, 0)
        frame = shader.load(self.normal_tex, pixel, 0)
        emission = shader.load(self.emission_tex, pixel, 0)
        metadata = self.surface_records[packed.x * 3]
        surface = decode_deferred_surface(base, frame, emission, packed.y)
        context = deferred_shading_context(ctx.uv, ctx.position.xy, depth, surface.normal_ws, metadata)
        if depth <= 0.0 or packed.x == 0 or metadata.y == 0:
            return shader.vec4(0.0)
        diffuse = shader.vec4(0.0)
        if self.diffuse_enabled:
            diffuse = resolve_ray_spatial(self.ray_diffuse, self.primary_tex, self.normal_tex, pixel, packed.x,
                                          depth, surface.normal_ws, frame.z, False)
        specular = shader.vec4(0.0)
        if self.specular_enabled:
            specular = resolve_ray_spatial(self.ray_specular, self.primary_tex, self.normal_tex, pixel, packed.x,
                                           depth, surface.normal_ws, frame.z, True)
        return shader.vec4(shader.lighting._shade_ray(surface, context, metadata.y, diffuse, specular), 1.0)
