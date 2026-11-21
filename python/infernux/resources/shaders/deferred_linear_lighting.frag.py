import infernux.shader as shader
from infernux.shader.lib.ambient_visibility import adjust_diffuse_lighting, ambient_occlusion_factor
from infernux.shader.lib.deferred_surface import decode_deferred_surface, deferred_raster_context


class DeferredLinearLighting(shader.fullscreen):
    shader_id = "infernux/internal/deferred_linear_lighting"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    diffuse_irradiance = shader.input("diffuse_irradiance", shader.Color2D)
    ambient_visibility = shader.input("ambient_visibility", shader.Color2D)
    diffuse_intensity = shader.parameter.float(default=1.0)
    diffuse_saturation = shader.parameter.float(default=1.0)
    ssao_strength = shader.parameter.float(default=0.0)
    ssao_contrast = shader.parameter.float(default=1.0)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        depth = shader.load(self.depth_tex, pixel, 0)
        packed = shader.load(self.surface_tex, pixel, 0)
        metadata = self.surface_records[packed.x * 3]
        if depth <= 0.0 or packed.x == 0 or metadata.y == 0:
            return shader.vec4(0.0)
        frame = shader.load(self.normal_tex, pixel, 0)
        surface = decode_deferred_surface(shader.load(self.albedo_tex, pixel, 0), frame,
                                          shader.load(self.emission_tex, pixel, 0), packed.y)
        shading = deferred_raster_context(ctx.uv, ctx.position.xy, depth, surface.normal_ws, metadata, self.depth_tex)
        diffuse = shader.load(self.diffuse_irradiance, pixel, 0)
        diffuse = adjust_diffuse_lighting(diffuse, self.diffuse_intensity, self.diffuse_saturation)
        surface.occlusion *= ambient_occlusion_factor(
            self.ambient_visibility, pixel, self.ssao_strength, self.ssao_contrast)
        return shader.vec4(shader.lighting._shade_ray(surface, shading, metadata.y, diffuse, shader.vec4(0.0)), 1.0)
