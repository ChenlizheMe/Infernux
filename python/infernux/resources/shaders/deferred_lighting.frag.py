import infernux.shader as shader
from infernux.shader.lib.ambient_visibility import ambient_occlusion_factor
from infernux.shader.lib.deferred_surface import decode_deferred_surface, deferred_raster_context


class DeferredLighting(shader.fullscreen):
    shader_id = "infernux/internal/deferred_lighting"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    ambient_visibility = shader.input("ambient_visibility", shader.Color2D)
    ssao_strength = shader.parameter.float(default=0.0)
    ssao_contrast = shader.parameter.float(default=1.0)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        depth = shader.load(self.depth_tex, pixel, 0)
        packed = shader.load(self.surface_tex, pixel, 0)
        metadata = self.surface_records[packed.x * 3]
        if depth <= 0.0 or packed.x == 0 or metadata.y == 0:
            return shader.vec4(0.0)
        base = shader.load(self.albedo_tex, pixel, 0)
        frame = shader.load(self.normal_tex, pixel, 0)
        emission = shader.load(self.emission_tex, pixel, 0)
        surface = decode_deferred_surface(base, frame, emission, packed.y)
        surface.occlusion *= ambient_occlusion_factor(
            self.ambient_visibility, pixel, self.ssao_strength, self.ssao_contrast)
        context = deferred_raster_context(ctx.uv, ctx.position.xy, depth, surface.normal_ws, metadata, self.depth_tex)
        return shader.vec4(shader.lighting.shade(surface, context, metadata.y), 1.0)
