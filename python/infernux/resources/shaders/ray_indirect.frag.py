import infernux.shader as shader
from infernux.shader.lib.deferred_surface import decode_deferred_surface, deferred_shading_context
from infernux.shader.lib.surface import decode_octahedral_normal


class RayIndirect(shader.fullscreen):
    shader_id = "infernux/internal/ray_indirect"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    primary = shader.input("ray_primary_surface", shader.UInt2D)
    specular = shader.parameter.bool(default=False)
    temporal = shader.parameter.bool(default=True)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        pixel = shader.ivec2(ctx.position.xy)
        depth = shader.load(self.depth_tex, pixel, 0)
        packed = shader.load(self.surface_tex, pixel, 0)
        base = shader.load(self.albedo_tex, pixel, 0)
        frame = shader.load(self.normal_tex, pixel, 0)
        emission = shader.load(self.emission_tex, pixel, 0)
        primary = shader.load(self.primary, pixel, 0)
        metadata = self.surface_records[packed.x * 3]
        surface = decode_deferred_surface(base, frame, emission, packed.y)
        context = deferred_shading_context(ctx.uv, ctx.position.xy, depth, surface.normal_ws, metadata)
        # The shared roughness policy derives variance before rejecting uncovered lanes.
        if depth <= 0.0 or primary.x == 0:
            return shader.uvec4(0)
        geometric_normal = decode_octahedral_normal(shader.unpack_half_2x16(primary.z))
        return shader.ray._indirect_signal(
            surface, context, geometric_normal, self.surface_records, metadata.y, self.specular, self.temporal,
        )
