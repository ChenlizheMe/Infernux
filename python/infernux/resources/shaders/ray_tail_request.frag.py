import infernux.shader as shader
from infernux.shader.lib.camera import world_position
from infernux.shader.lib.deferred_surface import decode_deferred_surface, deferred_shading_context
from infernux.shader.lib.surface import decode_octahedral_normal


class RayTailRequest(shader.fullscreen):
    shader_id = "infernux/internal/ray_tail_request"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    primary = shader.input("ray_primary_surface", shader.UInt2D)
    thin_layer = shader.input("thin_layer", shader.UInt2D)
    requests = shader.input("ray_tail_requests", shader.Buffer[shader.uvec4])
    specular_enabled = shader.parameter.bool(default=True)
    thin_enabled = shader.parameter.bool(default=False)
    temporal = shader.parameter.bool(default=True)

    def main(self, ctx: shader.fragment_input) -> tuple[shader.uvec4, shader.uvec4]:
        # Two rows share one seed. Two integer attachments stay within 32 B/sample.
        index = shader.uint(ctx.position.x)
        if index >= self.requests[0].y:
            return (shader.uvec4(0), shader.uvec4(0))
        extent = shader.uvec2(shader.dimensions(self.primary, 0))
        pixel = shader.ivec2(shader.ray._tail_pixel(index + self.requests[1].w, extent))
        uv = (shader.vec2(pixel) + shader.vec2(0.5)) / shader.vec2(extent)
        if self.thin_enabled and (not self.specular_enabled or (index & 1) != 0):
            thin = shader.load(self.thin_layer, pixel, 0)
            if thin.x == 0:
                return (shader.uvec4(0), shader.uvec4(0))
            position = world_position(uv, shader.as_float(thin.w), shader.camera())
            normal = decode_octahedral_normal(shader.unpack_half_2x16(thin.y))
            request = shader.ray._tail_thin_request(position, normal)
        else:
            primary = shader.load(self.primary, pixel, 0)
            if primary.x == 0:
                return (shader.uvec4(0), shader.uvec4(0))
            depth = shader.as_float(primary.y)
            packed = shader.load(self.surface_tex, pixel, 0)
            base = shader.load(self.albedo_tex, pixel, 0)
            frame = shader.load(self.normal_tex, pixel, 0)
            emission = shader.load(self.emission_tex, pixel, 0)
            metadata = self.surface_records[primary.x * 3]
            surface = decode_deferred_surface(base, frame, emission, packed.y)
            context = deferred_shading_context(uv, shader.vec2(pixel) + shader.vec2(0.5), depth,
                                               surface.normal_ws, metadata)
            normal = decode_octahedral_normal(shader.unpack_half_2x16(primary.z))
            request = shader.ray._tail_request(surface, context, normal, metadata.y, True, self.temporal)
        if ctx.position.y < 1.0:
            return (request.identity, request.anchor)
        return (request.outgoing, request.normal)
