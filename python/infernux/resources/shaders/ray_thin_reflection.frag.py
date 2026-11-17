import infernux.shader as shader
from infernux.shader.lib.camera import world_position
from infernux.shader.lib.surface import decode_octahedral_normal


class RayThinReflection(shader.fullscreen):
    shader_id = "infernux/internal/ray_thin_reflection"
    hidden = True
    layer = shader.input("thin_layer", shader.UInt2D)
    optics = shader.input("thin_optics", shader.UInt2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        layer = shader.load(self.layer, pixel, 0)
        if layer.x == 0:
            return shader.vec4(0.0)
        optics = shader.load(self.optics, pixel, 0)
        coverage = shader.unpack_half_2x16(optics.w).y
        ior = shader.unpack_half_2x16(optics.x).x
        if coverage <= 0.0 or ior == 1.0:
            return shader.vec4(0.0, 0.0, 0.0, 1.0)
        normal = decode_octahedral_normal(shader.unpack_half_2x16(layer.y))
        position = world_position(ctx.uv, shader.as_float(layer.w), shader.camera())
        return shader.ray._thin_reflection(position, normal, self.surface_records, ctx.position.xy)
