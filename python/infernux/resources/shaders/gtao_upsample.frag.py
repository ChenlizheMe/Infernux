import infernux.shader as shader
from infernux.shader.lib.ray_sh_filter import sh_guide_normal
from infernux.shader.lib.surface import decode_octahedral_normal


class GTAOUpsample(shader.fullscreen):
    shader_id = "infernux/internal/gtao_upsample"
    hidden = True
    visibility = shader.input("visibility", shader.Color2D)
    guide = shader.input("guide", shader.UInt2D)
    depth = shader.input("depth", shader.Depth2D)
    frame = shader.input("surface_frame", shader.Color2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        size = shader.ivec2(shader.dimensions(self.guide, 0))
        cell = shader.min(pixel >> 1, size - 1)
        depth = shader.load(self.depth, pixel, 0)
        if depth <= 0.0:
            return shader.vec4(1.0)
        world = decode_octahedral_normal(shader.load(self.frame, pixel, 0).xy)
        normal = (shader.camera().view_matrix @ shader.vec4(world, 0.0)).xyz
        total = 0.0
        sum_ao = 0.0
        for tap in range(5):
            offset = shader.ivec2(0)
            if tap > 0:
                sign = 2 * ((tap - 1) & 1) - 1
                offset = shader.ivec2(sign, 0) if tap <= 2 else shader.ivec2(0, sign)
            at = cell + offset
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= size):
                continue
            guide = shader.load(self.guide, at, 0)
            other_depth = shader.as_float(guide.x)
            if other_depth > 0.0:
                dot = shader.max(0.0, shader.dot(normal, sh_guide_normal(guide.y)))
                normal_weight = dot * dot
                normal_weight *= normal_weight
                normal_weight *= normal_weight
                spatial = 1.0 if tap == 0 else 0.36787944
                weight = spatial * normal_weight * shader.exp2(-200.0 * shader.abs(1.0 - depth / other_depth))
                sum_ao += weight * shader.load(self.visibility, at, 0).x
                total += weight
        return shader.vec4(sum_ao / total if total > 1e-6 else 1.0)
