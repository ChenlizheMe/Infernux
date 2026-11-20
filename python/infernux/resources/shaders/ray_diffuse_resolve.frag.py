import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_moments import diffuse_irradiance, unpack_moment
from infernux.shader.lib.ray_signal import ray_grid_pixel
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseResolve(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_resolve"
    hidden = True
    coefficients = shader.input("coefficients", shader.UInt2D)
    directions = shader.input("directions", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    frame = shader.input("frame", shader.Color2D)
    raw = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.ivec2(shader.dimensions(self.coefficients, 0))
        if self.raw:
            pixel = ray_grid_pixel(pixel, extent, full)
        receiver = shader.load(self.primary, pixel, 0)
        if receiver.x == 0:
            return shader.vec4(0.0)
        camera = shader.camera()
        uv = (shader.vec2(pixel) + shader.vec2(0.5)) / shader.vec2(full)
        depth = shader.as_float(receiver.y)
        position = world_position(uv, depth, camera)
        normal = decode_octahedral_normal(shader.unpack_half_2x16(receiver.z))
        shading = decode_octahedral_normal(shader.load(self.frame, pixel, 0).xy)
        cell = ray_grid_pixel(pixel, full, extent)
        site = ray_grid_pixel(cell, extent, full)
        if self.raw:
            dc = shader.load(self.coefficients, cell, 0)
            axes = shader.load(self.directions, cell, 0)
            color = shader.as_float(dc.xyz)
            return shader.vec4(diffuse_irradiance(unpack_moment(color.r, axes.r), unpack_moment(color.g, axes.g),
                                                   unpack_moment(color.b, axes.b), normal, shading), shader.float(dc.w))

        point = shader.vec2(cell) + shader.vec2(pixel - site) * shader.vec2(extent) / shader.vec2(full)
        base = shader.ivec2(shader.floor(point))
        band = 0.004 + 0.003 * linear_eye_depth(depth, camera)
        red = shader.vec4(0.0)
        green = shader.vec4(0.0)
        blue = shader.vec4(0.0)
        total = 0.0
        # Thin receivers may miss every inner site. Search the one-cell border only for those pixels.
        for tap in range(16):
            if tap == 4 and total > 0.0:
                break
            offset = shader.ivec2(tap & 1, tap >> 1)
            if tap >= 12:
                offset = shader.ivec2((tap & 1) * 3 - 1, (tap - 12) >> 1)
            elif tap >= 4:
                offset = shader.ivec2((tap & 3) - 1, ((tap - 4) >> 2) * 3 - 1)
            at = base + offset
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                continue
            source = ray_grid_pixel(at, extent, full)
            guide = shader.load(self.primary, source, 0)
            dc = shader.load(self.coefficients, at, 0)
            if guide.x == 0 or dc.w == 0:
                continue
            other_normal = decode_octahedral_normal(shader.unpack_half_2x16(guide.z))
            other_uv = (shader.vec2(source) + shader.vec2(0.5)) / shader.vec2(full)
            other = world_position(other_uv, shader.as_float(guide.y), camera)
            delta = shader.vec2(at) - point
            tent = shader.saturate(shader.vec2(1.0) - shader.abs(delta))
            weight = tent.x * tent.y if tap < 4 else shader.exp2(-shader.dot(delta, delta))
            weight *= shader.saturate((shader.dot(other_normal, normal) - 0.72) / 0.22)
            weight *= shader.saturate(1.0 - shader.abs(shader.dot(other - position, normal)) / band)
            axes = shader.load(self.directions, at, 0)
            color = shader.as_float(dc.xyz)
            red += weight * unpack_moment(color.r, axes.r)
            green += weight * unpack_moment(color.g, axes.g)
            blue += weight * unpack_moment(color.b, axes.b)
            total += weight
        if total <= 0.0:
            return shader.vec4(0.0)
        return shader.vec4(diffuse_irradiance(red, green, blue, normal, shading) / total, 1.0)
