import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_moments import sh_diffuse_irradiance
from infernux.shader.lib.ray_sh_filter import sh_depth_falloff, sh_guide_phase
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseResolve(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_resolve"
    hidden = True
    red_sh = shader.input("red_sh", shader.Color2D)
    green_sh = shader.input("green_sh", shader.Color2D)
    blue_sh = shader.input("blue_sh", shader.Color2D)
    primary = shader.input("primary", shader.UInt2D)
    frame = shader.input("frame", shader.Color2D)
    divisor = shader.parameter.int(default=2)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.ivec2(shader.dimensions(self.red_sh, 0))
        receiver = shader.load(self.primary, pixel, 0)
        if receiver.x == 0:
            return shader.vec4(0.0)
        depth = shader.as_float(receiver.y)
        shading = decode_octahedral_normal(shader.load(self.frame, pixel, 0).xy)
        cell = shader.min(shader.ivec2(shader.uvec2(pixel) // shader.uint(self.divisor)), extent - 1)
        phase = sh_guide_phase(shader._view_sample_index())
        inverse_depth = 1.0 / shader.max(depth, 1.0e-8)
        camera = shader.camera()
        normal = decode_octahedral_normal(shader.unpack_half_2x16(receiver.z))
        position = world_position(ctx.uv, depth, camera)
        plane = shader.transpose(camera.inverse_view_projection_matrix) @ shader.vec4(normal, -shader.dot(normal, position))
        tolerance = 0.001 + 0.002 * linear_eye_depth(depth, camera)
        orthographic = shader.abs(camera.projection_matrix[3][3]) > 0.5
        size_to_clip = shader.vec2(2.0, -2.0) / shader.vec2(full)
        subpixel = (shader.vec2(pixel - cell * self.divisor) + 0.5) / shader.float(self.divisor)
        red = shader.vec4(0.0)
        green = shader.vec4(0.0)
        blue = shader.vec4(0.0)
        total = 0.0
        for tap in range(9):
            offset = shader.ivec2(tap % 3 - 1, int(tap / 3) - 1)
            at = cell + offset
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                continue
            # The 2× and 4× coefficient grids map to the even raster positions
            # before adding the phase, including at odd viewport sizes.
            site = shader.min(at * self.divisor + phase, full - 1)
            donor = shader.load(self.primary, site, 0)
            if donor.x == 0:
                continue
            other_depth = shader.as_float(donor.y)
            delta = shader.vec2(site - pixel) * size_to_clip
            plane_error = shader.abs(shader.dot(plane.xy, delta) + plane.z * (other_depth - depth))
            # The projected plane residual is the world-space distance divided by clip W.
            inverse_w = 1.0 if orthographic else camera.depth_params.z * other_depth + camera.depth_params.w
            plane_weight = shader.saturate(1.3333333 - plane_error / (3.0 * tolerance * inverse_w))
            # Filter around the coefficient cell centers. The raster phase can
            # validate a donor, but does not move the reconstruction footprint.
            radius = shader.vec2(offset) + 0.5 - subpixel
            relative_depth = shader.abs(1.0 - other_depth * inverse_depth)
            weight = plane_weight * shader.exp2(-sh_depth_falloff(relative_depth, 1.0) - shader.dot(radius, radius))
            red += weight * shader.load(self.red_sh, at, 0)
            green += weight * shader.load(self.green_sh, at, 0)
            blue += weight * shader.load(self.blue_sh, at, 0)
            total += weight
        norm = 1.0 / shader.max(total, 1.0e-4)
        confidence = shader.saturate(4.0 * total)
        return shader.vec4(sh_diffuse_irradiance(red * norm, green * norm, blue * norm, shading), confidence)
