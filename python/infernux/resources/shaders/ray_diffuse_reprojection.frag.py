import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_surface_depth import ray_previous_eye_depth


class RayDiffuseReprojection(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_reprojection"
    hidden = True
    primary = shader.input("primary", shader.UInt2D)
    previous_primary = shader.input("previous_primary", shader.UInt2D)
    motion = shader.input("motion", shader.UInt2D)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        pixel = shader.ivec2(ctx.position.xy)
        surface = shader.load(self.primary, pixel, 0)
        movement = shader.load(self.motion, pixel, 0)
        if surface.x == 0 or surface.x != movement.x or (movement.w >> 31) == 0:
            return shader.uvec4(0)

        extent = shader.ivec2(shader.dimensions(self.primary, 0))
        uv = (shader.vec2(pixel) + 0.5) / shader.vec2(extent)
        jitter_delta = shader._view_jitter_delta_uv()
        uv_delta = jitter_delta - shader.unpack_half_2x16(movement.z)
        # Quantize the geometric offset before evaluating the raster footprint.
        # This keeps the history lookup and coverage mask on the same grid.
        encoded = shader.ivec2(shader.floor(shader.clamp(uv_delta, -1.0, 1.0) * 32767.0 + 0.5))
        packed_delta = (shader.as_uint(encoded.x) & shader.uint(0xFFFF)) | (shader.as_uint(encoded.y) << 16)
        previous = uv + shader.vec2(encoded) / 32767.0 - jitter_delta
        origin = shader.ivec2(shader.floor(previous * shader.vec2(extent) - 0.5))
        cam = shader.camera()
        depth = shader.as_float(surface.y)
        point = world_position(uv, depth, cam)
        view = (cam.view_matrix @ shader.vec4(point, 1.0)).xyz
        eye = ray_previous_eye_depth(movement.w)
        tolerance = 0.025 * linear_eye_depth(depth, cam) / shader.max(1.0e-5, shader.abs(shader.normalize(view).z))
        mask = shader.uint(0)
        for i in range(4):
            at = origin + shader.ivec2(i & 1, i >> 1)
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                continue
            old = shader.load(self.previous_primary, at, 0)
            if old.x == surface.x and shader.abs(linear_eye_depth(shader.as_float(old.y), cam) - eye) <= tolerance:
                mask |= shader.uint(1) << i
        return shader.uvec4(packed_delta, mask, shader.uint(0), shader.uint(0))
