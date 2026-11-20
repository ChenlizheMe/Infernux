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
        raster_motion = shader.unpack_half_2x16(movement.z)
        previous = uv - raster_motion
        origin = shader.ivec2(shader.floor(previous * shader.vec2(extent) - 0.5))
        cam = shader.camera()
        point = world_position(uv, shader.as_float(surface.y), cam)
        view = (cam.view_matrix @ shader.vec4(point, 1.0)).xyz
        eye = ray_previous_eye_depth(movement.w)
        tolerance = 0.025 * eye / shader.max(0.1, shader.abs(shader.normalize(view).z))
        mask = shader.uint(0)
        for i in range(4):
            at = origin + shader.ivec2(i & 1, i >> 1)
            if shader.any(at < shader.ivec2(0)) or shader.any(at >= extent):
                continue
            old = shader.load(self.previous_primary, at, 0)
            if old.x == surface.x and shader.abs(linear_eye_depth(shader.as_float(old.y), cam) - eye) <= tolerance:
                mask |= shader.uint(1) << i
        # Depth history is raster-aligned. Lighting history is reconstructed
        # at unjittered pixel centers, so publish the geometric displacement.
        motion = raster_motion - shader._view_jitter_delta_uv()
        return shader.uvec4(shader.pack_half_2x16(motion), mask, shader.uint(0), shader.uint(0))
