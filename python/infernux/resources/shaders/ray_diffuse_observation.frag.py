import infernux.shader as shader
from infernux.shader.lib.camera import world_position
from infernux.shader.lib.ray_sh_filter import sh_guide_normal, sh_guide_pixel
from infernux.shader.lib.ray_signal import pack_ray_signal
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseObservation(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_observation"
    hidden = True
    primary = shader.input("primary", shader.UInt2D)
    screen_depth = shader.input("screen_depth", shader.UInt2D)
    previous_radiance = shader.input("previous_radiance", shader.History2D, domain="scene_linear_hdr")
    reprojection = shader.input("reprojection", shader.UInt2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])
    tail_cache = shader.input("ray_tail_read", shader.Buffer[shader.uvec4])
    divisor = shader.parameter.int(default=4)
    temporal = shader.parameter.bool(default=True)

    def main(self, ctx: shader.fragment_input) -> shader.uvec4:
        cell = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.max(shader.ivec2(1), shader.ivec2(full / self.divisor))
        index = shader._view_sample_index() if self.temporal else shader.uint(0)
        half_grid = extent * 2
        site = sh_guide_pixel(cell * 2, half_grid, full, index)
        receiver = shader.load(self.primary, site, 0)
        if receiver.x == 0:
            return shader.uvec4(0)
        geometric_normal = decode_octahedral_normal(shader.unpack_half_2x16(receiver.z))
        camera = shader.camera()
        view_normal = sh_guide_normal(shader.load(self.screen_depth, site >> 1, 0).y)
        shading_normal = shader.normalize((shader.transpose(camera.view_matrix) @ shader.vec4(view_normal, 0.0)).xyz)
        # The exact triangle traversal needs the point on the phase-selected raster surface.
        uv = (shader.vec2(site) + 0.5) / shader.vec2(full)
        position = world_position(uv, shader.as_float(receiver.y), camera)
        # Each half-resolution texel represents one of the four full-resolution sample centers.
        scale = 0.25 * shader.vec2(full)
        phase = shader.vec2(site & 1)
        sample_grid = shader.vec4(scale, scale + 0.25 - 0.5 * phase)
        sample = shader.ray._diffuse_sample(
            position, geometric_normal, shading_normal, shader.uvec2(cell), self.temporal, self.screen_depth, sample_grid,
            self.previous_radiance, self.reprojection, self.surface_records, self.tail_cache
        )
        return pack_ray_signal(sample.radiance, sample.witness)
