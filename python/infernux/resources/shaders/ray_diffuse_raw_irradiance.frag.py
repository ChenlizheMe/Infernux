import infernux.shader as shader
from infernux.shader.lib.ray_moments import diffuse_irradiance, sample_direction
from infernux.shader.lib.ray_sh_filter import sh_trace_pixel
from infernux.shader.lib.ray_signal import unpack_ray_signal
from infernux.shader.lib.surface import decode_octahedral_normal


class RayDiffuseRawIrradiance(shader.fullscreen):
    shader_id = "infernux/internal/ray_diffuse_raw_irradiance"
    hidden = True
    observations = shader.input("observations", shader.UInt2D)
    primary = shader.input("primary", shader.UInt2D)
    frame = shader.input("frame", shader.Color2D)
    full_resolution = shader.parameter.bool(default=False)
    temporal = shader.parameter.bool(default=True)
    trace_divisor = shader.parameter.int(default=4)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        full = shader.ivec2(shader.dimensions(self.primary, 0))
        extent = shader.ivec2(shader.dimensions(self.observations, 0))
        cell = pixel
        if self.full_resolution:
            cell = shader.min(shader.ivec2(shader.uvec2(pixel) // shader.uint(self.trace_divisor)), extent - 1)
        if not self.full_resolution:
            index = shader._view_sample_index() if self.temporal else shader.uint(0)
            pixel = sh_trace_pixel(cell, self.trace_divisor, full, index)
        receiver = shader.load(self.primary, pixel, 0)
        if receiver.x == 0:
            return shader.vec4(0.0)
        normal = decode_octahedral_normal(shader.unpack_half_2x16(receiver.z))
        shading = decode_octahedral_normal(shader.load(self.frame, pixel, 0).xy)
        packet = shader.load(self.observations, cell, 0)
        sample = unpack_ray_signal(packet)
        axis = shader.vec4(1.0, sample_direction(packet))
        return shader.vec4(diffuse_irradiance(sample.r * axis, sample.g * axis, sample.b * axis, normal, shading), sample.a)
