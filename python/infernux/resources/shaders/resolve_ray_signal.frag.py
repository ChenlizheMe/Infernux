import infernux.shader as shader
from infernux.shader.lib.ray_signal import resolve_ray_diffuse, resolve_ray_specular
from infernux.shader.lib.surface import decode_octahedral_normal


class ResolveRaySignal(shader.fullscreen):
    shader_id = "infernux/internal/resolve_ray_signal"
    hidden = True
    signal = shader.input("signal", shader.UInt2D)
    primary = shader.input("ray_primary_surface", shader.UInt2D)
    frame = shader.input("gbuffer_normal", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    specular = shader.parameter.bool(default=False)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        surface_id = shader.load(self.surface_tex, pixel, 0).x
        depth = shader.load(self.depth_tex, pixel, 0)
        if surface_id == 0 or depth <= 0.0:
            return shader.vec4(0.0)
        frame = shader.load(self.frame, pixel, 0)
        normal = decode_octahedral_normal(frame.xy)
        if self.specular:
            return resolve_ray_specular(self.signal, self.primary, self.frame, pixel, surface_id, depth, normal, frame.z)
        return resolve_ray_diffuse(self.signal, self.primary, self.frame, pixel, surface_id, depth, normal, frame.z)
