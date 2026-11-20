import infernux.shader as shader
from infernux.shader.lib.color import final_overlay_color


class Grid(shader.frag):
    shader_id = "infernux/editor/grid"
    hidden = True
    surface_type = "transparent"
    cull_mode = "none"
    depth_write = False
    cast_shadows = False
    fade_start = shader.material.float(default=15.0)
    fade_end = shader.material.float(default=80.0)

    def grid_line(self, uv: shader.vec2) -> float:
        width = shader.max(shader.fwidth(uv), shader.vec2(0.000001))
        cell = shader.frac(uv)
        distance = shader.min(cell, 1.0 - cell) / width
        coverage = 1.0 - shader.smoothstep(0.0, 1.0, distance)
        density_fade = 1.0 - shader.smoothstep(0.5, 1.0, shader.max(width.x, width.y))
        return shader.max(coverage.x, coverage.y) * density_fade

    def main(self, ctx: shader.mesh_input) -> shader.FragmentOutput:
        camera = shader.camera()
        view = shader.view()
        uv = ctx.position.xy * view.pixel_size
        # Unproject in homogeneous coordinates. The grid crosses y=0 at
        # origin.y + depth * axis.y = 0, including for an orthographic camera.
        origin = camera.inverse_view_projection_matrix @ shader.vec4(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, 0.0, 1.0)
        axis = camera.inverse_view_projection_matrix @ shader.vec4(0.0, 0.0, 1.0, 0.0)
        shader.clip(shader.abs(axis.y) - 0.0000001)
        depth = -origin.y / axis.y
        shader.clip(shader.min(depth, 1.0 - depth))
        hit = origin + depth * axis
        position = hit.xyz / hit.w
        coord = position.xz
        minor = self.grid_line(coord)
        major = self.grid_line(coord / 10.0)
        derivative = shader.fwidth(coord)
        lod = shader.smoothstep(0.3, 0.6, shader.max(derivative.x, derivative.y))
        line_alpha = shader.lerp(minor * 0.25 + major * 0.4, major * 0.4, lod)
        distance = shader.length(coord - view.camera_position_ws.xz)
        distance_fade = 1.0 - shader.smoothstep(self.fade_start, self.fade_end, distance)
        direction = position - view.camera_position_ws
        direction *= shader.rsqrt(shader.max(shader.dot(direction, direction), 0.000000000001))
        angle_fade = shader.smoothstep(0.0, 0.15, shader.abs(direction.y))
        alpha = line_alpha * distance_fade * angle_fade
        shader.clip(alpha - 0.005)
        # Cover rounding in floating-point and 24-bit fixed-point depth when the
        # rasterised floor and the analytical grid share the same plane.
        bias = shader.max(depth * 0.000001, 2.0 / 16777215.0)
        return shader.FragmentOutput(color=shader.vec4(final_overlay_color(shader.vec3(0.6038)), alpha),
                                     depth=shader.min(1.0, depth + bias))
