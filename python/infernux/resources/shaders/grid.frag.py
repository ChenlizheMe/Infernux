import infernux.shader as shader
from infernux.shader.lib.camera import world_position, clip_position


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

    def intersection(self, near: shader.vec3, far: shader.vec3) -> float:
        direction = far - near
        if shader.abs(direction.y) < 0.0001:
            return -1.0
        return -near.y / direction.y

    def main(self, ctx: shader.mesh_input) -> shader.FragmentOutput:
        camera = shader.camera()
        view = shader.view()
        uv = ctx.position.xy * view.pixel_size
        near = world_position(uv, 1.0, camera)
        far = world_position(uv, 0.5, camera)
        t = self.intersection(near, far)
        shader.clip(t)
        position = near + t * (far - near)
        projected = clip_position(position, camera)
        depth = projected.z / projected.w
        shader.clip(shader.min(depth, 1.0 - depth))
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
        return shader.FragmentOutput(color=shader.vec4(shader.vec3(0.6038), alpha), depth=depth)
