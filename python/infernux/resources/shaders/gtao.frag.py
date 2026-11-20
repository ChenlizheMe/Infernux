import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.ray_sh_filter import sh_guide_normal
from infernux.shader.lib.gtao import hilbert_index, horizon_angle


class GTAO(shader.fullscreen):
    shader_id = "infernux/internal/gtao"
    hidden = True
    guide = shader.input("guide", shader.UInt2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        size = shader.ivec2(shader.dimensions(self.guide, 0))
        packed = shader.load(self.guide, pixel, 0)
        depth = shader.as_float(packed.x)
        if depth <= 0.0:
            return shader.vec4(1.0)

        camera = shader.camera()
        projection = camera.projection_matrix
        orthographic = shader.abs(projection[3][3]) > 0.5
        eye_depth = linear_eye_depth(depth, camera)
        scale = 1.0 if orthographic else eye_depth
        focal = shader.vec2(projection[0][0], projection[1][1])
        px_to_view = shader.vec2(2.0, -2.0) / (shader.vec2(size) * focal)
        projection_scale = 0.5 * shader.abs(focal.y) / scale
        radius = (0.125 / projection_scale) / shader.abs(scale * px_to_view.x * 0.9992)
        uv = (shader.vec2(pixel) + 0.5) / shader.vec2(size)
        center = (camera.view_matrix @ shader.vec4(world_position(uv, depth, camera), 1.0)).xyz
        origin = shader.vec3(center.xy, center.z * 0.9992) if orthographic else center * 0.9992
        view = shader.vec3(0.0, 0.0, 1.0) if orthographic else shader.normalize(-origin)
        normal = shader.normalize(sh_guide_normal(packed.y))

        sample_index = hilbert_index(pixel) + 288 * int(shader._view_sample_index() & shader.uint(63))
        jitter = shader.frac(0.5 + shader.float(sample_index) * shader.vec2(0.7548776865, 0.5698403120))
        angle = 3.14159265 * jitter.x
        axis = shader.vec3(shader.cos(angle), shader.sin(angle), 0.0)
        screen_axis = shader.vec2(axis.x, -axis.y)
        tangent = axis - shader.dot(axis, view) * view
        slice_normal = shader.normalize(shader.cross(tangent, view))
        projected = normal - shader.dot(normal, slice_normal) * slice_normal
        length_n = shader.length(projected)
        cos_n = shader.saturate(shader.dot(projected, view) / shader.max(length_n, 1e-6))
        angle_n = shader.sign(shader.dot(tangent, projected)) * horizon_angle(cos_n)
        open_horizon = shader.vec2(shader.cos(angle_n + 1.57079633), shader.cos(angle_n - 1.57079633))
        horizons = open_horizon
        for step in range(3):
            t = (shader.float(step) + shader.frac(jitter.y + shader.float(step * 3) * 0.618034)) / 3.0
            offset = shader.ivec2(shader.round(screen_axis * (radius * t * t + 1.3)))
            for side in range(2):
                tap = shader.clamp(pixel + (1 - 2 * side) * offset, shader.ivec2(0), size - 1)
                sample_depth = shader.as_float(shader.load(self.guide, tap, 0).x)
                if sample_depth > 0.0:
                    z = linear_eye_depth(sample_depth, camera)
                    plane = shader.vec2(tap - pixel) * px_to_view
                    position = center.xy + plane if orthographic else (center.xy / eye_depth + plane) * z
                    delta = shader.vec3(position, -z) - origin
                    distance = shader.length(delta)
                    support = shader.saturate(10.0 - 80.0 * projection_scale * distance)
                    tangent_cosine = shader.dot(delta, view) / shader.max(distance, 1e-6)
                    horizons[side] = shader.max(horizons[side], shader.mix(open_horizon[side], tangent_cosine, support))
        a = 2.0 * horizon_angle(horizons.x)
        b = -2.0 * horizon_angle(horizons.y)
        integrated = 0.25 * (2.0 * cos_n + (a + b) * shader.sin(angle_n) -
                             shader.cos(a - angle_n) - shader.cos(b - angle_n))
        small_radius_bias = 0.5 * shader.saturate(0.01 * (10.0 - radius))
        return shader.vec4(shader.max(0.03, small_radius_bias + shader.mix(length_n, 1.0, 0.05) * integrated))
