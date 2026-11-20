import infernux.shader as shader
from infernux.shader.lib.ambient_visibility import horizon_arc
from infernux.shader.lib.camera import linear_eye_depth, world_position
from infernux.shader.lib.surface import decode_octahedral_normal


class ContactVisibility(shader.fullscreen):
    """Local hemispherical visibility and positive eye depth on the same raster sample."""

    shader_id = "infernux/internal/contact_visibility"
    hidden = True
    depth_tex = shader.input("depth", shader.Depth2D)
    frame = shader.input("surface_frame", shader.Color2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        size = shader.ivec2(shader.dimensions(self.depth_tex, 0))
        source = shader.ivec2(ctx.uv * shader.vec2(size))
        depth = shader.load(self.depth_tex, source, 0)
        if depth <= 0.0:
            return shader.vec4(1.0, 0.0, 0.0, 1.0)

        camera = shader.camera()
        uv = (shader.vec2(source) + shader.vec2(0.5)) / shader.vec2(size)
        origin = world_position(uv, depth, camera)
        normal = decode_octahedral_normal(shader.load(self.frame, source, 0).xy)
        # The projected radius fixes an angular footprint in perspective and follows orthographic zoom.
        radius_px = shader.max(3.0, 0.03 * shader.float(shader.min(size.x, size.y)))
        scale = radius_px / shader.vec2(size)
        across = world_position(uv + shader.vec2(scale.x, 0.0), depth, camera) - origin
        down = world_position(uv + shader.vec2(0.0, scale.y), depth, camera) - origin
        radius = shader.length(across)
        view = shader.normalize(world_position(uv, 0.75, camera) - world_position(uv, 0.25, camera))
        facing = shader.max(shader.dot(normal, view), 0.0)
        visible = 0.0
        open_area = 0.0
        for axis in range(4):
            angle = (shader.float(axis) + 0.5) * 0.7853981634
            direction = shader.vec2(shader.cos(angle), shader.sin(angle))
            span = direction.x * across + direction.y * down
            tangent = shader.normalize(span - view * shader.dot(span, view))
            projected = shader.dot(normal, tangent)
            length_n = shader.sqrt(facing * facing + projected * projected)
            for side in range(2):
                sign = 1.0 if side == 0 else -1.0
                along = sign * projected
                open_cos = shader.clamp(-along / shader.max(length_n, 1e-6), -1.0, 1.0)
                horizon = open_cos
                for step in range(4):
                    t = shader.float(step + 1) * 0.25
                    at = source + shader.ivec2(shader.round(sign * direction * shader.max(1.0, t * t * radius_px)))
                    if shader.any(at < shader.ivec2(0)) or shader.any(at >= size):
                        continue
                    other = shader.load(self.depth_tex, at, 0)
                    if other <= 0.0:
                        continue
                    sample_uv = (shader.vec2(at) + shader.vec2(0.5)) / shader.vec2(size)
                    delta = world_position(sample_uv, other, camera) - origin
                    distance = shader.length(delta)
                    support = shader.saturate(2.0 * (1.0 - distance / radius))
                    sample_cos = shader.clamp(shader.dot(delta, view) / shader.max(distance, 1e-7), -1.0, 1.0)
                    horizon = shader.max(horizon, shader.mix(open_cos, sample_cos, support))
                visible += horizon_arc(horizon, facing, along)
                open_area += horizon_arc(open_cos, facing, along)
        visibility = 1.0 - shader.saturate((open_area - visible) / shader.max(open_area, 1e-6))
        eye_depth = linear_eye_depth(depth, camera)
        return shader.vec4(visibility, shader.log2(1.0 + eye_depth), 0.0, 1.0)
