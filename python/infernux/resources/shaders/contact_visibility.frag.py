import infernux.shader as shader
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
        # A fixed angular footprint adapts to depth, FOV and orthographic scale.
        radius_px = shader.max(3.0, 0.018 * shader.float(shader.min(size.x, size.y)))
        edge_uv = uv + shader.vec2(radius_px / shader.float(size.x), 0.0)
        radius = shader.length(world_position(edge_uv, depth, camera) - origin)
        occlusion = 0.0
        for side in range(6):
            angle = (shader.float(side) + 0.25) * 1.0471975512
            direction = shader.vec2(shader.cos(angle), shader.sin(angle))
            horizon = 0.0
            for step in range(3):
                fraction = 0.1 * shader.float((step + 1) * (step + 1))
                at = source + shader.ivec2(shader.round(direction * (radius_px * fraction)))
                if shader.any(at < shader.ivec2(0)) or shader.any(at >= size):
                    continue
                other = shader.load(self.depth_tex, at, 0)
                if other <= 0.0:
                    continue
                at_uv = (shader.vec2(at) + shader.vec2(0.5)) / shader.vec2(size)
                delta = world_position(at_uv, other, camera) - origin
                distance = shader.length(delta)
                elevation = shader.dot(normal, delta) / shader.max(distance, 1e-7)
                falloff = shader.saturate(1.0 - distance / shader.max(radius, 1e-7))
                horizon = shader.max(horizon, shader.saturate((elevation - 0.05) / 0.95) * falloff)
            occlusion += horizon * horizon
        eye_depth = linear_eye_depth(depth, camera)
        return shader.vec4(1.0 - occlusion / 6.0, shader.log2(1.0 + eye_depth), 0.0, 1.0)
