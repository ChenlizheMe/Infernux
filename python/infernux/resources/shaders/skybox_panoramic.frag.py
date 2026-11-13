import infernux.shader as shader


class PanoramicSky(shader.frag):
    shader_id = "infernux/skybox/panoramic"
    render_queue = 32767
    depth_write = False
    depth_test = "less_equal"
    pass_tag = "Skybox"
    cast_shadows = False
    albedo_map = shader.material.texture_2d()
    tint = shader.material.color()
    exposure = shader.material.float(default=1.0)
    rotation = shader.material.float()

    def main(self, ctx: shader.mesh_input) -> shader.vec4:
        direction = shader.normalize(ctx.position_ws)
        uv = shader.vec2(shader.frac(shader.atan2(direction.z, direction.x) / 6.28318530718 + 0.5 + self.rotation / 360.0),
                         shader.acos(shader.clamp(direction.y, -1.0, 1.0)) / 3.14159265359)
        color = shader.sample(self.albedo_map, uv).rgb * self.tint.rgb * self.exposure
        return shader.vec4(color, 1.0)
