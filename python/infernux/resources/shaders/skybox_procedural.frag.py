import infernux.shader as shader


class ProceduralSky(shader.frag):
    shader_id = "infernux/skybox/procedural"
    render_queue = 32767
    depth_write = False
    depth_test = "less_equal"
    pass_tag = "Skybox"
    cast_shadows = False
    sky_top_color = shader.material.color(default=(0.431, 0.494, 0.612, 1.0))
    sky_horizon_color = shader.material.color(default=(0.651, 0.725, 0.816, 1.0))
    ground_color = shader.material.color(default=(0.345, 0.345, 0.345, 1.0))
    exposure = shader.material.float(default=1.0)

    def main(self, ctx: shader.mesh_input) -> shader.vec4:
        y = shader.normalize(ctx.position_ws).y
        base = shader.mix(self.ground_color.rgb, self.sky_top_color.rgb, shader.smoothstep(-0.1, 0.45, y))
        if y < 0.0:
            horizon = 1.0 - shader.smoothstep(0.0, 0.1, -y)
            horizon *= horizon
        else:
            horizon = 1.0 - shader.smoothstep(0.0, 0.45, y)
        sky = shader.mix(base, self.sky_horizon_color.rgb, horizon * 0.35)
        haze = shader.pow(1.0 - shader.abs(y), 8.0) * 0.15
        return shader.vec4((sky + shader.vec3(haze)) * self.exposure, 1.0)
