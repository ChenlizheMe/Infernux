import infernux.shader as shader


class Error(shader.surface):
    shader_id = "infernux/internal/error"
    hidden = True
    shading_model = shader.ref("./unlit.shadingmodel.py")
    cull_mode = "none"
    render_queue = 2000

    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        uv_grid = ctx.uv0 * 4.0
        world_grid = ctx.position_ws.xz * 2.0
        uv_pattern = shader.fmod(shader.step(0.5, shader.frac(uv_grid.x)) + shader.step(0.5, shader.frac(uv_grid.y)), 2.0)
        world_pattern = shader.fmod(shader.step(0.5, shader.frac(world_grid.x)) + shader.step(0.5, shader.frac(world_grid.y)), 2.0)
        checker = shader.step(0.5, shader.mix(uv_pattern, world_pattern, 0.3))
        return shader.SurfaceData(albedo=shader.vec3(1.0, 0.0, 1.0) * checker)
