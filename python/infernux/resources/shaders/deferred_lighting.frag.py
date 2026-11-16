import infernux.shader as shader
from infernux.shader.lib.camera import world_position
from infernux.shader.lib.surface import decode_model_scalars, decode_octahedral_normal


class DeferredLighting(shader.fullscreen):
    shader_id = "infernux/internal/deferred_lighting"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    surface_tex = shader.input("gbuffer_surface", shader.UInt2D)
    depth_tex = shader.input("depth", shader.Depth2D)
    surface_records = shader.input("surface_records", shader.Buffer[shader.uvec4])

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        depth = shader.load(self.depth_tex, pixel, 0)
        packed = shader.load(self.surface_tex, pixel, 0)
        base = shader.load(self.albedo_tex, pixel, 0)
        frame = shader.load(self.normal_tex, pixel, 0)
        emission = shader.load(self.emission_tex, pixel, 0)
        normal_ws = decode_octahedral_normal(frame.xy)
        model_scalars = decode_model_scalars(packed.y)
        metadata = self.surface_records[packed.x * 3]
        surface = shader.SurfaceData(
            albedo=base.rgb, normal_ws=normal_ws,
            metallic=base.a, smoothness=1.0 - frame.z, occlusion=frame.w,
            emission=emission.rgb, alpha=1.0, specular_highlights=emission.a,
            shading_param0=model_scalars.x, shading_param1=model_scalars.y,
        )
        camera = shader.camera()
        position_ws = world_position(ctx.uv, depth, camera)
        view_depth = shader.abs((camera.view_matrix @ shader.vec4(position_ws, 1.0)).z)
        context = shader.ShadingContext(
            position_ws=position_ws, geometric_normal_ws=normal_ws,
            view_depth=view_depth, pixel_position=ctx.position.xy,
            layer_mask=metadata.x, receives_shadows=metadata.z != 0,
        )
        # Derivatives belong to the full pixel quad, before coverage and model selection.
        if depth <= 0.0 or packed.x == 0 or metadata.y == 0:
            return shader.vec4(0.0)
        return shader.vec4(shader.lighting.shade(surface, context, metadata.y), 1.0)
