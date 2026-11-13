import infernux.shader as shader
from infernux.shader.lib.camera import world_position


class DeferredLighting(shader.fullscreen):
    shader_id = "infernux/internal/deferred_lighting"
    hidden = True
    albedo_tex = shader.input("gbuffer_albedo", shader.Color2D)
    normal_tex = shader.input("gbuffer_normal", shader.Color2D)
    material_tex = shader.input("gbuffer_material", shader.Color2D)
    emission_tex = shader.input("gbuffer_emission", shader.Color2D)
    object_tex = shader.input("gbuffer_object", shader.UInt2D)
    depth_tex = shader.input("depth", shader.Depth2D)

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        depth = shader.load(self.depth_tex, pixel, 0)
        objects = shader.load(self.object_tex, pixel, 0)
        if depth <= 0.0 or objects.y == 0:
            return shader.vec4(0.0)

        base = shader.load(self.albedo_tex, pixel, 0)
        normals = shader.load(self.normal_tex, pixel, 0)
        material = shader.load(self.material_tex, pixel, 0)
        emission = shader.load(self.emission_tex, pixel, 0)
        normal_ws = shader.normalize(normals.xyz * 2.0 - 1.0)
        surface = shader.SurfaceData(
            albedo=base.rgb, normal_ws=normal_ws,
            metallic=material.x, smoothness=normals.w, occlusion=material.y,
            emission=emission.rgb, alpha=base.a, specular_highlights=material.z,
            shading_param0=material.w, shading_param1=emission.a,
        )
        camera = shader.camera()
        position_ws = world_position(ctx.uv, depth, camera)
        view_depth = shader.abs((camera.view_matrix @ shader.vec4(position_ws, 1.0)).z)
        context = shader.ShadingContext(
            position_ws=position_ws, geometric_normal_ws=normal_ws,
            view_depth=view_depth, pixel_position=ctx.position.xy,
            layer_mask=objects.x, receives_shadows=objects.z != 0,
        )
        return shader.vec4(shader.lighting.shade(surface, context, objects.y), base.a)
