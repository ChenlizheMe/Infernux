import infernux.shader as shader
from infernux.shader.lib.camera import world_position
from infernux.shader.lib.surface import decode_model_scalars, decode_octahedral_normal


def decode_deferred_surface(base: shader.vec4, frame: shader.vec4, emission: shader.vec4,
                            packed_scalars: shader.uint) -> shader.SurfaceData:
    scalars = decode_model_scalars(packed_scalars)
    return shader.SurfaceData(
        albedo=base.rgb, normal_ws=decode_octahedral_normal(frame.xy),
        metallic=base.a, smoothness=1.0 - frame.z, occlusion=frame.w,
        emission=emission.rgb, alpha=1.0, specular_highlights=emission.a,
        shading_param0=scalars.x, shading_param1=scalars.y,
    )


def deferred_shading_context(uv: shader.vec2, pixel: shader.vec2, depth: float,
                             normal_ws: shader.vec3, metadata: shader.uvec4) -> shader.ShadingContext:
    camera = shader.camera()
    position_ws = world_position(uv, depth, camera)
    view_depth = shader.abs((camera.view_matrix @ shader.vec4(position_ws, 1.0)).z)
    return shader.ShadingContext(
        position_ws=position_ws, geometric_normal_ws=normal_ws,
        view_depth=view_depth, pixel_position=pixel,
        layer_mask=metadata.x, receives_shadows=metadata.z != 0,
    )
