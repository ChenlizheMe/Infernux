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


def load_deferred_surface(albedo: shader.Color2D, normal: shader.Color2D, emission: shader.Color2D,
                          surface: shader.UInt2D, pixel: shader.ivec2) -> shader.SurfaceData:
    return decode_deferred_surface(shader.load(albedo, pixel, 0), shader.load(normal, pixel, 0),
                                   shader.load(emission, pixel, 0), shader.load(surface, pixel, 0).y)


def deferred_shading_context(uv: shader.vec2, pixel: shader.vec2, depth: float,
                             normal_ws: shader.vec3, metadata: shader.uvec4) -> shader.ShadingContext:
    camera = shader.camera()
    position_ws = world_position(uv, depth, camera)
    view_depth = shader.abs((camera.view_matrix @ shader.vec4(position_ws, 1.0)).z)
    return shader._resolved_shading_context(position_ws, normal_ws, view_depth, pixel,
                                            metadata.x, metadata.z != 0, 0.0)


def _depth_slope(image: shader.Depth2D, pixel: shader.ivec2, axis: shader.ivec2,
                 last: shader.ivec2, depth: float) -> float:
    lo = shader.max(pixel - axis, shader.ivec2(0))
    hi = shader.min(pixel + axis, last)
    backward = depth - shader.load(image, lo, 0)
    forward = shader.load(image, hi, 0) - depth
    use_forward = shader.all(lo == pixel) or (shader.any(hi != pixel) and shader.abs(forward) < shader.abs(backward))
    return forward if use_forward else backward


def deferred_raster_context(uv: shader.vec2, pixel: shader.vec2, depth: float, normal_ws: shader.vec3,
                            metadata: shader.uvec4, image: shader.Depth2D) -> shader.ShadingContext:
    context = deferred_shading_context(uv, pixel, depth, normal_ws, metadata)
    if context.receives_shadows:
        extent = shader.ivec2(shader.dimensions(image, 0))
        last = extent - shader.ivec2(1)
        at = shader.ivec2(pixel)
        dx = _depth_slope(image, at, shader.ivec2(1, 0), last, depth)
        dy = _depth_slope(image, at, shader.ivec2(0, 1), last, depth)
        # Device depth is affine on a triangle. Transform its clip-space plane back
        # to world space without subtracting distant world positions at adjacent pixels.
        slope = shader.vec2(dx, -dy) * shader.vec2(extent) * 0.5
        ndc = shader.vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0)
        plane = shader.vec4(-slope.x, -slope.y, 1.0, shader.dot(slope, ndc) - depth)
        context.shadow_plane_ws = (shader.transpose(shader.camera().view_projection_matrix) @ plane).xyz
    return context
