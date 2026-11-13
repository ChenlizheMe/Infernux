import infernux.shader as shader


def world_position(uv: shader.vec2, depth: float, camera: shader.CameraContext) -> shader.vec3:
    """Reconstruct one raster sample from top-left image UV and device depth."""
    clip = shader.vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0)
    world = camera.inverse_view_projection_matrix @ clip
    return world.xyz / world.w


def clip_position(position_ws: shader.vec3, camera: shader.CameraContext) -> shader.vec4:
    return camera.view_projection_matrix @ shader.vec4(position_ws, 1.0)


def linear_eye_depth(depth: float, camera: shader.CameraContext) -> float:
    """Positive eye-space Z for the engine's perspective or orthographic depth."""
    if shader.abs(camera.projection_matrix[3][3]) > 0.5:
        return shader.mix(camera.projection_params.y, camera.projection_params.x, depth)
    return 1.0 / (camera.depth_params.z * depth + camera.depth_params.w)
