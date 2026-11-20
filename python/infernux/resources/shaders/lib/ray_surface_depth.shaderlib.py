"""Depth word shared by the raster surface replay and ray history reprojection."""

import infernux.shader as shader


def pack_ray_surface_depth(raster_depth: float, receiver_depth: float, previous_clip: shader.vec4,
                           has_history: bool) -> shader.uint:
    receiver = shader.as_uint(receiver_depth)
    # The replay already writes raster D32. A signed offset preserves the exact receiver bits.
    offset = receiver - shader.as_uint(raster_depth) + shader.uint(0x8000)
    if not has_history or previous_clip.z <= 0.0 or previous_clip.z > previous_clip.w or offset > shader.uint(0xFFFF):
        return receiver
    camera = shader.camera()
    eye_depth = previous_clip.w
    if camera.projection_matrix[3][3] != 0.0:
        eye_depth = shader.mix(camera.projection_params.y, camera.projection_params.x, previous_clip.z / previous_clip.w)
    # Round positive eye Z to bfloat16. Its spare sign bit marks valid history.
    bits = shader.as_uint(eye_depth)
    rounded = bits + shader.uint(0x7FFF) + ((bits >> 16) & shader.uint(1))
    return shader.uint(0x80000000) | (rounded & shader.uint(0x7FFF0000)) | offset


def ray_receiver_depth_bits(raster_depth: float, motion_depth: shader.uint) -> shader.uint:
    if (motion_depth >> 31) == 0:
        return motion_depth
    return shader.as_uint(raster_depth) + (motion_depth & shader.uint(0xFFFF)) - shader.uint(0x8000)


def ray_previous_eye_depth(motion_depth: shader.uint) -> float:
    return shader.as_float(motion_depth & shader.uint(0x7FFF0000))


def previous_depth_plane(normal: shader.vec3, uv: shader.vec2, eye_depth: float,
                         camera: shader.CameraContext) -> shader.vec4:
    """Device-depth plane of the previous receiver, with validity in W."""
    axis = shader.vec3(0.0, 0.0, 1.0) if shader.abs(normal.z) < 0.9 else shader.vec3(0.0, 1.0, 0.0)
    tangent = shader.cross(normal, axis)
    transform = camera.previous_view_projection_matrix
    t = transform @ shader.vec4(tangent, 0.0)
    b = transform @ shader.vec4(shader.cross(normal, tangent), 0.0)
    device = (camera.projection_params.y - eye_depth) / (camera.projection_params.y - camera.projection_params.x)
    if shader.abs(camera.projection_matrix[3][3]) <= 0.5:
        device = (1.0 / eye_depth - camera.depth_params.w) / camera.depth_params.z
    clip = shader.vec3(uv * shader.vec2(2.0, -2.0) + shader.vec2(-1.0, 1.0), device)
    plane = shader.cross(t.xyz - clip * t.w, b.xyz - clip * b.w)
    if shader.abs(plane.z) <= 1.0e-10:
        return shader.vec4(0.0)
    return shader.vec4(plane.xy / plane.z, device, 1.0)
