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
