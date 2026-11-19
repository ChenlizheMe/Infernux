import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth
from infernux.shader.lib.ray_signal import ray_grid_pixel, ray_previous_eye_depth
from infernux.shader.lib.surface import decode_octahedral_normal


def path_pixel(cell: shader.ivec2, extent: shader.ivec2, full: shader.ivec2) -> shader.ivec2:
    return ray_grid_pixel(cell, extent, full) if shader.any(extent != full) else cell


def related_receiver(primary: shader.uvec4, other: shader.uvec4, expected_depth: float) -> bool:
    if primary.x != other.x:
        return False
    a = decode_octahedral_normal(shader.unpack_half_2x16(primary.z))
    b = decode_octahedral_normal(shader.unpack_half_2x16(other.z))
    distance = linear_eye_depth(shader.as_float(other.y), shader.camera())
    return shader.dot(a, b) > 0.86 and shader.abs(distance - expected_depth) <= 0.04 + 0.04 * expected_depth


def previous_path(primary: shader.uvec4, motion: shader.uvec4, previous_surface: shader.UInt2D,
                  paths: shader.UInt2D, uv: shader.vec2,
                  extent: shader.ivec2, full: shader.ivec2) -> shader.uvec4:
    if (motion.w >> 31) == 0 or motion.x != primary.x:
        return shader.uvec4(0)
    prev_uv = uv - shader.unpack_half_2x16(motion.z) + shader._view_jitter_delta_uv()
    if shader.any(prev_uv < shader.vec2(0.0)) or shader.any(prev_uv >= shader.vec2(1.0)):
        return shader.uvec4(0)
    cell = shader.ivec2(prev_uv * shader.vec2(extent))
    other = shader.load(previous_surface, path_pixel(cell, extent, full), 0)
    if related_receiver(primary, other, ray_previous_eye_depth(motion.w)):
        return shader.load(paths, cell, 0)
    return shader.uvec4(0)
