import infernux.shader as shader


def hilbert_index(pixel: shader.ivec2) -> int:
    point = pixel & shader.ivec2(63)
    index = 0
    for bit in range(6):
        step = 32 >> bit
        x = int((point.x & step) != 0)
        y = int((point.y & step) != 0)
        index += step * step * ((3 * x) ^ y)
        if y == 0:
            if x != 0:
                point = shader.ivec2(63) - point
            point = point.yx
    return index


def horizon_angle(cosine: float) -> float:
    bound = shader.abs(shader.clamp(cosine, -1.0, 1.0))
    root = shader.as_float(shader.uint(532487669) + (shader.as_uint(1.0 - bound) >> 1))
    angle = (1.570796 - 0.156583 * bound) * root
    return angle if cosine >= 0.0 else 3.141593 - angle
