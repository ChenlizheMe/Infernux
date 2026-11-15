"""UV transformations; animated helpers take an explicit engine-clock sample."""

import infernux.shader as shader


def tiling_offset(uv: shader.vec2, tiling: shader.vec2, offset: shader.vec2) -> shader.vec2:
    return uv * tiling + offset


def rotate(uv: shader.vec2, center: shader.vec2, angle: float) -> shader.vec2:
    delta = uv - center
    sine = shader.sin(angle)
    cosine = shader.cos(angle)
    return shader.vec2(delta.x * cosine - delta.y * sine, delta.x * sine + delta.y * cosine) + center


def flipbook(uv: shader.vec2, grid: shader.vec2, frame: float, invert_y: bool) -> shader.vec2:
    """Wrap positive or negative frame indices through a row-major atlas."""
    count = grid.x * grid.y
    index = shader.floor(frame)
    index = index - shader.floor(index / count) * count
    row = shader.floor(index / grid.x)
    column = index - row * grid.x
    if invert_y:
        row = grid.y - 1.0 - row
    return (uv + shader.vec2(column, row)) / grid


def twirl(uv: shader.vec2, center: shader.vec2, strength: float, offset: shader.vec2) -> shader.vec2:
    return rotate(uv, center, shader.length(uv - center) * strength) + offset


def radial_shear(uv: shader.vec2, center: shader.vec2, strength: shader.vec2, offset: shader.vec2) -> shader.vec2:
    delta = uv - center
    return uv + delta * strength * shader.length(delta) + offset


def polar(uv: shader.vec2, center: shader.vec2) -> shader.vec2:
    delta = uv - center
    return shader.vec2(shader.atan2(delta.y, delta.x) / 6.283185307179586 + 0.5, shader.length(delta))


def pan(uv: shader.vec2, tiling: shader.vec2, velocity: shader.vec2, time: float) -> shader.vec2:
    return uv * tiling + velocity * time


def triplanar_weights(normal_ws: shader.vec3, sharpness: float) -> shader.vec3:
    weights = shader.pow(shader.abs(normal_ws), shader.vec3(sharpness))
    return weights / (weights.x + weights.y + weights.z)


def triplanar_sample(image: shader.Color2D, position_ws: shader.vec3, normal_ws: shader.vec3, tiling: float, sharpness: float) -> shader.vec4:
    weights = triplanar_weights(normal_ws, sharpness)
    return (shader.sample(image, position_ws.yz * tiling) * weights.x
            + shader.sample(image, position_ws.xz * tiling) * weights.y
            + shader.sample(image, position_ws.xy * tiling) * weights.z)


def parallax_offset(uv: shader.vec2, view_ts: shader.vec3, height: float, scale: float) -> shader.vec2:
    z = shader.max(shader.abs(view_ts.z), 0.0001) * (-1.0 if view_ts.z < 0.0 else 1.0)
    return uv + view_ts.xy / z * (height - 0.5) * scale
