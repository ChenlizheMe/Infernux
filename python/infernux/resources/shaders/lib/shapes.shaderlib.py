"""Pixel-antialiased procedural coverage masks in the fragment stage."""

import infernux.shader as shader


def cross_mask(uv: shader.vec2, center: shader.vec2, extent: shader.vec2, thickness: float) -> float:
    """Plus coverage; extent is the half-length of each axis-aligned bar."""
    distance = shader.abs(uv - center)
    width = shader.fwidth(distance)
    vertical = (1.0 - shader.smoothstep(thickness - width.x, thickness + width.x, distance.x)) * (1.0 - shader.step(extent.y, distance.y))
    horizontal = (1.0 - shader.smoothstep(thickness - width.y, thickness + width.y, distance.y)) * (1.0 - shader.step(extent.x, distance.x))
    return shader.max(horizontal, vertical)


def circle_mask(uv: shader.vec2, center: shader.vec2, radius: float) -> float:
    distance = shader.length(uv - center)
    width = shader.fwidth(distance)
    return 1.0 - shader.smoothstep(radius - width, radius + width, distance)


def rectangle_mask(uv: shader.vec2, center: shader.vec2, half_size: shader.vec2) -> float:
    distance = shader.abs(uv - center) - half_size
    width = shader.fwidth(distance)
    coverage = 1.0 - shader.smoothstep(-width, width, distance)
    return coverage.x * coverage.y


def line_mask(uv: shader.vec2, start: shader.vec2, end: shader.vec2, thickness: float) -> float:
    segment = end - start
    along = shader.saturate(shader.dot(uv - start, segment) / shader.dot(segment, segment))
    distance = shader.length(uv - start - segment * along)
    width = shader.fwidth(distance)
    return 1.0 - shader.smoothstep(thickness - width, thickness + width, distance)
