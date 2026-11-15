"""Depth effects consume positive eye-space distance, independently of depth encoding."""

import infernux.shader as shader


def fade(scene_depth: float, surface_depth: float, distance: float) -> float:
    return shader.saturate((scene_depth - surface_depth) / distance)


def linear_fog(view_depth: float, start: float, end: float) -> float:
    return shader.saturate((view_depth - start) / (end - start))


def exponential_fog(view_depth: float, density: float) -> float:
    return 1.0 - shader.exp(-density * view_depth)


def exponential_squared_fog(view_depth: float, density: float) -> float:
    extinction = density * view_depth
    return 1.0 - shader.exp(-extinction * extinction)


def height_fog(view_depth: float, world_height: float, density: float, falloff: float, base_height: float) -> float:
    return exponential_fog(view_depth, density) * shader.exp(-shader.max(world_height - base_height, 0.0) * falloff)


def edge(image: shader.Depth2D, uv: shader.vec2, texel_size: shader.vec2, threshold: float) -> float:
    a = shader.sample(image, uv)
    b = shader.sample(image, uv + shader.vec2(texel_size.x, 0.0))
    c = shader.sample(image, uv + shader.vec2(0.0, texel_size.y))
    d = shader.sample(image, uv + texel_size)
    return shader.step(threshold, shader.abs(a - d) + shader.abs(b - c))
