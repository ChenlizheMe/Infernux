"""Procedural value, gradient and cellular noise without textures or host state."""

import infernux.shader as shader


def hash_1d(value: float) -> float:
    value = shader.frac(value * 0.1031)
    value = value * (value + 33.33)
    return shader.frac(value * (value + value))


def hash_2d(value: shader.vec2) -> float:
    p = shader.frac(value.xyx * 0.1031)
    p = p + shader.dot(p, p.yzx + 33.33)
    return shader.frac((p.x + p.y) * p.z)


def hash_3d(value: shader.vec3) -> float:
    p = shader.frac(value * 0.1031)
    p = p + shader.dot(p, p.yzx + 33.33)
    return shader.frac((p.x + p.y) * p.z)


def hash_vec2(value: shader.vec2) -> shader.vec2:
    p = shader.frac(value.xyx * shader.vec3(0.1031, 0.1030, 0.0973))
    p = p + shader.dot(p, p.yzx + 33.33)
    return shader.frac((p.xx + p.yz) * p.zy)


def hash_vec3(value: shader.vec3) -> shader.vec3:
    p = shader.frac(value * shader.vec3(0.1031, 0.1030, 0.0973))
    p = p + shader.dot(p, p.yxz + 33.33)
    return shader.frac((p.xxy + p.yxx) * p.zyx)


def value_2d(position: shader.vec2) -> float:
    cell = shader.floor(position)
    fraction = shader.frac(position)
    weight = fraction * fraction * (3.0 - 2.0 * fraction)
    low = shader.lerp(hash_2d(cell), hash_2d(cell + shader.vec2(1.0, 0.0)), weight.x)
    high = shader.lerp(hash_2d(cell + shader.vec2(0.0, 1.0)), hash_2d(cell + shader.vec2(1.0)), weight.x)
    return shader.lerp(low, high, weight.y)


def value_3d(position: shader.vec3) -> float:
    cell = shader.floor(position)
    fraction = shader.frac(position)
    w = fraction * fraction * (3.0 - 2.0 * fraction)
    a = shader.lerp(hash_3d(cell), hash_3d(cell + shader.vec3(1.0, 0.0, 0.0)), w.x)
    b = shader.lerp(hash_3d(cell + shader.vec3(0.0, 1.0, 0.0)), hash_3d(cell + shader.vec3(1.0, 1.0, 0.0)), w.x)
    c = shader.lerp(hash_3d(cell + shader.vec3(0.0, 0.0, 1.0)), hash_3d(cell + shader.vec3(1.0, 0.0, 1.0)), w.x)
    d = shader.lerp(hash_3d(cell + shader.vec3(0.0, 1.0, 1.0)), hash_3d(cell + shader.vec3(1.0)), w.x)
    return shader.lerp(shader.lerp(a, b, w.y), shader.lerp(c, d, w.y), w.z)


def _gradient(position: shader.vec2) -> shader.vec2:
    p = position - shader.floor(position / 289.0) * 289.0
    x = (34.0 * p.x + 1.0) * p.x
    x = x - shader.floor(x / 289.0) * 289.0
    x = (34.0 * (x + p.y) + 1.0) * (x + p.y)
    x = x - shader.floor(x / 289.0) * 289.0
    x = shader.frac(x / 41.0) * 2.0 - 1.0
    return shader.normalize(shader.vec2(x, shader.abs(x) - 0.5))


def gradient_2d(position: shader.vec2) -> float:
    cell = shader.floor(position)
    f = shader.frac(position)
    w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    a = shader.dot(_gradient(cell), f)
    b = shader.dot(_gradient(cell + shader.vec2(1.0, 0.0)), f - shader.vec2(1.0, 0.0))
    c = shader.dot(_gradient(cell + shader.vec2(0.0, 1.0)), f - shader.vec2(0.0, 1.0))
    d = shader.dot(_gradient(cell + shader.vec2(1.0)), f - shader.vec2(1.0))
    return shader.lerp(shader.lerp(a, b, w.x), shader.lerp(c, d, w.x), w.y)


def cellular_2d(position: shader.vec2) -> shader.vec2:
    """Return nearest-cell distance and its deterministic scalar identity."""
    cell = shader.floor(position)
    fraction = shader.frac(position)
    nearest = 8.0
    identity = 0.0
    for y in range(-1, 2):
        for x in range(-1, 2):
            neighbor = shader.vec2(float(x), float(y))
            delta = neighbor + hash_vec2(cell + neighbor) - fraction
            distance = shader.dot(delta, delta)
            if distance < nearest:
                nearest = distance
                identity = hash_2d(cell + neighbor)
    return shader.vec2(shader.sqrt(nearest), identity)


def fbm_2d(position: shader.vec2, octaves: int, lacunarity: float, gain: float) -> float:
    amplitude = 0.5
    value = 0.0
    for octave in range(octaves):
        value = value + amplitude * value_2d(position)
        position = position * lacunarity
        amplitude = amplitude * gain
    return value


def fbm_3d(position: shader.vec3, octaves: int, lacunarity: float, gain: float) -> float:
    amplitude = 0.5
    value = 0.0
    for octave in range(octaves):
        value = value + amplitude * value_3d(position)
        position = position * lacunarity
        amplitude = amplitude * gain
    return value
