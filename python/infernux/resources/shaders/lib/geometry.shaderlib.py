"""Geometry operations shared by vertex, fragment and numerical GPU programs."""

import infernux.shader as shader


def rotate_axis(value: shader.vec3, axis: shader.vec3, angle: float) -> shader.vec3:
    """Rotate around a nonzero axis by radians, using the right-hand rule."""
    direction = shader.normalize(axis)
    sine = shader.sin(angle)
    cosine = shader.cos(angle)
    return value * cosine + shader.cross(direction, value) * sine + direction * shader.dot(direction, value) * (1.0 - cosine)


def project(value: shader.vec3, direction: shader.vec3) -> shader.vec3:
    return direction * (shader.dot(value, direction) / shader.dot(direction, direction))


def reject(value: shader.vec3, direction: shader.vec3) -> shader.vec3:
    return value - project(value, direction)


def transform_direction(matrix: shader.mat4, direction: shader.vec3) -> shader.vec3:
    return (matrix @ shader.vec4(direction, 0.0)).xyz
