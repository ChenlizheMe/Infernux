"""Linear Rec.709 luminance and the sRGB transfer functions."""

import infernux.shader as shader


def luminance(rgb: shader.vec3) -> shader.f32:
    return shader.dot(rgb, shader.vec3(0.2126, 0.7152, 0.0722))


def _decode(value: shader.f32) -> shader.f32:
    if value <= 0.04045:
        return value / 12.92
    return shader.pow((value + 0.055) / 1.055, 2.4)


def srgb_to_linear(rgb: shader.vec3) -> shader.vec3:
    return shader.vec3(_decode(rgb.r), _decode(rgb.g), _decode(rgb.b))


def _encode(value: shader.f32) -> shader.f32:
    if value <= 0.0031308:
        return value * 12.92
    return 1.055 * shader.pow(value, 1.0 / 2.4) - 0.055


def linear_to_srgb(rgb: shader.vec3) -> shader.vec3:
    return shader.vec3(_encode(rgb.r), _encode(rgb.g), _encode(rgb.b))
