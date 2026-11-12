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


def rgb_to_hsv(rgb: shader.vec3) -> shader.vec3:
    k = shader.vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0)
    p = shader.lerp(shader.vec4(rgb.bg, k.wz), shader.vec4(rgb.gb, k.xy), shader.step(rgb.b, rgb.g))
    q = shader.lerp(shader.vec4(p.xyw, rgb.r), shader.vec4(rgb.r, p.yzx), shader.step(p.x, rgb.r))
    delta = q.x - shader.min(q.w, q.y)
    return shader.vec3(shader.abs(q.z + (q.w - q.y) / (6.0 * delta + 1e-10)), delta / (q.x + 1e-10), q.x)


def hsv_to_rgb(hsv: shader.vec3) -> shader.vec3:
    p = shader.abs(shader.frac(hsv.xxx + shader.vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0)
    return hsv.z * shader.lerp(shader.vec3(1.0), shader.clamp(p - 1.0, 0.0, 1.0), hsv.y)


def _white_lms(xy: shader.vec2) -> shader.vec3:
    xyz = shader.vec3(xy.x, xy.y, 1.0 - xy.x - xy.y) / xy.y
    return shader.vec3(shader.dot(xyz, shader.vec3(0.7328, 0.4296, -0.1624)),
                       shader.dot(xyz, shader.vec3(-0.7036, 1.6975, 0.0061)),
                       shader.dot(xyz, shader.vec3(0.0030, 0.0136, 0.9834)))


def white_balance(rgb: shader.vec3, temperature: shader.f32, tint: shader.f32) -> shader.vec3:
    """CAT02 chromatic adaptation of linear sRGB, with zero anchored at D65."""
    d65 = shader.vec2(0.31271, 0.32902)
    t = temperature / 65.0
    x = d65.x - t * (0.1 if t < 0.0 else 0.05)
    y = d65.y + (x - d65.x) * (2.87 - 3.0 * (x + d65.x)) + tint / 65.0 * 0.05
    balance = _white_lms(d65) / _white_lms(shader.vec2(x, y))
    lms = shader.vec3(shader.dot(rgb, shader.vec3(0.3904725024, 0.5499043704, 0.0089015942)),
                       shader.dot(rgb, shader.vec3(0.0709258615, 0.9631073867, 0.0013580925)),
                       shader.dot(rgb, shader.vec3(0.0231426779, 0.1280122110, 0.9360519444))) * balance
    return shader.max(shader.vec3(shader.dot(lms, shader.vec3(2.8583110961, -1.6287079604, -0.0248186967)),
                                  shader.dot(lms, shader.vec3(-0.2104347759, 1.1584149339, 0.0003204633)),
                                  shader.dot(lms, shader.vec3(-0.0418895045, -0.1181543331, 1.0688865654))), 0.0)
