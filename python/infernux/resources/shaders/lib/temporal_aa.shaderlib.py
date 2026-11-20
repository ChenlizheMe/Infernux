import infernux.shader as shader
from infernux.shader.lib.color import luminance


def taa_color_to_ycbcr(color: shader.vec4) -> shader.vec4:
    y = luminance(color.rgb)
    return shader.vec4(y, (color.b - y) / 1.8556, (color.r - y) / 1.5748, color.a)


def taa_ycbcr_to_color(color: shader.vec4) -> shader.vec4:
    rgb = color.x + shader.vec3(1.5748 * color.z, -0.187324 * color.y - 0.468124 * color.z,
                               1.8556 * color.y)
    return shader.vec4(shader.max(rgb, 0.0), shader.saturate(color.a))


def taa_history_sample(history: shader.History2D, uv: shader.vec2, size: shader.vec2) -> shader.vec4:
    point = uv * size
    center = shader.floor(point - 0.5) + 0.5
    f = point - center
    squared = f * f
    w0 = f * (-0.5 + f * (1.0 - 0.5 * f))
    w2 = f * (0.5 + f * (2.0 - 1.5 * f))
    w3 = squared * (-0.5 + 0.5 * f)
    middle = 1.0 + squared * (-2.5 + 1.5 * f) + w2
    # Combine the central pair into one bilinear lookup. Omitting the four
    # corner taps leaves a five-tap cross with a new normalization.
    left_top = (center - 1.0) / size
    right_bottom = (center + 2.0) / size
    middle_uv = (center + w2 / middle) / size
    top = w0.y * middle.x
    left = w0.x * middle.y
    main = middle.x * middle.y
    right = w3.x * middle.y
    bottom = w3.y * middle.x
    color = shader.sample_level(history, shader.vec2(middle_uv.x, left_top.y), 0.0) * top
    color += shader.sample_level(history, shader.vec2(left_top.x, middle_uv.y), 0.0) * left
    color += shader.sample_level(history, middle_uv, 0.0) * main
    color += shader.sample_level(history, shader.vec2(right_bottom.x, middle_uv.y), 0.0) * right
    color += shader.sample_level(history, shader.vec2(middle_uv.x, right_bottom.y), 0.0) * bottom
    return color / (top + left + main + right + bottom)
