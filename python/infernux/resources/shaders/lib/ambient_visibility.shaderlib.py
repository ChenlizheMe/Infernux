import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth
from infernux.shader.lib.color import luminance


def horizon_arc(cos_h: float, facing: float, along: float) -> float:
    """Twice the cosine-weighted solid angle swept from the view ray to this horizon."""
    sin_h = shader.sqrt(shader.max(0.0, 1.0 - cos_h * cos_h))
    return facing * sin_h * sin_h + along * (shader.acos(cos_h) - sin_h * cos_h)


def contact_visibility(signal: shader.Color2D, uv: shader.vec2, eye_depth: float) -> float:
    """Upsample visibility without transferring occlusion across a foreground silhouette."""
    size = shader.ivec2(shader.dimensions(signal, 0))
    point = uv * shader.vec2(size) - shader.vec2(0.5)
    base = shader.ivec2(shader.floor(point))
    log_depth = shader.log2(1.0 + eye_depth)
    band = 0.04 * eye_depth / (1.0 + eye_depth) + 0.0005
    total = 0.0
    obscured = 0.0
    for tap in range(4):
        cell = shader.clamp(base + shader.ivec2(tap & 1, tap >> 1), shader.ivec2(0), size - shader.ivec2(1))
        guide = shader.load(signal, cell, 0)
        tent = shader.saturate(shader.vec2(1.0) - shader.abs(point - shader.vec2(cell)))
        weight = tent.x * tent.y * shader.saturate(1.0 - shader.abs(guide.y - log_depth) / band)
        obscured += weight * (1.0 - guide.x)
        total += weight
    return 1.0 - obscured / shader.max(total, 1e-5)


def adjust_diffuse_lighting(diffuse: shader.vec4, visibility: shader.Color2D, uv: shader.vec2,
                            depth: float, intensity: float, saturation: float,
                            strength: float, contrast: float) -> shader.vec4:
    scale = intensity
    if strength > 0.0 and diffuse.a > 0.0:
        contact = contact_visibility(visibility, uv, linear_eye_depth(depth, shader.camera()))
        # Scale the odds of occlusion. Fully open and fully occluded samples stay fixed.
        contact /= 1.0 + (contrast - 1.0) * (1.0 - contact)
        scale *= shader.mix(1.0, contact, strength)
    color = shader.mix(diffuse.rgb, shader.vec3(luminance(diffuse.rgb)), 1.0 - saturation)
    return shader.vec4(color * scale, diffuse.a)
