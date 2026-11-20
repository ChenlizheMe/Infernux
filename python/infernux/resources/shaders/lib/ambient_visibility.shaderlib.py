import infernux.shader as shader
from infernux.shader.lib.color import luminance


def adjust_diffuse_lighting(diffuse: shader.vec4, visibility: shader.Color2D, pixel: shader.ivec2,
                            intensity: float, saturation: float,
                            strength: float, contrast: float) -> shader.vec4:
    scale = intensity
    if strength > 0.0 and diffuse.a > 0.0:
        contact = shader.saturate(0.1 + shader.load(visibility, pixel, 0).x)
        contact /= 1.0 + (contrast - 1.0) * (1.0 - contact)
        scale *= shader.mix(shader.pow(contact, 2.0 * strength), 1.0, 0.4)
    color = shader.mix(diffuse.rgb, shader.vec3(luminance(diffuse.rgb)), 1.0 - saturation)
    return shader.vec4(color * scale, diffuse.a)
