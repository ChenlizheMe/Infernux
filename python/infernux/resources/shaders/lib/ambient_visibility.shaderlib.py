import infernux.shader as shader
from infernux.shader.lib.color import luminance


def ambient_occlusion_factor(visibility: shader.Color2D, pixel: shader.ivec2, strength: float, contrast: float,
                             visibility_bias: float, minimum_light: float) -> float:
    if strength <= 0.0:
        return 1.0
    contact = shader.load(visibility, pixel, 0).x
    contact /= 1.0 + (contrast - 1.0) * (1.0 - contact)
    return shader.mix(shader.pow(shader.saturate(visibility_bias + contact), 2.0 * strength), 1.0, minimum_light)


def adjust_diffuse_lighting(diffuse: shader.vec4, intensity: float, saturation: float) -> shader.vec4:
    color = shader.mix(diffuse.rgb, shader.vec3(luminance(diffuse.rgb)), 1.0 - saturation)
    return shader.vec4(color * intensity, diffuse.a)
