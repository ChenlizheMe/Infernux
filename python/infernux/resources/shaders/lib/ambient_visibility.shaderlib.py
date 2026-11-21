import infernux.shader as shader
from infernux.shader.lib.color import luminance


def ambient_occlusion_factor(visibility: shader.Color2D, pixel: shader.ivec2, strength: float, contrast: float) -> float:
    if strength <= 0.0:
        return 1.0
    contact = shader.load(visibility, pixel, 0).x
    contact /= 1.0 + (contrast - 1.0) * (1.0 - contact)
    return shader.mix(shader.pow(shader.saturate(0.1 + contact), 2.0 * strength), 1.0, 0.4)


def adjust_diffuse_lighting(diffuse: shader.vec4, intensity: float, saturation: float) -> shader.vec4:
    if intensity <= 0.0 or diffuse.a <= 0.0:
        return shader.vec4(0.0)
    color = shader.mix(diffuse.rgb, shader.vec3(luminance(diffuse.rgb)), 1.0 - saturation)
    # Scale the resolved coverage without promoting a partial estimate to full coverage.
    coverage = shader.saturate(diffuse.a) * shader.saturate(intensity)
    return shader.vec4(color * shader.max(1.0, intensity), coverage)
