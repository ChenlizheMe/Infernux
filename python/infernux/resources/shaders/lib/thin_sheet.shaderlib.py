import infernux.shader as shader
from infernux.shader.lib.camera import world_position


def thin_sheet_cosine(uv: shader.vec2, normal: shader.vec3, camera: shader.CameraContext) -> float:
    near = world_position(uv, 1.0, camera)
    middle = world_position(uv, 0.5, camera)
    return shader.saturate(shader.abs(shader.dot(normal, shader.normalize(near - middle))))


def thin_sheet_fresnel(cosine: float, ior: float) -> float:
    if ior == 1.0:
        return 0.0
    sin_squared = (1.0 - cosine * cosine) / (ior * ior)
    if sin_squared >= 1.0:
        return 1.0
    transmitted = shader.sqrt(1.0 - sin_squared)
    perpendicular = (cosine - ior * transmitted) / (cosine + ior * transmitted)
    parallel = (ior * cosine - transmitted) / (ior * cosine + transmitted)
    return 0.5 * (perpendicular * perpendicular + parallel * parallel)


def thin_sheet_attenuation(cosine: float, ior: float, optical_depth: shader.vec3) -> shader.vec3:
    transmitted = shader.sqrt(shader.max(1.0 - (1.0 - cosine * cosine) / (ior * ior), 0.0))
    return shader.exp(-optical_depth / shader.max(transmitted, 0.0001))
