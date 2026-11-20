import infernux.shader as shader


def pack_direction(m: shader.vec4) -> shader.uint:
    biased = shader.clamp(m.yzw / shader.max(m.x, 1.0e-20), shader.vec3(-1.0), shader.vec3(1.0)) * 511.0 + shader.vec3(511.5)
    q = shader.uvec3(biased)
    return q.x | (q.y << 10) | (q.z << 20)


def unpack_moment(dc: float, q: shader.uint) -> shader.vec4:
    bits = shader.uvec3(q, q >> 10, q >> 20) & shader.uvec3(1023)
    direction = (shader.vec3(bits) - shader.vec3(511.0)) / 511.0
    return shader.vec4(dc, dc * direction)


def diffuse_irradiance(red: shader.vec4, green: shader.vec4, blue: shader.vec4,
                       geometric: shader.vec3, shading: shader.vec3) -> shader.vec3:
    """Integrate the hemisphere on the face, then rotate the lobe to the material normal."""
    # Every observed direction lies above the face. Its first moment integrates
    # the face cosine exactly. The zonal second band estimates the normal-map change.
    axis_weight = 0.5 * (shading + geometric)
    diffuse = shader.vec3(shader.dot(axis_weight, red.yzw), shader.dot(axis_weight, green.yzw),
                          shader.dot(axis_weight, blue.yzw))
    luminance_axis = 0.2126 * red.yzw + 0.7152 * green.yzw + 0.0722 * blue.yzw
    axis = luminance_axis / shader.max(shader.length(luminance_axis), 1.0e-12)
    energy = shader.vec3(red.x, green.x, blue.x)
    first = shader.abs(shader.vec3(shader.dot(red.yzw, axis), shader.dot(green.yzw, axis), shader.dot(blue.yzw, axis)))
    rho = shader.saturate(first / shader.max(energy, shader.vec3(1.0e-12)))
    cosine = shader.dot(axis, shading)
    face_cosine = shader.dot(axis, geometric)
    return shader.max(shader.vec3(0.0), diffuse + energy * rho * (2.0 * rho - shader.vec3(1.0)) *
                      (0.46875 * (cosine * cosine - face_cosine * face_cosine)))
