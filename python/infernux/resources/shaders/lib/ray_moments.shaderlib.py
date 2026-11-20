import infernux.shader as shader


def pack_direction(m: shader.vec4) -> shader.uint:
    biased = shader.clamp(m.yzw / shader.max(m.x, 1.0e-20), shader.vec3(-1.0), shader.vec3(1.0)) * 511.0 + shader.vec3(511.5)
    q = shader.uvec3(biased)
    return q.x | (q.y << 10) | (q.z << 20)


def unpack_moment(dc: float, q: shader.uint) -> shader.vec4:
    bits = shader.uvec3(q, q >> 10, q >> 20) & shader.uvec3(1023)
    direction = (shader.vec3(bits) - shader.vec3(511.0)) / 511.0
    return shader.vec4(dc, dc * direction)


def sample_direction(packet: shader.uvec4) -> shader.vec3:
    return unpack_moment(1.0, packet.z).yzw


def diffuse_irradiance(red: shader.vec4, green: shader.vec4, blue: shader.vec4,
                       geometric: shader.vec3, shading: shader.vec3) -> shader.vec3:
    """Integrate the hemisphere on the face, then rotate the lobe to the material normal."""
    # The geometric cosine moment fixes the irradiance scale. The fitted ZH3
    # lobe supplies only the change between geometric and material normals.
    axis_weight = 0.5 * (shading + geometric)
    lambertian = shader.vec3(shader.dot(axis_weight, red.yzw), shader.dot(axis_weight, green.yzw),
                             shader.dot(axis_weight, blue.yzw))
    first = 0.2126 * red.yzw + 0.7152 * green.yzw + 0.0722 * blue.yzw
    axis = first / shader.max(shader.length(first), 1.0e-12)
    energy = shader.vec3(red.x, green.x, blue.x)
    axial = shader.abs(shader.vec3(shader.dot(red.yzw, axis), shader.dot(green.yzw, axis),
                                   shader.dot(blue.yzw, axis)))
    # Convert unnormalized Cartesian moments to the orthonormal SH1/SH0 ratio.
    ratio = 1.7320508075688772 * shader.saturate(axial / shader.max(energy, shader.vec3(1.0e-12)))
    fitted = energy * ratio * (0.08 + 0.60 * ratio)
    c = shader.dot(axis, shading)
    base = shader.dot(axis, geometric)
    # Y20, the unnormalized Lambert lobe, and the DC normalization combine
    # into 3*sqrt(5)/32 after subtracting the geometric-normal evaluation.
    return shader.max(shader.vec3(0.0), lambertian + 0.2096313728906053 * fitted * (c*c - base*base))
