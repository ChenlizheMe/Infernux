import infernux.shader as shader


def sh_facing(view_normal: shader.vec3) -> float:
    projection = shader.camera().projection_matrix
    # Reverse-Z perspective and orthographic projections both increase
    # clip-W minus clip-Z in the forward direction.
    toward_eye = -shader.sign(projection[3][2] - projection[2][2])
    return shader.max(0.3, toward_eye * view_normal.z)


def sh_bilateral(depth: float, other_depth: float, facing: float, visibility: float, other_visibility: float) -> float:
    relative_depth = shader.abs(1.0 - other_depth / shader.max(depth, 1.0e-8))
    return shader.exp2(-shader.min(30.0, 80.0 * facing * relative_depth) -
                       20.0 * shader.abs(visibility - other_visibility))


def sh_normal_weight(normal: shader.vec3, other: shader.vec3) -> float:
    alignment = shader.dot(normal, other)
    falloff = 1.0 - 0.9 * alignment * alignment
    return 0.01 / (falloff * falloff)


def sh_clip_history(coefficients: shader.vec4) -> shader.vec4:
    return shader.vec4(shader.max(0.0, coefficients.x), shader.clamp(coefficients.yzw, -1024.0, 1024.0))


def sh_guide_phase(index: shader.uint) -> shader.ivec2:
    phase = shader.int(index & shader.uint(3))
    return shader.ivec2(phase >> 1, (phase ^ (phase >> 1)) & 1)


def sh_guide_pixel(cell: shader.ivec2, grid: shader.ivec2, full: shader.ivec2, index: shader.uint) -> shader.ivec2:
    half = shader.max(full / 2, shader.ivec2(1))
    return shader.clamp(2 * (cell * half / grid) + sh_guide_phase(index), shader.ivec2(0), full - 1)


def sh_guide_normal(packed: shader.uint) -> shader.vec3:
    bits = shader.ivec3(shader.uvec3(packed, packed >> 8, packed >> 16) & shader.uvec3(255))
    return shader.vec3(bits - (bits >> 7) * 256) / 127.0


def sh_blue_noise(texture: shader.Color2D, pixel: shader.ivec2, index: shader.uint) -> shader.vec2:
    offset = shader.frac(0.5 + shader.float(index) * shader.vec2(0.7548776865, 0.5698403120))
    texel = (pixel + shader.ivec2(offset * 256.0)) & shader.ivec2(255)
    return shader.load(texture, texel, 0).xy * (255.0 / 256.0) + (0.5 / 256.0)


def sh_resolve_offset(pattern: shader.uint, tap: int) -> shader.ivec2:
    if tap == 0:
        return shader.ivec2(0)
    # Five offsets in six-bit fields. Each signed coordinate spans [-2, 2].
    offsets = shader.uvec4(612369161, 403289690, 471905354, 25773075)
    bits = offsets[pattern] >> (6 * (tap - 1))
    return shader.ivec2(bits & shader.uint(7), (bits >> 3) & shader.uint(7)) - shader.ivec2(2)
