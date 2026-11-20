import infernux.shader as shader


def sh_bilateral(depth: float, other_depth: float, facing: float, visibility: float, other_visibility: float) -> float:
    relative_depth = shader.abs(1.0 - other_depth / shader.max(depth, 1.0e-8))
    return shader.exp2(-shader.min(30.0, 80.0 * facing * relative_depth) -
                       20.0 * shader.abs(visibility - other_visibility))


def sh_normal_weight(normal: shader.vec3, other: shader.vec3) -> float:
    alignment = shader.max(0.0, shader.dot(normal, other))
    falloff = 1.0 - 0.9 * alignment * alignment
    return 0.01 / (falloff * falloff)



def sh_resolve_offset(pattern: shader.uint, tap: int) -> shader.ivec2:
    if tap == 0:
        return shader.ivec2(0)
    # Five offsets in six-bit fields. Each signed coordinate spans [-2, 2].
    offsets = shader.uvec4(612369161, 403289690, 471905354, 25773075)
    bits = offsets[pattern] >> (6 * (tap - 1))
    return shader.ivec2(bits & shader.uint(7), (bits >> 3) & shader.uint(7)) - shader.ivec2(2)
