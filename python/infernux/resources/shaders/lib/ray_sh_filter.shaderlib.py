import infernux.shader as shader


def sh_bilateral(depth: float, other_depth: float, facing: float, visibility: float, other_visibility: float) -> float:
    relative_depth = shader.abs(1.0 - other_depth / shader.max(depth, 1.0e-8))
    return shader.exp2(-shader.min(30.0, 80.0 * facing * relative_depth) -
                       20.0 * shader.abs(visibility - other_visibility))


def sh_normal_weight(normal: shader.vec3, other: shader.vec3) -> float:
    alignment = shader.dot(normal, other)
    falloff = 1.0 - 0.9 * alignment * alignment
    return 0.01 / (falloff * falloff)


def sh_guide_pixel(cell: shader.ivec2, grid: shader.ivec2, full: shader.ivec2, index: shader.uint) -> shader.ivec2:
    # Rotate four representative subpixels in a fixed order for a view.
    phase = shader.int(index & shader.uint(3))
    offset = shader.ivec2(phase >> 1, (phase ^ (phase >> 1)) & 1)
    center = (shader.vec2(cell) + 0.5) * shader.vec2(full) / shader.vec2(grid)
    return shader.clamp(shader.ivec2(shader.floor(center - 1.0)) + offset,
                        shader.ivec2(0), full - shader.ivec2(1))


def sh_resolve_offset(pattern: shader.uint, tap: int) -> shader.ivec2:
    if tap == 0:
        return shader.ivec2(0)
    # Five offsets in six-bit fields. Each signed coordinate spans [-2, 2].
    offsets = shader.uvec4(612369161, 403289690, 471905354, 25773075)
    bits = offsets[pattern] >> (6 * (tap - 1))
    return shader.ivec2(bits & shader.uint(7), (bits >> 3) & shader.uint(7)) - shader.ivec2(2)
