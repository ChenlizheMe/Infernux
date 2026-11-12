"""Shared author vocabulary; the GPU compiler owns numeric implementation."""

from infernux._compiler.value_types import VALUE_TYPES as TYPES

INTRINSICS = {name: name for name in (
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "exp", "exp2", "log", "log2", "sqrt", "rsqrt", "abs", "sign",
    "floor", "ceil", "round", "trunc", "min", "max", "clamp", "saturate",
    "step", "smoothstep", "normalize", "length", "dot", "cross", "distance",
    "pow", "reflect", "refract", "ddx", "ddy", "fwidth", "isnan", "isinf", "isfinite",
    "radians", "degrees", "lerp", "frac", "mul", "transpose", "determinant", "all", "any",
    "rcp", "mad",
)}
INTRINSICS.update(
    mix="lerp", fract="frac", as_float="asfloat", as_int="asint", as_uint="asuint",
    f16_to_f32="f16tof32", f32_to_f16="f32tof16", count_bits="countbits",
    reverse_bits="reversebits", first_bit_high="firstbithigh", first_bit_low="firstbitlow",
)
