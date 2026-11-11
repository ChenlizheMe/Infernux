"""Shared author vocabulary; the GPU compiler owns numeric implementation."""

TYPES = {
    "float": "float", "int": "int", "uint": "uint", "bool": "bool",
    "f32": "float", "i32": "int", "u32": "uint",
    "vec2": "float2", "vec3": "float3", "vec4": "float4",
    "ivec2": "int2", "ivec3": "int3", "ivec4": "int4",
    "uvec2": "uint2", "uvec3": "uint3", "uvec4": "uint4",
    "mat3": "float3x3", "mat4": "float4x4",
}

INTRINSICS = {name: name for name in (
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "exp", "exp2", "log", "log2", "sqrt", "rsqrt", "abs", "sign",
    "floor", "ceil", "round", "trunc", "min", "max", "clamp", "saturate",
    "step", "smoothstep", "normalize", "length", "dot", "cross", "distance",
    "pow", "reflect", "refract", "ddx", "ddy", "fwidth", "isnan", "isinf", "isfinite",
    "radians", "degrees", "lerp", "frac", "mul", "transpose", "determinant", "all", "any",
)}
INTRINSICS.update(mix="lerp", fract="frac")
