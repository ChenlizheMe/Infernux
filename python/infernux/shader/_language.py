"""Shared author vocabulary; the GPU compiler owns numeric implementation."""

TYPES = {
    "float": "float", "int": "int", "uint": "uint", "bool": "bool",
    "vec2": "float2", "vec3": "float3", "vec4": "float4",
    "ivec2": "int2", "ivec3": "int3", "ivec4": "int4",
    "uvec2": "uint2", "uvec3": "uint3", "uvec4": "uint4",
    "mat3": "float3x3", "mat4": "float4x4",
}

INTRINSICS = {name: name for name in (
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "exp", "exp2", "log", "log2", "sqrt", "rsqrt", "abs", "sign",
    "floor", "ceil", "round", "trunc", "min", "max", "clamp", "saturate",
    "step", "smoothstep", "normalize", "length", "dot", "cross", "distance",
    "pow", "reflect", "refract", "ddx", "ddy", "fwidth",
)}
INTRINSICS.update(mix="lerp", fract="frac")
