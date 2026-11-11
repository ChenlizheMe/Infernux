"""Native GPU value vocabulary shared by Shader and numerical Compute.

This module contains declarations only: importing types does not load NumPy,
a host compiler, or a device. Backend storage conversion belongs to Compute.
"""

SCALAR_STORAGE_TYPES = {"float32": "float", "int32": "int", "uint32": "uint", "int64": "int64_t"}

VALUE_TYPES = {
    "float": "float", "int": "int", "uint": "uint", "bool": "bool",
    "f32": "float", "i32": "int", "u32": "uint",
    "vec2": "float2", "vec3": "float3", "vec4": "float4",
    "ivec2": "int2", "ivec3": "int3", "ivec4": "int4",
    "uvec2": "uint2", "uvec3": "uint3", "uvec4": "uint4",
    "bvec2": "bool2", "bvec3": "bool3", "bvec4": "bool4",
    "mat2": "float2x2", "mat3": "float3x3", "mat4": "float4x4",
}
