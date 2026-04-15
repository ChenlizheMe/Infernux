@shader_id: lib/utils

@import: lib/common
@import: lib/color
@import: lib/noise
@import: lib/shapes
@import: lib/uv
@import: lib/texture_utils
@import: lib/lighting_utils
@import: lib/vertex_utils

// ============================================================================
// lib/utils.glsl — General-purpose shader utility toolkit
//
// Aggregates all context-free utility libraries into a single import.
// No UBO or varying dependencies — works in ANY shader type
// (surface, fullscreen, post-processing, compute, etc.)
//
// Usage: @import: lib/utils
//
// Includes:
//   lib/common          — constants, remap, saturate, comparison, wave, etc.
//   lib/color           — sRGB, HSV, HSL, brightness, contrast, blend modes
//   lib/noise           — hash, value/gradient/simplex noise, fbm, voronoi
//   lib/shapes          — SDF: circle, ellipse, rect, ring, polygon, star, etc.
//   lib/uv              — tiling, rotation, flipbook, parallax, triplanar, etc.
//   lib/texture_utils   — normal blending, detail blend, height blend, LOD, etc.
//   lib/lighting_utils  — fresnel, GGX, Cook-Torrance, rim, attenuation, SSS
//   lib/vertex_utils    — billboard, wind, displacement, Gerstner wave, morph
//
// For individual imports, use the specific library (e.g. @import: lib/noise).
// ============================================================================

// All constants (PI, EPSILON, etc.) and utility functions (saturate, etc.)
// are provided by lib/common — no duplicates needed here.
