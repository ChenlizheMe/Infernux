#version 450
@shader_id: error
@pass_tag: opaque
@hidden

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    // Procedural purple-black checkerboard pattern
    // Use both world position and UV for a consistent grid regardless of mesh scale
    float scale = 4.0;  // checkerboard density
    vec2 uv = fragTexCoord * scale;

    // Also blend with world-space grid so it's visible even with bad UVs
    vec2 worldGrid = fragWorldPos.xz * 2.0;

    // Pick whichever gives a more visible pattern
    float uvChecker  = step(0.5, fract(uv.x))  + step(0.5, fract(uv.y));
    float worldChecker = step(0.5, fract(worldGrid.x)) + step(0.5, fract(worldGrid.y));

    // XOR pattern: 0 or 2 → cell A, 1 → cell B
    float uvPattern    = mod(uvChecker, 2.0);
    float worldPattern = mod(worldChecker, 2.0);

    // Combine: prefer UV-based, mix in world-based
    float pattern = mix(uvPattern, worldPattern, 0.3);
    float checker = step(0.5, pattern);

    // Purple (#FF00FF) and black
    vec3 purple = vec3(1.0, 0.0, 1.0);
    vec3 black  = vec3(0.0, 0.0, 0.0);
    vec3 color  = mix(black, purple, checker);

    outColor = vec4(color, 1.0);
}
