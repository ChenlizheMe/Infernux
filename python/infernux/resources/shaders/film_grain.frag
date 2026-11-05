#version 450

ShaderInfo {
    Name "Film Grain"
    Hidden On
    Capabilities [Fullscreen, EngineGlobals]
    Resources {
        Texture2D _SourceTex
    }
    PushConstants pc {
        Float intensity
        Float response
        Float size
        Float colored
    }
    Inputs {
        Float2 inUV
    }
    Outputs {
        Float4 outColor
    }
}

// A neutral procedural grain plate, refreshed at film cadence. The plate
// never supplies coordinates for the scene color.
uint grainBits(uvec3 cell) {
    uint bits = cell.x * 1973u ^ cell.y * 9277u ^ cell.z * 26699u;
    bits ^= bits >> 16u;
    bits *= 0x7feb352du;
    bits ^= bits >> 15u;
    bits *= 0x846ca68bu;
    return bits ^ (bits >> 16u);
}

float grainSample(uvec2 cell, uint seed) {
    uint bits = grainBits(uvec3(cell, seed));
    // Two independent halves give a centered, triangular density.
    return (float(bits & 65535u) + float(bits >> 16u)) / 65535.0 - 1.0;
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 source = texelFetch(_SourceTex, pixel, 0);
    if (pc.intensity <= 0.0) {
        outColor = source;
        return;
    }

    vec2 plate = (vec2(pixel) + 0.5) / max(pc.size, 0.5);
    uvec2 cell = uvec2(plate);
    uint frame = uint(mod(floor(_Globals._Time.x * 24.0), 65536.0));
    vec3 grain = vec3(grainSample(cell, frame * 3u));
    if (pc.colored > 0.5) {
        grain.g = mix(grain.r, grainSample(cell, frame * 3u + 1u), 0.45);
        grain.b = mix(grain.r, grainSample(cell, frame * 3u + 2u), 0.45);
    }

    float luma = clamp(dot(source.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    float amplitude = pc.intensity * mix(1.0, 1.0 - sqrt(luma), pc.response);
    outColor = vec4(max(source.rgb * (1.0 + grain * amplitude), vec3(0.0)), source.a);
}
