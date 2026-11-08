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

float softGrain(vec2 p, uint seed) {
    uvec2 cell = uvec2(ivec2(floor(p)));
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float bottom = mix(grainSample(cell, seed), grainSample(cell + uvec2(1, 0), seed), f.x);
    float top = mix(grainSample(cell + uvec2(0, 1), seed), grainSample(cell + uvec2(1, 1), seed), f.x);
    return mix(bottom, top, f.y);
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 source = texelFetch(_SourceTex, pixel, 0);
    if (pc.intensity <= 0.0) {
        outColor = source;
        return;
    }

    vec2 plate = (vec2(pixel) + 0.5) / max(pc.size, 0.5);
    uint frame = uint(mod(floor(_Globals._Time.x * 24.0), 65536.0));
    uint seed = frame * 5u;
    const mat2 rotate = mat2(.8, .6, -.6, .8);
    float density = .74 * softGrain(rotate * plate, seed)
                  + .26 * softGrain(plate * 1.8 + vec2(71.0, 13.0), seed + 1u);
    vec3 grain = vec3(density);
    if (pc.colored > 0.5) {
        grain = mix(grain, vec3(softGrain(plate, seed + 2u),
                                softGrain(plate, seed + 3u),
                                softGrain(plate, seed + 4u)), 0.30);
    }

    float luma = clamp(dot(source.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
    float amplitude = pc.intensity * mix(1.0, 1.0 - sqrt(luma), pc.response);
    outColor = vec4(max(source.rgb * (1.0 + grain * amplitude), vec3(0.0)), source.a);
}
