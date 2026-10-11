#version 450

ShaderInfo {
    Name "Route Additive Delta"
    Hidden On
    Capabilities [Fullscreen]
    Resources {
        Texture2D _OriginalTex
        Texture2D _ProcessedTex
    }
    Inputs {
        Float2 inUV
    }
    Outputs {
        Float4 outColor
    }
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec3 original = texelFetch(_OriginalTex, pixel, 0).rgb;
    vec3 processed = texelFetch(_ProcessedTex, pixel, 0).rgb;
    outColor = vec4(max(processed - original, vec3(0.0)), 0.0);
}
