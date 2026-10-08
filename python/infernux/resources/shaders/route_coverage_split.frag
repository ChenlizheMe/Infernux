#version 450

ShaderInfo {
    Name "Route Coverage Split"
    Hidden On
    Capabilities [Fullscreen]
    Resources {
        Texture2D _OriginalTex
        Texture2D _ProcessedTex
    }
    PushConstants pc {
        Float outside
    }
    Inputs {
        Float2 inUV
    }
    Outputs {
        Float4 outColor
    }
}

void main() {
    // Keep each pixel's premultiplied RGBA intact, including MSAA coverage.
    // Multiplying by coverage again would darken edges and lose Bloom energy.
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    bool geometry = texelFetch(_OriginalTex, pixel, 0).a > 0.0;
    bool selectOutside = pc.outside > 0.5;
    outColor = (geometry != selectOutside)
        ? texelFetch(_ProcessedTex, pixel, 0)
        : vec4(0.0);
}
