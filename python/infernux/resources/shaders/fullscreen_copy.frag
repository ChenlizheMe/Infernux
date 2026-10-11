#version 450

ShaderInfo {
    Name "Fullscreen Copy"
    Hidden On
    Capabilities [Fullscreen]
    Resources {
        Texture2D _SourceTex
    }
    Outputs {
        Float4 outColor
    }
}

// Equal-size render graph transfers preserve texels exactly. Interpolated UVs
// with a linear sampler can blend neighbours even at nominal pixel centres.
// Use Fullscreen Blit when the source actually needs resampling.
void main() {
    outColor = texelFetch(_SourceTex, ivec2(gl_FragCoord.xy), 0);
}
