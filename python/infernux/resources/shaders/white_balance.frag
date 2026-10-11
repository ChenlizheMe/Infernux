#version 450

ShaderInfo {
    Name "White Balance"
    Hidden On
    Imports ["Lib Color"]
    Capabilities [Fullscreen]
    Resources {
        Texture2D _SourceTex
    }
    PushConstants pc {
        Float temperature
        Float tint
        Float _pad0
        Float _pad1
    }
    Inputs {
        Float2 inUV
    }
    Outputs {
        Float4 outColor
    }
}

// White Balance post-process — adjusts color temperature and tint.
// Matches Unity URP White Balance.
//
// Push constants:
//   [0] temperature — color temperature shift (-100 to 100, 0 = neutral)
//   [1] tint        — green-magenta tint (-100 to 100, 0 = neutral)

void main() {
    vec4 source = texture(_SourceTex, inUV);
    outColor = vec4(whiteBalance(source.rgb, pc.temperature, pc.tint), source.a);
}
