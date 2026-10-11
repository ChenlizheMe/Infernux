#version 450

ShaderInfo {
    Name "Route Additive Composite"
    Hidden On
    Capabilities [Fullscreen]
    Resources {
        Texture2D _BaseTex
        Texture2D _AdditiveTex
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
    vec4 base = texelFetch(_BaseTex, pixel, 0);
    vec3 additive = texelFetch(_AdditiveTex, pixel, 0).rgb;
    outColor = vec4(base.rgb + additive, base.a);
}
