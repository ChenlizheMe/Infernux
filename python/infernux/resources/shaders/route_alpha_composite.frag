#version 450

ShaderInfo {
    Name "Route Alpha Composite"
    Hidden On
    Capabilities [Fullscreen]
    Resources {
        Texture2D _BaseTex
        Texture2D _LayerTex
    }
    Inputs {
        Float2 inUV
    }
    Outputs {
        Float4 outColor
    }
}

// Premultiplied-alpha composition for isolated queue, layer, and stage images.
// Rendering into a transparent route target through normal GPU blending
// produces premultiplied RGB. Keeping that representation across intermediate
// accumulators avoids dark fringes and double-multiplication.

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 base = texelFetch(_BaseTex, pixel, 0);
    vec4 layer = texelFetch(_LayerTex, pixel, 0);
    float inverseAlpha = 1.0 - layer.a;
    outColor = vec4(
        layer.rgb + base.rgb * inverseAlpha,
        layer.a + base.a * inverseAlpha
    );
}
