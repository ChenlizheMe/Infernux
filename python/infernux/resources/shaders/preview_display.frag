#version 450

ShaderInfo {
    Name "Preview Display"
    Hidden On
    Imports ["Lib Color"]
    Capabilities [Fullscreen]
    Resources {
        Texture2D source_tex
    }
    PushConstants pc {
        Float sphereRadius
        Float hdr
    }
    Inputs {
        Float2 inUV
    }
    Outputs {
        Float4 outColor
    }
}

// Both producers render at twice the display size. Process each HDR sample
// before the 2x2 reduction; alpha-weighted linear colors keep transparent edges
// free of black halos. The sRGB attachment, not this shader, encodes the result.
void main() {
    ivec2 origin = ivec2(gl_FragCoord.xy) * 2;
    float sourceSize = float(textureSize(source_tex, 0).x);
    vec3 premultiplied = vec3(0.0);
    float alphaSum = 0.0;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            ivec2 pixel = origin + ivec2(x, y);
            vec4 sampleColor = texelFetch(source_tex, pixel, 0);
            float alpha = clamp(sampleColor.a, 0.0, 1.0);
            vec3 color = max(sampleColor.rgb, vec3(0.0));
            if (pc.sphereRadius > 0.0) {
                float distance = length((vec2(pixel) + 0.5) / sourceSize - 0.5);
                float feather = 1.5 / sourceSize;
                alpha *= clamp((pc.sphereRadius + feather - distance) / (2.0 * feather), 0.0, 1.0);
                // Sphere materials draw against transparent black. Undo that
                // matte before display mapping; the GUI blends straight alpha.
                if (alpha > 0.0 && alpha < 0.999)
                    color /= alpha;
            }
            if (pc.hdr > 0.5)
                color = reinhardTonemap(color);
            premultiplied += color * alpha;
            alphaSum += alpha;
        }
    }
    outColor = alphaSum > 0.0 ? vec4(premultiplied / alphaSum, alphaSum * 0.25) : vec4(0.0);
}
