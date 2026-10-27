#version 450

ShaderInfo {
    Name "Skybox Panoramic"
    Queue 32767
    Cull Back
    DepthWrite Off
    DepthTest Less_Equal
    PassTag Skybox
    CastShadows Off
    Capabilities [Standalone]
    Properties {
        Texture2D texSampler = white
        Color tint = [1.0, 1.0, 1.0, 1.0]
        Float exposure = 1.0
        Float rotation = 0.0
    }
    Inputs {
        Float3 fragWorldDir
    }
    Outputs {
        Float4 outColor
    }
}

void main() {
    vec3 direction = normalize(fragWorldDir);
    vec2 uv = vec2(fract(atan(direction.z, direction.x) / 6.28318530718 + 0.5
                         + material.rotation / 360.0),
                   acos(clamp(direction.y, -1.0, 1.0)) / 3.14159265359);
    outColor = vec4(texture(texSampler, uv).rgb * material.tint.rgb * material.exposure, 1.0);
}
