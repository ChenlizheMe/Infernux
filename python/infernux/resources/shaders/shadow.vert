#version 450

ShaderInfo {
    Name "infernux/internal/shadow"
    Hidden On
    PassTag Shadow
    Capabilities [Standalone]
}

void main() {
    gl_Position = ubo.proj * ubo.view * pc.model * vec4(inPosition, 1.0);
}
