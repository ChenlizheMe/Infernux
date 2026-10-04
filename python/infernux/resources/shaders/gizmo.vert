#version 450

ShaderInfo {
    Name "Gizmo"
    Hidden On
    Capabilities [Standalone, NoMotionVectors]
    Outputs {
        Float3 fragColor
    }
}

// UBO: model, view, projection matrices

void main() {
    // Gizmos share the ordinary draw-list instance transform.  Reading the
    // per-instance matrix keeps dynamic and batched Gizmos on the same
    // Object-to-World publication as the mesh they describe; pc.model only
    // represents the first member of an instanced batch.
    mat4 objectToWorld = instanceModels[gl_InstanceIndex];
    gl_Position = ubo.proj * ubo.view * objectToWorld * vec4(inPosition, 1.0);
    fragColor = inColor;
}
