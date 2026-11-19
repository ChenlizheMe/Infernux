#pragma once

#include "SlangCompiler.h"

namespace infernux
{

// All graphics stages read one view-owned camera allocation. The public
// context uses canonical clip coordinates; target lowering handles the API.
inline const ShaderSourceModule &GetSlangCameraModule()
{
    static const ShaderSourceModule module{"inx_camera", "<infernux/shader/camera>", R"slang(
import inx_geometry;
[[vk::binding(5, 1)]] public ConstantBuffer<InxCameraData> inx_view;
// The engine stores camera matrices in top-left raster space on both backends.
public float2 inx_raster_motion(float4 current_clip, float4 previous_clip)
{
    return 0.5 * (current_clip.xy / max(abs(current_clip.w), 1e-6) -
                  previous_clip.xy / max(abs(previous_clip.w), 1e-6));
}
public bool inx_mesh_has_motion_history(InxMeshInput input)
{
    return (input.object_flags & 1u) != 0u &&
           input.motion_current_clip.w > 0.0 && input.motion_previous_clip.w > 0.0;
}
public float2 inx_mesh_motion(InxMeshInput input)
{
    return inx_mesh_has_motion_history(input)
        ? inx_raster_motion(input.motion_current_clip, input.motion_previous_clip) : float2(0.0);
}
public float3 inx_camera_toward_eye(float3 position_ws)
{
    // Increasing reversed clip depth at fixed screen XY moves toward the eye.
    float4 direction = mul(inx_view.inverse_view_projection, float4(0.0, 0.0, 1.0, 0.0));
    return direction.xyz - position_ws * direction.w;
}
public struct InxCameraContext
{
    public column_major float4x4 view_matrix;
    public column_major float4x4 projection_matrix;
    public column_major float4x4 view_projection_matrix;
    public column_major float4x4 previous_view_projection_matrix;
    public column_major float4x4 inverse_view_projection_matrix;
    public float4 projection_params;
    public float4 depth_params;
};
public float4x4 inx_canonical_projection(float4x4 matrix)
{
    matrix[1] = -matrix[1];
    return matrix;
}
public InxCameraContext inx_camera_context()
{
    InxCameraContext result;
    result.view_matrix = inx_view.view;
    result.projection_matrix = inx_canonical_projection(inx_view.projection);
    result.view_projection_matrix = mul(result.projection_matrix, result.view_matrix);
    result.previous_view_projection_matrix = inx_canonical_projection(inx_view.previous_view_projection);
    // inverse(FlipY * VP) = inverse(VP) * FlipY.
    result.inverse_view_projection_matrix = mul(inx_view.inverse_view_projection,
        float4x4(1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1));
    result.projection_params = inx_view.projection_params;
    result.depth_params = inx_view.depth_params;
    return result;
}
)slang"};
    return module;
}

} // namespace infernux
