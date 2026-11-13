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
