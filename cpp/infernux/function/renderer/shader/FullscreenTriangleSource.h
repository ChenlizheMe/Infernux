#pragma once

#include <string_view>

namespace infernux
{

inline constexpr std::string_view FullscreenTriangleVertexSource = R"(
[shader("vertex")]
float4 vertex_main(uint id : SV_VulkanVertexID) : SV_Position {
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * 2 - 1, 0, 1);
}
)";

} // namespace infernux
