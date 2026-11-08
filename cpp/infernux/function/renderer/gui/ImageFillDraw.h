#pragma once

#include <core/math/ImageFill.h>
#include <imgui_internal.h>

namespace infernux
{
inline void AppendFilledImage(ImDrawList &draw, ImTextureID texture, float minX, float minY, float maxX, float maxY,
                              float u0, float v0, float u1, float v1, ImU32 color, float amount, int method, int origin,
                              bool clockwise, float rotation, bool mirrorH, bool mirrorV)
{
    const auto geometry = imagefill::Build(amount, method, origin, clockwise);
    if (!geometry.count || !(color & IM_COL32_A_MASK))
        return;
    const bool textured = texture != 0;
    if (textured)
        draw.PushTextureID(texture);
    else
        draw.PushTextureID(ImGui::GetIO().Fonts->TexRef);
    const float angle = std::fmod(rotation, 360.0f) * 0.017453292519943295f;
    const float cosine = std::cos(angle), sine = std::sin(angle);
    const ImVec2 center((minX + maxX) * 0.5f, (minY + maxY) * 0.5f);
    for (size_t p = 0; p < geometry.count; ++p) {
        const auto &polygon = geometry.polygons[p];
        draw.PrimReserve(static_cast<int>((polygon.count - 2) * 3), static_cast<int>(polygon.count));
        const unsigned base = draw._VtxCurrentIdx;
        for (size_t i = 1; i + 1 < polygon.count; ++i) {
            draw.PrimWriteIdx(static_cast<ImDrawIdx>(base));
            draw.PrimWriteIdx(static_cast<ImDrawIdx>(base + i));
            draw.PrimWriteIdx(static_cast<ImDrawIdx>(base + i + 1));
        }
        for (size_t i = 0; i < polygon.count; ++i) {
            const auto point = polygon.points[i];
            float x = (point.x - 0.5f) * (maxX - minX), y = (0.5f - point.y) * (maxY - minY);
            if (mirrorH)
                x = -x;
            if (mirrorV)
                y = -y;
            const ImVec2 position(center.x + x * cosine - y * sine, center.y + x * sine + y * cosine);
            const ImVec2 uv =
                textured ? ImVec2(u0 + point.x * (u1 - u0), v1 - point.y * (v1 - v0)) : draw._Data->TexUvWhitePixel;
            draw.PrimWriteVtx(position, uv, color);
        }
    }
    draw.PopTextureID();
}
} // namespace infernux
