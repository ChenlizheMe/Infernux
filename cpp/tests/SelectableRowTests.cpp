#include <function/renderer/gui/InxGUIContext.h>
#include <imgui.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

struct InkBounds
{
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    bool found = false;
    void Add(ImVec2 position)
    {
        found = true;
        minX = std::min(minX, position.x);
        minY = std::min(minY, position.y);
        maxX = std::max(maxX, position.x);
        maxY = std::max(maxY, position.y);
    }
};

static void CheckRow(float scale, float windowWidth)
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1200, 700);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char *atlas;
    int atlasWidth, atlasHeight;
    io.Fonts->GetTexDataAsRGBA32(&atlas, &atlasWidth, &atlasHeight);
    infernux::InxGUIContext context;
    ImVec2 clickPoint;
    int clicked = 0;
    // Let the new window finish its initial navigation/focus setup before
    // delivering a press/release gesture in the visible left inset.
    for (int frame = 0; frame < 6; ++frame) {
        if (frame > 0)
            io.AddMousePosEvent(clickPoint.x, clickPoint.y);
        if (frame == 3)
            io.AddMouseButtonEvent(0, true);
        if (frame == 4)
            io.AddMouseButtonEvent(0, false);
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(40, 40));
        ImGui::SetNextWindowSize(ImVec2(windowWidth, 200));
        ImGui::Begin("Rows", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        ImGui::PushFont(ImGui::GetFont(), 13.0f * scale);
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImDrawList *draw = ImGui::GetWindowDrawList();
        const int firstVertex = draw->VtxBuffer.Size;
        clicked += context.SelectableRow("WWW###stable_row", false, "WWW", {1, 0, 0, 1},
                                         28.0f * scale, 10.0f * scale);
        const ImVec2 rowMin = ImGui::GetItemRectMin();
        const ImVec2 rowMax = ImGui::GetItemRectMax();
        clickPoint = ImVec2(origin.x + 2.0f, (rowMin.y + rowMax.y) * 0.5f);
        InkBounds name, status;
        for (int index = firstVertex; index < draw->VtxBuffer.Size; ++index) {
            const auto &vertex = draw->VtxBuffer[index];
            if (vertex.col == ImGui::GetColorU32(ImGuiCol_Text))
                name.Add(vertex.pos);
            if (vertex.col == ImGui::GetColorU32(ImVec4(1, 0, 0, 1)))
                status.Add(vertex.pos);
        }
        assert(name.found && status.found);
        assert(name.minX >= origin.x + 10.0f * scale);
        assert(status.maxX <= rowMax.x - 10.0f * scale);
        assert(name.maxX < status.minX);
        const float rowCenter = (rowMin.y + rowMax.y) * 0.5f;
        assert(std::abs((name.minY + name.maxY) * 0.5f - rowCenter) <= 2.0f * scale);
        assert(std::abs((status.minY + status.maxY) * 0.5f - rowCenter) <= 2.0f * scale);
        assert(std::abs(name.minY - status.minY) < 0.01f);
        ImGui::PopFont();
        ImGui::End();
        ImGui::Render();
    }
    // The left inset remains part of the full-row input target.
    assert(clicked == 1);
    ImGui::DestroyContext();
}

int main()
{
    for (float scale : {1.0f, 2.0f})
        for (float width : {320.0f, 480.0f})
            CheckRow(scale, width);
}
