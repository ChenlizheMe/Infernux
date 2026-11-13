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
        clicked += context.SelectableRow("WWW###stable_row", false, "WWW", {1, 0, 0, 1}, 28.0f * scale, 10.0f * scale);
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

static void CheckPropertyGroups(float scale, float windowWidth)
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
    // Numeric fields require the mono face; the default atlas font stands in.
    infernux::InxGUIContext::s_monoFont = io.Fonts->Fonts[0];
    for (int frame = 0; frame < 4; ++frame) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(40, 40));
        ImGui::SetNextWindowSize(ImVec2(windowWidth, 200));
        ImGui::SetNextWindowContentSize(ImVec2(0, 700));
        ImGui::SetNextWindowScroll(ImVec2(0, frame > 0 ? 65.0f : 0.0f));
        ImGui::Begin("Properties", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings);
        ImGui::PushFont(ImGui::GetFont(), 13.0f * scale);
        ImGui::Indent(20);
        ImGui::BeginGroup();
        const auto *draw = ImGui::GetWindowDrawList();
        const int firstVertex = draw->VtxBuffer.Size;
        const std::array<float, 4> background{0.125f, 0.25f, 0.375f, 1};
        const auto frameOrigin = ImGui::GetCursorScreenPos();
        const float frameAvailable = ImGui::GetContentRegionAvail().x;
        context.BeginFramedGroup(3, 2, 0, background, {0, 0, 0, 1});
        assert(std::abs(ImGui::GetCursorScreenPos().x - frameOrigin.x - 3) < 0.1f);
        assert(std::abs(ImGui::GetContentRegionAvail().x - (frameAvailable - 6)) < 0.1f);
        for (const auto *label : {"Speed", "A deliberately very long property name"}) {
            ImGui::PushID(label);
            const auto origin = ImGui::GetCursorScreenPos();
            const float available = ImGui::GetContentRegionAvail().x;
            context.FieldLabel(label, 140);
            float value = 1;
            context.DragFloat("##value", &value);
            const auto minimum = ImGui::GetItemRectMin();
            assert(std::abs(minimum.x - origin.x - std::min(140.0f, std::max(0.0f, available - 48.0f))) < 0.1f);
            assert(std::abs(minimum.y - origin.y) < 0.1f);
            assert(ImGui::GetItemRectMax().x <= frameOrigin.x + frameAvailable - 3.0f + 0.1f);
            ImGui::PopID();
        }
        context.BeginFramedGroup(0, 0, 0, background, {0, 0, 0, 1});
        ImGui::Dummy(ImVec2(10, frame % 2 ? 60.0f : 10.0f));
        context.EndFramedGroup();
        context.EndFramedGroup();
        const auto minimum = ImGui::GetItemRectMin();
        const auto maximum = ImGui::GetItemRectMax();
        InkBounds fill;
        for (int index = firstVertex; index < draw->VtxBuffer.Size; ++index)
            if (draw->VtxBuffer[index].col ==
                ImGui::GetColorU32(ImVec4(background[0], background[1], background[2], 1)))
                fill.Add(draw->VtxBuffer[index].pos);
        assert(fill.found);
        assert(std::abs(fill.minX - frameOrigin.x) < 0.1f);
        assert(std::abs(fill.maxX - (frameOrigin.x + frameAvailable)) < 0.1f);
        assert(std::abs(fill.minY - minimum.y) < 0.1f);
        assert(std::abs(fill.maxY - maximum.y) < 0.1f);
        assert(std::abs(ImGui::GetContentRegionAvail().x - frameAvailable) < 0.1f);
        ImGui::EndGroup();
        ImGui::Unindent(20);
        ImGui::PopFont();
        ImGui::End();
        ImGui::Render();
    }
    infernux::InxGUIContext::s_monoFont = nullptr;
    ImGui::DestroyContext();
}

static void CheckAutoHeightIconButton()
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(640, 480);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char *atlas;
    int atlasWidth, atlasHeight;
    io.Fonts->GetTexDataAsRGBA32(&atlas, &atlasWidth, &atlasHeight);
    infernux::InxGUIContext context;
    context.resolveEditorIcon = [](const std::string &) { return uint64_t{1}; };
    io.Fonts->SetTexID(static_cast<ImTextureID>(2));
    for (int frame = 0; frame < 3; ++frame) {
        ImGui::NewFrame();
        ImGui::SetNextWindowSize(ImVec2(420, 280));
        ImGui::Begin("Inline icon", nullptr, ImGuiWindowFlags_NoSavedSettings);
        context.IconButton("close", "glyphs/close", "", 20.0f, 0.0f);
        assert(ImGui::GetItemRectSize().y > 6.0f);
        bool imageDrawn = false;
        for (const ImDrawCmd &command : ImGui::GetWindowDrawList()->CmdBuffer)
            imageDrawn |= command.GetTexID() == static_cast<ImTextureID>(1) && command.ElemCount >= 6;
        assert(imageDrawn && "Automatic button height must still draw its icon");
        ImGui::End();
        ImGui::Render();
    }
    ImGui::DestroyContext();
}

static void CheckResizablePopup()
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1280, 720);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char *atlas;
    int atlasWidth, atlasHeight;
    io.Fonts->GetTexDataAsRGBA32(&atlas, &atlasWidth, &atlasHeight);
    infernux::InxGUIContext context;
    int observed = 0;
    for (int frame = 0; frame < 5; ++frame) {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(40, 40));
        ImGui::SetNextWindowSize(ImVec2(640, 480));
        ImGui::Begin("Popup owner", nullptr, ImGuiWindowFlags_NoSavedSettings);
        if (frame == 0)
            context.OpenPopup("curve");
        ImGui::SetNextWindowPos(ImVec2(100, 100), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(420, 280), ImGuiCond_FirstUseEver);
        if (context.BeginPopup("curve", true)) {
            ImGui::TextUnformatted("Short content must not shrink the editor window");
            assert(std::abs(ImGui::GetWindowWidth() - 420.0f) < 0.1f);
            ++observed;
            context.EndPopup();
        }
        ImGui::End();
        ImGui::Render();
    }
    assert(observed == 5);
    ImGui::DestroyContext();
}

static void CheckRangedSliderEditReporting(bool integer)
{
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(640, 480);
    io.DeltaTime = 1.0f / 60.0f;
    unsigned char *atlas;
    int atlasWidth, atlasHeight;
    io.Fonts->GetTexDataAsRGBA32(&atlas, &atlasWidth, &atlasHeight);
    infernux::InxGUIContext::s_monoFont = io.Fonts->Fonts[0];
    infernux::InxGUIContext context;
    ImVec2 clickPoint;
    float value = 0;
    int integerValue = 0;
    for (int frame = 0; frame < 5; ++frame) {
        if (frame == 2) {
            io.AddMousePosEvent(clickPoint.x, clickPoint.y);
            io.AddMouseButtonEvent(0, true);
        } else if (frame == 3) {
            io.AddMouseButtonEvent(0, false);
        }
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(20, 20));
        ImGui::SetNextWindowSize(ImVec2(420, 200));
        ImGui::Begin("Ranged scalar", nullptr, ImGuiWindowFlags_NoSavedSettings);
        ImGui::SetNextItemWidth(300);
        if (integer)
            context.IntSlider("amount", &integerValue, 0, 100);
        else
            context.FloatSlider("amount", &value, 0, 1);
        const ImVec2 lo = ImGui::GetItemRectMin();
        const ImVec2 hi = ImGui::GetItemRectMax();
        assert(hi.x - lo.x >= 299);
        if (frame < 2) {
            clickPoint = ImVec2(lo.x + 100, (lo.y + hi.y) * 0.5f);
            assert(!context.IsItemEdited());
        } else if (frame == 2) {
            assert(integer ? integerValue > 20 : value > 0.2f);
            assert(context.IsItemEdited());
            assert(ImGui::IsItemActive());
        } else if (frame == 3) {
            assert(ImGui::IsItemDeactivatedAfterEdit());
        }
        ImGui::End();
        ImGui::Render();
    }
    infernux::InxGUIContext::s_monoFont = nullptr;
    ImGui::DestroyContext();
}

int main()
{
    CheckRangedSliderEditReporting(false);
    CheckRangedSliderEditReporting(true);
    CheckAutoHeightIconButton();
    CheckResizablePopup();
    for (float scale : {1.0f, 2.0f})
        for (float width : {320.0f, 480.0f}) {
            CheckRow(scale, width);
            CheckPropertyGroups(scale, width);
        }
}
