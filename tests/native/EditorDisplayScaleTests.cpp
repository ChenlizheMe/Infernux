#include <platform/window/EditorDisplayScale.h>

#include <algorithm>
#include <cmath>
#include <imgui.h>
#include <iostream>

namespace
{
void Check(bool condition)
{
    if (!condition)
        throw std::runtime_error("editor draw geometry and pointer hit testing disagree");
}

void CheckFrame(float displayScale, float pixelDensity)
{
    auto &io = ImGui::GetIO();
    io.DisplaySize = ImVec2(std::round(3072.0f / pixelDensity), std::round(1980.0f / pixelDensity));
    io.DisplayFramebufferScale = ImVec2(3072.0f / io.DisplaySize.x, 1980.0f / io.DisplaySize.y);
    // SDL's pixel-density query is based on width. Integer window extents at
    // fractional scaling can produce slightly different X/Y framebuffer ratios.
    const float uiScale = infernux::ResolveEditorDisplayScale(displayScale, io.DisplayFramebufferScale.x);
    ImGui::GetStyle() = ImGuiStyle{};
    ImGui::GetStyle().ScaleAllSizes(uiScale);
    ImGui::GetStyle().FrameRounding = 0;
    ImGui::GetStyle().FrameBorderSize = 0;

    io.Fonts->Clear();
    ImFontConfig font;
    font.SizePixels = 18.0f * uiScale;
    io.Fonts->AddFontDefault(&font);
    // Same publication as ReloadGUIFont: Clear() can write the *previous*
    // current font size back into style during a warm reload in ImGui 1.92.
    ImGui::GetStyle().FontSizeBase = font.SizePixels;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    const ImU32 buttonColor = IM_COL32(37, 83, 139, 255);
    const ImU32 autoButtonColor = IM_COL32(139, 83, 37, 255);
    ImVec2 buttonMin, buttonMax;
    ImVec2 autoButtonMin, autoButtonMax;
    bool autoButtonClicked = false;
    bool autoButtonActive = false;
    auto frame = [&]() {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(80 * uiScale, 60 * uiScale));
        ImGui::SetNextWindowSize(ImVec2(600 * uiScale, 400 * uiScale));
        ImGui::Begin("DPI contract", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoTitleBar);
        Check(std::abs(ImGui::GetStyle().FontSizeBase - font.SizePixels) < 0.01f);
        Check(std::abs(ImGui::GetFontSize() - font.SizePixels) <= 1.0f); // raster size rounds to pixels
        Check(ImGui::CalcTextSize("DPI target").y == ImGui::GetFontSize());
        // An ordinary ImGui widget owns both the drawing and hit rectangle.
        ImGui::PushStyleColor(ImGuiCol_Button, buttonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, buttonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, buttonColor);
        const bool clicked = ImGui::Button("DPI target", ImVec2(180 * uiScale, 48 * uiScale));
        ImGui::PopStyleColor(3);
        buttonMin = ImGui::GetItemRectMin();
        buttonMax = ImGui::GetItemRectMax();
        ImGui::PushStyleColor(ImGuiCol_Button, autoButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, autoButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, autoButtonColor);
        autoButtonClicked = ImGui::Button("Automatic font-sized target");
        autoButtonActive = ImGui::IsItemActive();
        ImGui::PopStyleColor(3);
        autoButtonMin = ImGui::GetItemRectMin();
        autoButtonMax = ImGui::GetItemRectMax();
        Check(std::abs(ImGui::GetItemRectSize().y - ImGui::GetFontSize() - 2.0f * ImGui::GetStyle().FramePadding.y) <
              0.01f);
        ImGui::End();
        ImGui::Render();
        return clicked;
    };
    frame();
    frame();

    auto *draw = ImGui::GetDrawData();
    Check(std::abs(draw->DisplaySize.x * draw->FramebufferScale.x - 3072) < 0.01f);
    Check(std::abs(draw->DisplaySize.y * draw->FramebufferScale.y - 1980) < 0.01f);
    ImVec2 paintMin(1e20f, 1e20f), paintMax(-1e20f, -1e20f);
    ImVec2 autoPaintMin(1e20f, 1e20f), autoPaintMax(-1e20f, -1e20f);
    for (const auto *list : draw->CmdLists)
        for (const auto &vertex : list->VtxBuffer)
            if (vertex.col == buttonColor) {
                paintMin.x = std::min(paintMin.x, vertex.pos.x);
                paintMin.y = std::min(paintMin.y, vertex.pos.y);
                paintMax.x = std::max(paintMax.x, vertex.pos.x);
                paintMax.y = std::max(paintMax.y, vertex.pos.y);
            } else if (vertex.col == autoButtonColor) {
                autoPaintMin.x = std::min(autoPaintMin.x, vertex.pos.x);
                autoPaintMin.y = std::min(autoPaintMin.y, vertex.pos.y);
                autoPaintMax.x = std::max(autoPaintMax.x, vertex.pos.x);
                autoPaintMax.y = std::max(autoPaintMax.y, vertex.pos.y);
            }
    Check(std::abs(paintMin.x - buttonMin.x) < 0.01f && std::abs(paintMin.y - buttonMin.y) < 0.01f);
    Check(std::abs(paintMax.x - buttonMax.x) < 0.01f && std::abs(paintMax.y - buttonMax.y) < 0.01f);
    Check(std::abs((paintMax.x - paintMin.x) * io.DisplayFramebufferScale.x - 180 * displayScale) < 0.01f);
    Check(std::abs(autoPaintMin.x - autoButtonMin.x) < 0.01f && std::abs(autoPaintMin.y - autoButtonMin.y) < 0.01f);
    Check(std::abs(autoPaintMax.x - autoButtonMax.x) < 0.01f && std::abs(autoPaintMax.y - autoButtonMax.y) < 0.01f);

    // Derive a click from the actual painted framebuffer rectangle, then cross
    // the one framebuffer -> window boundary used by screenshot-driven tools.
    const ImVec2 pixelPoint((paintMin.x + paintMax.x) * 0.5f * io.DisplayFramebufferScale.x,
                            (paintMin.y + paintMax.y) * 0.5f * io.DisplayFramebufferScale.y);
    io.AddMousePosEvent(pixelPoint.x / io.DisplayFramebufferScale.x, pixelPoint.y / io.DisplayFramebufferScale.y);
    frame();
    io.AddMouseButtonEvent(0, true);
    Check(!frame());
    io.AddMouseButtonEvent(0, false);
    Check(frame());
    // A point beyond the painted rectangle must not activate the same button.
    io.AddMousePosEvent(buttonMax.x + 8, buttonMax.y + 8);
    frame();
    io.AddMouseButtonEvent(0, true);
    Check(!frame());
    io.AddMouseButtonEvent(0, false);
    Check(!frame());

    // Exercise the font-derived rectangle with real ImGui press/release events,
    // including the framebuffer conversion after each same-context font reload.
    const ImVec2 autoPixelPoint((autoPaintMin.x + autoPaintMax.x) * 0.5f * io.DisplayFramebufferScale.x,
                                (autoPaintMin.y + autoPaintMax.y) * 0.5f * io.DisplayFramebufferScale.y);
    io.AddMousePosEvent(autoPixelPoint.x / io.DisplayFramebufferScale.x,
                        autoPixelPoint.y / io.DisplayFramebufferScale.y);
    Check(!frame());
    Check(!autoButtonClicked && !autoButtonActive);
    io.AddMouseButtonEvent(0, true);
    Check(!frame());
    Check(!autoButtonClicked && autoButtonActive);
    io.AddMouseButtonEvent(0, false);
    Check(!frame());
    Check(autoButtonClicked && !autoButtonActive);

    io.AddMousePosEvent(autoButtonMax.x + 8, autoButtonMax.y + 8);
    Check(!frame());
    io.AddMouseButtonEvent(0, true);
    Check(!frame());
    Check(!autoButtonClicked && !autoButtonActive);
    io.AddMouseButtonEvent(0, false);
    Check(!frame());
    Check(!autoButtonClicked && !autoButtonActive);
    std::cout << "3072x1980 display=" << displayScale << " density=" << pixelDensity << " ui=" << uiScale
              << " paint/input aligned\n";
}
} // namespace

int main()
{
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::GetIO().DeltaTime = 1.0f / 60.0f;
    // Transitions reuse one context: 100 -> 125 -> 150 -> 200 -> 100%.
    // Also exercise Retina/Wayland window units, including fractional density.
    for (float density : {1.0f, 1.25f, 1.5f, 2.0f})
        for (float scale : {1.0f, 1.25f, 1.5f, 2.0f, 1.0f})
            CheckFrame(scale, density);
    ImGui::DestroyContext();
}
