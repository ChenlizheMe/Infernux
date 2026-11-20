#include <function/editor/EditorThemeRegistry.h>

#include <cassert>
#include <stdexcept>
#include <string>

namespace
{

template <typename Callback> void ExpectMissingToken(const std::string &kind, Callback callback)
{
    try {
        callback();
        assert(false && "Missing editor theme token must throw");
    } catch (const std::out_of_range &error) {
        const std::string message = error.what();
        assert(message.find(kind) != std::string::npos);
        assert(message.find("INFERNUX_TEST_MISSING_TOKEN") != std::string::npos);
    }
}

} // namespace

int main()
{
    using infernux::EditorThemeRegistry;

    assert(EditorThemeRegistry::ThemeNames() == std::vector<std::string>{"infernux"});
    const ImVec4 accent = EditorThemeRegistry::Color("ROLE_ACCENT");
    assert(accent.w == 1.0f);

    const ImVec2 spacing = EditorThemeRegistry::Vec2("INSPECTOR_HEADER_ITEM_SPC");
    assert(spacing.x == 4.0f);
    assert(spacing.y == 2.0f);

    assert(EditorThemeRegistry::Float("COMPONENT_ICON_SIZE") == 18.0f);
    assert(EditorThemeRegistry::Float("SCENE_OVERLAY_ROUNDING") == 0.0f);

    const ImVec4 baseLogError = EditorThemeRegistry::Color("LOG_ERROR");

    ExpectMissingToken("color", [] { (void)EditorThemeRegistry::Color("INFERNUX_TEST_MISSING_TOKEN"); });
    ExpectMissingToken("vec2", [] { (void)EditorThemeRegistry::Vec2("INFERNUX_TEST_MISSING_TOKEN"); });
    ExpectMissingToken("float", [] { (void)EditorThemeRegistry::Float("INFERNUX_TEST_MISSING_TOKEN"); });

    ImGui::CreateContext();
    EditorThemeRegistry::ApplyImGuiColors();
    assert(ImGui::GetColorU32(ImGuiCol_WindowBg) == IM_COL32(22, 23, 24, 255));
    assert(ImGui::GetColorU32(ImGuiCol_ChildBg) == IM_COL32(28, 29, 31, 255));
    assert(ImGui::GetColorU32(ImGuiCol_FrameBg) == IM_COL32(17, 18, 19, 255));
    assert(ImGui::GetColorU32(ImGuiCol_TitleBg) == IM_COL32(14, 15, 16, 255));
    assert(ImGui::GetColorU32(ImGuiCol_DockingEmptyBg) == ImGui::GetColorU32(ImGuiCol_TitleBg));
    assert(ImGui::GetColorU32(ImGuiCol_MenuBarBg) == IM_COL32(10, 11, 12, 255));
    const ImU32 neutralHover = IM_COL32(42, 43, 46, 255);
    assert(ImGui::GetColorU32(ImGuiCol_ButtonHovered) == neutralHover);
    assert(ImGui::GetColorU32(ImGuiCol_HeaderHovered) == neutralHover);
    assert(ImGui::GetColorU32(ImGuiCol_TabHovered) == neutralHover);
    assert(ImGui::GetColorU32(ImGuiCol_NavCursor) == ImGui::ColorConvertFloat4ToU32(accent));
    assert(ImGui::ColorConvertFloat4ToU32(EditorThemeRegistry::Color("INSPECTOR_HEADER_PRIMARY_ACTIVE")) ==
           IM_COL32(63, 46, 50, 255));

    const ImVec4 customSurface(0.82f, 0.78f, 0.64f, 1.0f);
    const ImVec4 customField(0.72f, 0.70f, 0.58f, 1.0f);
    EditorThemeRegistry::Register("tests/palette", "theme:tests/palette",
                                  {{"ROLE_ACCENT", ImVec4(0.30f, 0.56f, 0.86f, 1.0f)},
                                   {"ROLE_BG_SURFACE", customSurface},
                                   {"FRAME_BG", customField}},
                                  {}, {});
    assert(EditorThemeRegistry::SetActiveTheme("tests/palette"));
    assert(ImGui::GetColorU32(ImGuiCol_ButtonHovered) == neutralHover);
    assert(ImGui::GetColorU32(ImGuiCol_ChildBg) == ImGui::ColorConvertFloat4ToU32(customSurface));
    assert(ImGui::GetColorU32(ImGuiCol_FrameBg) == ImGui::ColorConvertFloat4ToU32(customField));
    const ImVec4 standardSelection = EditorThemeRegistry::Color("SELECTION_BG");
    const ImVec4 variantAccent = EditorThemeRegistry::Color("ROLE_ACCENT");
    const ImVec4 text = EditorThemeRegistry::Color("TEXT");
    const ImVec4 histogram = ImGui::GetStyle().Colors[ImGuiCol_PlotHistogram];
    const ImVec4 hoveredHistogram = ImGui::GetStyle().Colors[ImGuiCol_PlotHistogramHovered];
    assert(histogram.x == text.x && histogram.y == text.y && histogram.z == text.z);
    assert(hoveredHistogram.x == variantAccent.x && hoveredHistogram.y == variantAccent.y &&
           hoveredHistogram.z == variantAccent.z);
    const ImVec4 progressFill = EditorThemeRegistry::Color("STATUS_PROGRESS_CLR");
    assert(progressFill.x == variantAccent.x && progressFill.y == variantAccent.y && progressFill.z == variantAccent.z);
    const ImVec4 focusedPanel = ImGui::GetStyle().Colors[ImGuiCol_TabSelectedOverline];
    const ImVec4 unfocusedPanel = ImGui::GetStyle().Colors[ImGuiCol_TabDimmedSelectedOverline];
    assert(focusedPanel.x == standardSelection.x);
    assert(focusedPanel.y == standardSelection.y);
    assert(focusedPanel.z == standardSelection.z);
    assert(focusedPanel.w == 1.0f);
    assert(unfocusedPanel.x == variantAccent.x);
    assert(unfocusedPanel.y == variantAccent.y);
    assert(unfocusedPanel.z == variantAccent.z);
    assert(unfocusedPanel.w == 0.60f);
    assert(EditorThemeRegistry::Color("LOG_ERROR").x == baseLogError.x);
    const ImVec4 prefabText = EditorThemeRegistry::Color("PREFAB_TEXT");
    assert(prefabText.x == variantAccent.x && prefabText.y == variantAccent.y && prefabText.z == variantAccent.z);
    assert(variantAccent.x != baseLogError.x);

    EditorThemeRegistry::RemoveOwner("theme:tests/palette");
    assert(EditorThemeRegistry::ActiveTheme() == "infernux");
    assert(EditorThemeRegistry::ThemeNames() == std::vector<std::string>{"infernux"});
    assert(ImGui::GetColorU32(ImGuiCol_ChildBg) == IM_COL32(28, 29, 31, 255));

    ImGui::DestroyContext();

    return 0;
}
