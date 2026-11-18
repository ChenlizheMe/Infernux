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
    assert(EditorThemeRegistry::SetActiveTheme("graphite"));
    EditorThemeRegistry::ApplyImGuiColors();
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
    ImGui::DestroyContext();

    return 0;
}
