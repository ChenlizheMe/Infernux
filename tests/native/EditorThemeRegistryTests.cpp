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

    assert(EditorThemeRegistry::Float("COMPONENT_ICON_SIZE") == 16.0f);

    ExpectMissingToken("color", [] { (void)EditorThemeRegistry::Color("INFERNUX_TEST_MISSING_TOKEN"); });
    ExpectMissingToken("vec2", [] { (void)EditorThemeRegistry::Vec2("INFERNUX_TEST_MISSING_TOKEN"); });
    ExpectMissingToken("float", [] { (void)EditorThemeRegistry::Float("INFERNUX_TEST_MISSING_TOKEN"); });

    ImGui::CreateContext();
    assert(EditorThemeRegistry::SetActiveTheme("graphite"));
    EditorThemeRegistry::ApplyImGuiColors();
    const ImVec4 standardSelection = EditorThemeRegistry::Color("SELECTION_BG");
    const ImVec4 variantAccent = EditorThemeRegistry::Color("ROLE_ACCENT");
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
    ImGui::DestroyContext();

    return 0;
}
