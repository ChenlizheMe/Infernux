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

    int publications = 0;
    EditorThemeRegistry::SetChangeCallback([&] { ++publications; });
    EditorThemeRegistry::SetDisplayScale(2.0f);
    EditorThemeRegistry::Definition theme;
    theme.name = "test/editor-theme";
    theme.colors = {{"ROLE_ACCENT", ImVec4(0.1f, 0.6f, 0.9f, 1.0f)},
                    {"ROLE_BG_SURFACE", ImVec4(0.9f, 0.9f, 0.85f, 1.0f)},
                    {"ROLE_TEXT_DIM", ImVec4(0.1f, 0.9f, 0.2f, 1.0f)}};
    theme.vec2s = {{"GLOBAL_FRAME_PAD", ImVec2(9.0f, 5.0f)}};
    const auto gameDefault = EditorThemeRegistry::Color("UI_DEFAULT_BUTTON_BG");
    const auto generation = EditorThemeRegistry::Generation();
    EditorThemeRegistry::ReplaceOwner("test:themes", {theme}, theme.name);
    assert(EditorThemeRegistry::Generation() == generation + 1);
    assert(publications == 1);
    assert(ImGui::GetStyle().FramePadding.x == 18.0f);
    assert(ImGui::GetStyle().FramePadding.y == 10.0f);
    assert(EditorThemeRegistry::Color("TEXT_DIM").y == 0.9f);
    assert(EditorThemeRegistry::Color("OBJECT_FIELD_BG").x == 0.9f);
    assert(EditorThemeRegistry::Color("PREFAB_BTN_HOVERED").z > EditorThemeRegistry::Color("PREFAB_BTN_HOVERED").x);
    assert(EditorThemeRegistry::Color("PREFAB_BTN_ACTIVE").z > EditorThemeRegistry::Color("PREFAB_BTN_ACTIVE").x);
    assert(EditorThemeRegistry::Color("UI_DEFAULT_BUTTON_BG").x == gameDefault.x);
    EditorThemeRegistry::ReplaceOwner("test:themes", {theme}, theme.name);
    assert(EditorThemeRegistry::Generation() == generation + 1);
    assert(publications == 1);

    // Explicit component overrides win over deterministic semantic derivation.
    theme.colors["PREFAB_BTN_HOVERED"] = ImVec4(0.3f, 0.2f, 0.1f, 1.0f);
    theme.colors["OBJECT_FIELD_BG"] = ImVec4(0.8f, 0.8f, 0.75f, 1.0f);
    EditorThemeRegistry::Register(theme.name, "test:themes", theme.colors, theme.vec2s, theme.floats);
    assert(publications == 2);
    assert(EditorThemeRegistry::Color("PREFAB_BTN_HOVERED").x == 0.3f);
    assert(EditorThemeRegistry::Color("OBJECT_FIELD_BG").x == 0.8f);
    EditorThemeRegistry::RemoveOwner("test:themes");
    assert(publications == 3);
    assert(EditorThemeRegistry::ActiveTheme() == "infernux");
    assert(ImGui::GetStyle().FramePadding.x == 12.0f);
    EditorThemeRegistry::RemoveOwner("test:themes");
    assert(publications == 3);
    EditorThemeRegistry::SetChangeCallback({});
    EditorThemeRegistry::SetDisplayScale(1.0f);
    ImGui::DestroyContext();

    return 0;
}
