#include "EditorThemeRegistry.h"

#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace infernux
{
namespace
{

struct ThemeData
{
    std::unordered_map<std::string, ImVec4> colors;
    std::unordered_map<std::string, ImVec2> vec2s;
    std::unordered_map<std::string, float> floats;
};

ThemeData BuildBaseTheme()
{
    ThemeData t;
#define INX_THEME_COLOR(name, r, g, b, a) t.colors.emplace(#name, ImVec4(r, g, b, a));
#define INX_THEME_VEC2(name, x, y) t.vec2s.emplace(#name, ImVec2(x, y));
#define INX_THEME_FLOAT(name, v) t.floats.emplace(#name, v);
#include "EditorThemeTable.inl"
#undef INX_THEME_COLOR
#undef INX_THEME_VEC2
#undef INX_THEME_FLOAT
    return t;
}

inline ImVec4 Mix(const ImVec4 &a, const ImVec4 &b, float t)
{
    return ImVec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t);
}

inline ImVec4 Alpha(const ImVec4 &c, float a) { return ImVec4(c.x, c.y, c.z, a); }

// Explicit role derivation precedes component overrides. Functional colors
// (Play/Pause, node types and game UI defaults) retain their own semantics.
void DeriveRoles(ThemeData &theme, const std::unordered_map<std::string, ImVec4> &overrides)
{
    const auto derive = [&](const char *role, std::initializer_list<const char *> tokens) {
        if (!overrides.count(role))
            return;
        for (const auto *token : tokens)
            theme.colors.at(token) = Alpha(theme.colors.at(role), theme.colors.at(token).w);
    };
    derive("ROLE_BG_BASE", {"WINDOW_BG", "MENU_BAR_BG", "STATUS_BAR_BG", "STATUS_PROGRESS_BG", "LOG_DIM"});
    derive("ROLE_BG_SURFACE", {"FRAME_BG", "BTN_IDLE", "BTN_NORMAL", "BTN_DISABLED", "INSPECTOR_INLINE_BTN_IDLE",
                               "INSPECTOR_LIST_BODY_BG", "CURVE_EDITOR_BG", "UI_EDITOR_CANVAS_BG",
                               "SCENE_OVERLAY_COMBO_BG", "STATUS_ACTIVITY_BG", "CONSOLE_SEGMENT_ACTIVE"});
    derive("ROLE_BG_RAISED", {"POPUP_BG", "HEADER", "INSPECTOR_HEADER_PRIMARY", "INSPECTOR_HEADER_SECONDARY",
                              "INSPECTOR_HEADER_TERTIARY", "INSPECTOR_HEADER_LIST", "PREFAB_HEADER_BG",
                              "PROJECT_EXPAND_STRIP_BG", "PROJECT_SUBASSET_CELL_BG", "UI_EDITOR_CANVAS_HEADER_BG"});
    derive("ROLE_BG_HOVER", {"HIERARCHY_ROW_HOVER", "BTN_SUBTLE_HOVER", "FRAME_BG_HOVERED"});
    derive("ROLE_TEXT", {"TEXT", "LOG_INFO", "META_TEXT", "UI_EDITOR_CANVAS_HEADER_TEXT", "CURVE_EDITOR_KEY",
                         "SCENE_TOOL_ICON", "SCENE_TOOL_ICON_SELECTED"});
    derive("ROLE_TEXT_DIM", {"TEXT_DISABLED", "TEXT_DIM", "TEXT_DIM2", "LOG_TRACE", "LOG_BADGE",
                             "STATUS_PROGRESS_LABEL_CLR", "UI_EDITOR_LABEL_COLOR", "UI_EDITOR_FALLBACK_TEXT"});
    derive("ROLE_BORDER", {"BORDER", "STATUS_BAR_BORDER", "INSPECTOR_LIST_BODY_BORDER", "UI_EDITOR_CANVAS_BORDER",
                           "CURVE_EDITOR_GRID", "CURVE_EDITOR_AXIS", "COLOR_SWATCH_BORDER"});
    derive("ROLE_ERROR", {"ERROR_TEXT", "LOG_ERROR"});
    derive("ROLE_WARNING", {"WARNING_TEXT", "LOG_WARNING"});
    derive("ROLE_SUCCESS", {"SUCCESS_TEXT"});
    derive("ROLE_ACCENT", {"APPLY_BUTTON", "PREFAB_TEXT", "PREFAB_TEXT2", "STATUS_PROGRESS_CLR", "SELECTION_BG",
                           "BTN_SELECTED", "HIERARCHY_ROW_SELECTED", "INSPECTOR_INLINE_BTN_ACTIVE", "INSPECTOR_INLINE_BTN_ON",
                           "PREFAB_BTN_NORMAL", "SCENE_TOOL_SELECTED_BG", "TREE_DND_LINE_CLR", "UI_EDITOR_ALIGN_GUIDE", "UI_EDITOR_ALIGN_GUIDE_FAINT",
                           "UI_EDITOR_ELEMENT_HOVER", "UI_EDITOR_ELEMENT_SELECT", "CURVE_EDITOR_LINE", "CURVE_EDITOR_KEY_SELECTED"});
    const bool accent = overrides.count("ROLE_ACCENT") != 0;
    if (accent) {
        theme.colors["PROJECT_SUBASSET_ACCENT"] = Alpha(Mix(ImVec4(0, 0, 0, 1), theme.colors.at("ROLE_ACCENT"), 0.7f),
                                                       theme.colors.at("PROJECT_SUBASSET_ACCENT").w);
    }
    if (accent || overrides.count("ROLE_TEXT") || overrides.count("ROLE_BG_BASE")) {
        theme.colors["PREFAB_BTN_HOVERED"] = Mix(theme.colors.at("ROLE_ACCENT"), theme.colors.at("ROLE_TEXT"), 0.18f);
        theme.colors["PREFAB_BTN_ACTIVE"] = Mix(theme.colors.at("ROLE_ACCENT"), theme.colors.at("ROLE_BG_BASE"), 0.12f);
        theme.colors["SCENE_TOOL_SELECTED_HOVER"] = theme.colors.at("PREFAB_BTN_HOVERED");
        theme.colors["SCENE_TOOL_SELECTED_ACTIVE"] = theme.colors.at("PREFAB_BTN_ACTIVE");
    }
    if (accent || overrides.count("ROLE_BG_SURFACE") || overrides.count("ROLE_BG_HOVER") || overrides.count("ROLE_BG_RAISED")) {
        const auto tint = [&](std::initializer_list<const char *> tokens, const char *surface, float strength) {
            for (const auto *token : tokens)
                theme.colors.at(token) = Alpha(Mix(theme.colors.at(surface), theme.colors.at("ROLE_ACCENT"), strength),
                                                theme.colors.at(token).w);
        };
        tint({"BTN_HOVERED", "BTN_SUBTLE_HOVER", "INSPECTOR_INLINE_BTN_HOVER", "FRAME_BG_HOVERED",
              "HEADER_HOVERED", "INSPECTOR_HEADER_PRIMARY_HOVERED", "INSPECTOR_HEADER_SECONDARY_HOVERED",
              "INSPECTOR_HEADER_TERTIARY_HOVERED", "INSPECTOR_HEADER_LIST_HOVERED", "SPLITTER_HOVER",
              "PROJECT_EXPAND_STRIP_HOVER", "SCENE_OVERLAY_COMBO_HOVER"}, "ROLE_BG_HOVER", 0.18f);
        tint({"BTN_ACTIVE", "FRAME_BG_ACTIVE", "HEADER_ACTIVE", "INSPECTOR_HEADER_PRIMARY_ACTIVE",
              "INSPECTOR_HEADER_SECONDARY_ACTIVE", "INSPECTOR_HEADER_TERTIARY_ACTIVE", "INSPECTOR_HEADER_LIST_ACTIVE",
              "SPLITTER_ACTIVE", "SCENE_OVERLAY_COMBO_ACTIVE", "UI_EDITOR_CANVAS_HEADER_BG_FOC",
              "CURVE_EDITOR_TANGENT"}, "ROLE_BG_SURFACE", 0.32f);
        tint({"INSPECTOR_HEADER_SELECTED"}, "ROLE_BG_RAISED", 0.35f);
        tint({"INSPECTOR_HEADER_SELECTED_HOVERED"}, "ROLE_BG_RAISED", 0.45f);
        tint({"INSPECTOR_HEADER_SELECTED_ACTIVE"}, "ROLE_BG_RAISED", 0.55f);
    }
}

ThemeData MakeAccentVariant(const ThemeData &base, const ImVec4 &accent)
{
    ThemeData theme = base;
    theme.colors["ROLE_ACCENT"] = accent;
    DeriveRoles(theme, {{"ROLE_ACCENT", accent}});
    return theme;
}

ThemeData CanonicalBaseTheme()
{
    auto theme = BuildBaseTheme();
    DeriveRoles(theme, theme.colors);
    return theme;
}

struct Registry
{
    std::unordered_map<std::string, ThemeData> themes;
    std::unordered_map<std::string, std::string> owners;
    std::vector<std::string> order;
    std::string active;
    unsigned long generation = 1;
    float displayScale = 1.0f;
    std::function<void()> changed;

    Registry()
    {
        ThemeData base = CanonicalBaseTheme();
        Add("infernux", base);                                                       // theme red (default)
        Add("graphite", MakeAccentVariant(base, ImVec4(0.30f, 0.56f, 0.86f, 1.0f))); // steel blue
        Add("amber", MakeAccentVariant(base, ImVec4(0.92f, 0.62f, 0.22f, 1.0f)));    // instrument amber
        active = "infernux";
    }

    void Add(const std::string &name, const ThemeData &data)
    {
        if (!themes.count(name))
            order.push_back(name);
        themes[name] = data;
    }

    ThemeData &Active()
    {
        return themes[active];
    }
};

Registry &Reg()
{
    // Engine-lifetime singleton (intentionally leaked, like SceneManager).
    static Registry *r = new Registry();
    return *r;
}

bool Same(const ImVec4 &a, const ImVec4 &b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
bool Same(const ImVec2 &a, const ImVec2 &b) { return a.x == b.x && a.y == b.y; }
bool Same(float a, float b) { return a == b; }

template <typename T> bool SameMap(const std::unordered_map<std::string, T> &a, const std::unordered_map<std::string, T> &b)
{
    if (a.size() != b.size())
        return false;
    for (const auto &[key, value] : a) {
        const auto found = b.find(key);
        if (found == b.end() || !Same(value, found->second))
            return false;
    }
    return true;
}

bool SameTheme(const ThemeData &a, const ThemeData &b)
{
    return SameMap(a.colors, b.colors) && SameMap(a.vec2s, b.vec2s) && SameMap(a.floats, b.floats);
}

void PublishActiveChange()
{
    auto &registry = Reg();
    ++registry.generation;
    if (ImGui::GetCurrentContext()) {
        EditorThemeRegistry::ApplyImGuiColors();
        EditorThemeRegistry::ConfigureImGuiDimensions(ImGui::GetStyle(), registry.displayScale);
    }
    if (registry.changed)
        registry.changed();
}

} // namespace

const std::unordered_map<std::string, ImVec4> &EditorThemeRegistry::Colors()
{
    return Reg().Active().colors;
}
const std::unordered_map<std::string, ImVec2> &EditorThemeRegistry::Vec2s()
{
    return Reg().Active().vec2s;
}
const std::unordered_map<std::string, float> &EditorThemeRegistry::Floats()
{
    return Reg().Active().floats;
}

const ImVec4 &EditorThemeRegistry::Color(const std::string &name)
{
    const auto &m = Reg().Active().colors;
    const auto it = m.find(name);
    if (it == m.end())
        throw std::out_of_range("Editor theme is missing required color token: " + name);
    return it->second;
}

const ImVec2 &EditorThemeRegistry::Vec2(const std::string &name)
{
    const auto &m = Reg().Active().vec2s;
    const auto it = m.find(name);
    if (it == m.end())
        throw std::out_of_range("Editor theme is missing required vec2 token: " + name);
    return it->second;
}

float EditorThemeRegistry::Float(const std::string &name)
{
    const auto &m = Reg().Active().floats;
    const auto it = m.find(name);
    if (it == m.end())
        throw std::out_of_range("Editor theme is missing required float token: " + name);
    return it->second;
}

std::vector<std::string> EditorThemeRegistry::ThemeNames()
{
    return Reg().order;
}
const std::string &EditorThemeRegistry::ActiveTheme()
{
    return Reg().active;
}

bool EditorThemeRegistry::SetActiveTheme(const std::string &name)
{
    auto &r = Reg();
    if (!r.themes.count(name))
        return false;
    if (r.active != name) {
        r.active = name;
        PublishActiveChange();
    }
    return true;
}

unsigned long EditorThemeRegistry::Generation()
{
    return Reg().generation;
}

static ThemeData BuildDeclaredTheme(const std::unordered_map<std::string, ImVec4> &colors,
                                    const std::unordered_map<std::string, ImVec2> &vec2s,
                                    const std::unordered_map<std::string, float> &floats)
{
    auto base = CanonicalBaseTheme();
    for (const auto &[key, value] : colors) {
        if ((key.rfind("UI_DEFAULT_", 0) == 0) || !base.colors.count(key))
            throw std::invalid_argument("Unknown editor color token: " + key);
        for (const float channel : {value.x, value.y, value.z, value.w})
            if (!std::isfinite(channel) || channel < 0.0f || channel > 1.0f)
                throw std::invalid_argument("Editor color must contain finite RGBA values in [0, 1]: " + key);
        base.colors[key] = value;
    }
    DeriveRoles(base, colors);
    for (const auto &[key, value] : colors)
        base.colors[key] = value;
    for (const auto &[key, value] : vec2s) {
        if (key.rfind("UI_DEFAULT_", 0) == 0 || !base.vec2s.count(key))
            throw std::invalid_argument("Unknown editor dimension token: " + key);
        if (!std::isfinite(value.x) || !std::isfinite(value.y))
            throw std::invalid_argument("Editor dimensions must be finite: " + key);
        if (key.rfind("GLOBAL_", 0) == 0 && (value.x < 0.0f || value.y < 0.0f))
            throw std::invalid_argument("Global editor spacing must be non-negative: " + key);
        base.vec2s[key] = value;
    }
    for (const auto &[key, value] : floats) {
        if (key.rfind("UI_DEFAULT_", 0) == 0 || !base.floats.count(key))
            throw std::invalid_argument("Unknown editor dimension token: " + key);
        if (!std::isfinite(value))
            throw std::invalid_argument("Editor dimensions must be finite: " + key);
        if (key.rfind("GLOBAL_", 0) == 0 && value < 0.0f)
            throw std::invalid_argument("Global editor dimensions must be non-negative: " + key);
        if ((key.rfind("PANEL_", 0) == 0 || key.rfind("BUTTON_", 0) == 0) && value < 0.0f)
            throw std::invalid_argument("Theme chrome dimensions must be non-negative: " + key);
        if ((key == "SURFACE_GRAIN_OPACITY" || key == "SURFACE_HEX_OPACITY") && (value < 0.0f || value > 1.0f))
            throw std::invalid_argument("Surface opacity must be in [0, 1]: " + key);
        if ((key == "SURFACE_GRAIN_SCALE" || key == "SURFACE_HEX_SCALE") && (value < 0.25f || value > 8.0f))
            throw std::invalid_argument("Surface scale must be in [0.25, 8]: " + key);
        base.floats[key] = value;
    }
    return base;
}

static void ValidateIdentity(const std::string &name, const std::string &owner)
{
    auto &registry = Reg();
    const auto existing = registry.owners.find(name);
    if (name.empty() || owner.empty() || (registry.themes.count(name) &&
        (existing == registry.owners.end() || existing->second != owner)))
        throw std::invalid_argument("Theme identity is already owned or empty: " + name);
}

void EditorThemeRegistry::Register(const std::string &name, const std::string &owner,
                                  const std::unordered_map<std::string, ImVec4> &colors,
                                  const std::unordered_map<std::string, ImVec2> &vec2s,
                                  const std::unordered_map<std::string, float> &floats)
{
    ValidateIdentity(name, owner);
    const auto declared = BuildDeclaredTheme(colors, vec2s, floats);
    auto &registry = Reg();
    const bool changed = registry.active == name && !SameTheme(registry.Active(), declared);
    registry.Add(name, declared);
    registry.owners[name] = owner;
    if (changed)
        PublishActiveChange();
}

void EditorThemeRegistry::ReplaceOwner(const std::string &owner, const std::vector<Definition> &themes,
                                       const std::string &preferred)
{
    if (owner.empty())
        throw std::invalid_argument("Theme owner must not be empty");
    std::unordered_map<std::string, ThemeData> replacements;
    for (const auto &theme : themes) {
        ValidateIdentity(theme.name, owner);
        if (!replacements.emplace(theme.name, BuildDeclaredTheme(theme.colors, theme.vec2s, theme.floats)).second)
            throw std::invalid_argument("Duplicate theme identity: " + theme.name);
    }
    auto &registry = Reg();
    const auto oldName = registry.active;
    const auto oldData = registry.Active();
    for (auto iterator = registry.owners.begin(); iterator != registry.owners.end();) {
        if (iterator->second != owner || replacements.count(iterator->first)) {
            ++iterator;
            continue;
        }
        const auto name = iterator->first;
        registry.themes.erase(name);
        registry.order.erase(std::remove(registry.order.begin(), registry.order.end(), name), registry.order.end());
        iterator = registry.owners.erase(iterator);
    }
    for (const auto &theme : themes) {
        registry.Add(theme.name, replacements.at(theme.name));
        registry.owners[theme.name] = owner;
    }
    if (!preferred.empty())
        registry.active = registry.themes.count(preferred) ? preferred : "infernux";
    else if (!registry.themes.count(registry.active))
        registry.active = "infernux";
    if (oldName != registry.active || !SameTheme(oldData, registry.Active()))
        PublishActiveChange();
}

void EditorThemeRegistry::RemoveOwner(const std::string &owner) { ReplaceOwner(owner, {}, ""); }

void EditorThemeRegistry::SetChangeCallback(std::function<void()> callback) { Reg().changed = std::move(callback); }

void EditorThemeRegistry::SetDisplayScale(float scale)
{
    if (!std::isfinite(scale) || scale <= 0.0f)
        throw std::invalid_argument("Editor display scale must be positive and finite");
    Reg().displayScale = scale;
}

void EditorThemeRegistry::ConfigureImGuiDimensions(ImGuiStyle &style, float scale)
{
    const auto vec = [scale](const char *name) { const auto v = Vec2(name); return ImVec2(v.x * scale, v.y * scale); };
    const auto scalar = [scale](const char *name) { return Float(name) * scale; };
    style.WindowPadding = vec("GLOBAL_WINDOW_PAD");
    style.FramePadding = vec("GLOBAL_FRAME_PAD");
    style.CellPadding = vec("GLOBAL_CELL_PAD");
    style.ItemSpacing = vec("GLOBAL_ITEM_SPC");
    style.ItemInnerSpacing = vec("GLOBAL_INNER_SPC");
    style.IndentSpacing = scalar("GLOBAL_INDENT");
    style.ScrollbarSize = scalar("GLOBAL_SCROLLBAR_SIZE");
    style.GrabMinSize = scalar("GLOBAL_GRAB_MIN_SIZE");

    style.WindowBorderSize = scalar("GLOBAL_WINDOW_BORDER");
    style.ChildBorderSize = scalar("GLOBAL_CHILD_BORDER");
    style.PopupBorderSize = scalar("GLOBAL_POPUP_BORDER");
    style.FrameBorderSize = scalar("GLOBAL_FRAME_BORDER");
    style.TabBorderSize = scalar("GLOBAL_TAB_BORDER");
    style.TabBarBorderSize = scalar("GLOBAL_TAB_BAR_BORDER");

    style.WindowRounding = scalar("GLOBAL_WINDOW_ROUNDING");
    style.ChildRounding = scalar("GLOBAL_CHILD_ROUNDING");
    style.FrameRounding = scalar("GLOBAL_FRAME_ROUNDING");
    style.PopupRounding = scalar("GLOBAL_POPUP_ROUNDING");
    style.ScrollbarRounding = scalar("GLOBAL_SCROLLBAR_ROUNDING");
    style.GrabRounding = scalar("GLOBAL_GRAB_ROUNDING");
    style.TabRounding = scalar("GLOBAL_TAB_ROUNDING");

    style.AntiAliasedLines = true;
    style.AntiAliasedFill = true;
}

void EditorThemeRegistry::ApplyImGuiColors()
{
    ImGuiStyle &style = ImGui::GetStyle();
    ImVec4 *c = style.Colors;

    const ImVec4 accent = Color("ROLE_ACCENT");
    const ImVec4 bg = Color("WINDOW_BG");
    const ImVec4 surf = Color("FRAME_BG");
    const ImVec4 hover = Color("ROLE_BG_HOVER");
    const ImVec4 text = Color("TEXT");
    const ImVec4 dim = Color("TEXT_DISABLED");
    const ImVec4 border = Color("BORDER");
    const ImVec4 panelFocus = Color("SELECTION_BG");

    const ImVec4 transparent(0.0f, 0.0f, 0.0f, 0.0f);

    // Text
    c[ImGuiCol_Text] = text;
    c[ImGuiCol_TextDisabled] = dim;
    c[ImGuiCol_TextSelectedBg] = Alpha(accent, 0.35f);
    c[ImGuiCol_InputTextCursor] = text;
    c[ImGuiCol_TextLink] = accent;
    c[ImGuiCol_UnsavedMarker] = accent;
    c[ImGuiCol_TreeLines] = border;

    // Backgrounds
    c[ImGuiCol_WindowBg] = bg;
    c[ImGuiCol_ChildBg] = surf;
    c[ImGuiCol_PopupBg] = Color("POPUP_BG");
    c[ImGuiCol_FrameBg] = surf;
    c[ImGuiCol_FrameBgHovered] = Color("FRAME_BG_HOVERED");
    c[ImGuiCol_FrameBgActive] = Color("FRAME_BG_ACTIVE");

    // Title / menu
    c[ImGuiCol_TitleBg] = bg;
    c[ImGuiCol_TitleBgActive] = bg;
    c[ImGuiCol_TitleBgCollapsed] = Alpha(bg, 0.75f);
    c[ImGuiCol_MenuBarBg] = Color("MENU_BAR_BG");

    // Scrollbar
    c[ImGuiCol_ScrollbarBg] = transparent;
    c[ImGuiCol_ScrollbarGrab] = Mix(surf, text, 0.14f);
    c[ImGuiCol_ScrollbarGrabHovered] = Mix(surf, text, 0.26f);
    c[ImGuiCol_ScrollbarGrabActive] = Alpha(accent, 0.70f);

    // Accent widgets
    c[ImGuiCol_CheckMark] = accent;
    c[ImGuiCol_SliderGrab] = Alpha(accent, 0.88f);
    c[ImGuiCol_SliderGrabActive] = accent;
    c[ImGuiCol_NavHighlight] = transparent;
    c[ImGuiCol_NavWindowingHighlight] = Alpha(accent, 0.70f);
    c[ImGuiCol_NavWindowingDimBg] = Alpha(bg, 0.20f);

    // Buttons — surface with an accent-tinted hover/active for clear feedback.
    c[ImGuiCol_Button] = Color("BTN_NORMAL");
    c[ImGuiCol_ButtonHovered] = Color("BTN_HOVERED");
    c[ImGuiCol_ButtonActive] = Color("BTN_ACTIVE");

    // Headers / selectables — the primary "pick an item" feedback across the
    // editor (Hierarchy tree, Inspector headers, lists), so keep it clearly read.
    c[ImGuiCol_Header] = Color("HEADER");
    c[ImGuiCol_HeaderHovered] = Color("HEADER_HOVERED");
    c[ImGuiCol_HeaderActive] = Color("HEADER_ACTIVE");

    // Borders / separators
    c[ImGuiCol_Border] = border;
    c[ImGuiCol_BorderShadow] = transparent;
    c[ImGuiCol_Separator] = border;
    c[ImGuiCol_SeparatorHovered] = Alpha(accent, 0.60f);
    c[ImGuiCol_SeparatorActive] = Alpha(accent, 0.80f);

    // Resize grip
    c[ImGuiCol_ResizeGrip] = transparent;
    c[ImGuiCol_ResizeGripHovered] = Alpha(accent, 0.30f);
    c[ImGuiCol_ResizeGripActive] = Alpha(accent, 0.50f);

    // Tabs
    c[ImGuiCol_Tab] = bg;
    c[ImGuiCol_TabHovered] = hover;
    c[ImGuiCol_TabSelected] = surf;
    // A focused editor panel uses the canonical Infernux selection red at
    // full opacity. The dimmed overline below intentionally keeps its current
    // subdued accent treatment for visible but unfocused dock tabs.
    c[ImGuiCol_TabSelectedOverline] = Alpha(panelFocus, 1.0f);
    c[ImGuiCol_TabDimmed] = bg;
    c[ImGuiCol_TabDimmedSelected] = surf;
    c[ImGuiCol_TabDimmedSelectedOverline] = Alpha(accent, 0.60f);

    // Docking
    c[ImGuiCol_DockingPreview] = Alpha(accent, 0.25f);
    c[ImGuiCol_DockingEmptyBg] = Mix(bg, ImVec4(0, 0, 0, 1), 0.4f);

    // Plots
    c[ImGuiCol_PlotLines] = Mix(surf, text, 0.45f);
    c[ImGuiCol_PlotLinesHovered] = accent;
    c[ImGuiCol_PlotHistogram] = text;
    c[ImGuiCol_PlotHistogramHovered] = accent;

    // Drag-drop
    c[ImGuiCol_DragDropTarget] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    c[ImGuiCol_DragDropTargetBg] = transparent;

    // Modal
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.56f);

    // Tables
    c[ImGuiCol_TableHeaderBg] = surf;
    c[ImGuiCol_TableBorderStrong] = border;
    c[ImGuiCol_TableBorderLight] = Mix(bg, border, 0.5f);
    c[ImGuiCol_TableRowBg] = transparent;
    c[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 0.02f);
}

} // namespace infernux
