#include "EditorThemeRegistry.h"

#include <stdexcept>

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

inline ImVec4 Alpha(const ImVec4 &c, float a)
{
    return ImVec4(c.x, c.y, c.z, a);
}

// A theme is fully described by a small set of semantic ROLE_* colors; the
// ImGui global palette is composed from them (see ApplyImGuiColors). Switching
// themes therefore only needs to override these roles. Semantic colors
// (LOG_ERROR, PREFAB_TEXT, ...) and the canonical focus SELECTION_BG are
// intentionally not accent-driven.
ThemeData MakeAccentVariant(const ThemeData &base, const ImVec4 &accent)
{
    ThemeData v = base;
    v.colors["ROLE_ACCENT"] = accent;
    v.colors["SELECTION_BG"] = accent;
    v.colors["ROW_SELECTED"] = Alpha(accent, 0.16f);
    v.colors["APPLY_BUTTON"] = accent;
    v.colors["STATUS_PROGRESS_CLR"] = accent;
    v.colors["BTN_SELECTED"] = Alpha(accent, base.colors.at("BTN_SELECTED").w);
    v.colors["UI_EDITOR_ELEMENT_SELECT"] = accent;
    v.colors["BORDER_PLAY"] = accent;
    v.colors["PLAY_ACTIVE"] = Mix(base.colors.at("ROLE_BG_SURFACE"), accent, 0.65f);
    for (const char *key :
         {"HEADER_ACTIVE", "HEADER_HOVERED", "FRAME_BG_ACTIVE", "HIERARCHY_ROW_SELECTED", "INSPECTOR_HEADER_SELECTED",
          "INSPECTOR_HEADER_SELECTED_ACTIVE", "INSPECTOR_HEADER_SELECTED_HOVERED"}) {
        v.colors[key] = Mix(base.colors.at("ROLE_BG_SURFACE"), accent, 0.22f);
    }

    return v;
}

struct Registry
{
    std::unordered_map<std::string, ThemeData> themes;
    std::vector<std::string> order;
    std::string active;
    unsigned long generation = 1;

    Registry()
    {
        ThemeData base = BuildBaseTheme();
        Add("infernux", base);                                                       // Mission Control (default)
        Add("mission_control", base);                                                // alias of the default
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

ImVec4 EditorThemeRegistry::Color(const std::string &name)
{
    const auto &m = Reg().Active().colors;
    const auto it = m.find(name);
    if (it == m.end())
        throw std::out_of_range("Editor theme is missing required color token: " + name);
    return it->second;
}

ImVec2 EditorThemeRegistry::Vec2(const std::string &name)
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
        ++r.generation;
    }
    return true;
}

unsigned long EditorThemeRegistry::Generation()
{
    return Reg().generation;
}

void EditorThemeRegistry::ConfigureImGuiDimensions(ImGuiStyle &style)
{
    style.WindowPadding = Vec2("PANEL_PADDING");
    // Window titles and docking tabs use the outer style. Panel contents
    // apply FIELD_PADDING after Begin independently of title typography.
    style.FramePadding =
        ImVec2(Vec2("FIELD_PADDING").x, (Float("EDITOR_TAB_HEIGHT") - Float("EDITOR_TITLE_FONT_SIZE")) * 0.5f);
    style.CellPadding = ImVec2(4.0f, 4.0f);
    style.ItemSpacing = Vec2("CONTENT_SPACING");
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.IndentSpacing = 18.0f;
    style.ScrollbarSize = 8.0f;
    style.GrabMinSize = 8.0f;

    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.TabBorderSize = 0.0f;
    style.TabBarBorderSize = 1.0f;
    style.TabBarOverlineSize = 2.0f;
    style.TabCloseButtonMinWidthSelected = -1.0f;
    style.TabCloseButtonMinWidthUnselected = 0.0f;
    style.WindowMenuButtonPosition = ImGuiDir_Right;

    style.WindowRounding = 0.0f;
    style.ChildRounding = 0.0f;
    style.FrameRounding = 0.0f;
    style.PopupRounding = 0.0f;
    style.ScrollbarRounding = 0.0f;
    style.GrabRounding = 0.0f;
    style.TabRounding = 0.0f;

    style.AntiAliasedLines = true;
    style.AntiAliasedFill = true;
}

void EditorThemeRegistry::ApplyImGuiColors()
{
    ImGuiStyle &style = ImGui::GetStyle();
    ImVec4 *c = style.Colors;

    const ImVec4 accent = Color("ROLE_ACCENT");
    const ImVec4 bg = Color("ROLE_BG_BASE");
    const ImVec4 surf = Color("ROLE_BG_SURFACE");
    const ImVec4 raised = Color("ROLE_BG_RAISED");
    const ImVec4 hover = Color("ROLE_BG_HOVER");
    const ImVec4 text = Color("ROLE_TEXT");
    const ImVec4 dim = Color("ROLE_TEXT_DIM");
    const ImVec4 border = Color("ROLE_BORDER");
    const ImVec4 field = Color("FRAME_BG");
    const ImVec4 panelFocus = Color("SELECTION_BG");

    const ImVec4 transparent(0.0f, 0.0f, 0.0f, 0.0f);

    // Text
    c[ImGuiCol_Text] = text;
    c[ImGuiCol_TextDisabled] = Color("TEXT_DISABLED");
    c[ImGuiCol_TextSelectedBg] = Alpha(accent, 0.35f);

    // Backgrounds
    c[ImGuiCol_WindowBg] = bg;
    c[ImGuiCol_ChildBg] = surf;
    c[ImGuiCol_PopupBg] = surf;
    c[ImGuiCol_FrameBg] = field;
    c[ImGuiCol_FrameBgHovered] = Color("FRAME_BG_HOVERED");
    c[ImGuiCol_FrameBgActive] = Mix(field, accent, 0.18f);

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
    // Slider grabs are warm-white plates; the accent belongs to the filled track.
    c[ImGuiCol_SliderGrab] = text;
    c[ImGuiCol_SliderGrabActive] = Mix(text, accent, 0.35f);
    c[ImGuiCol_NavHighlight] = transparent;

    // Buttons — raised plate, hover lifts with a faint accent, press commits.
    c[ImGuiCol_Button] = raised;
    c[ImGuiCol_ButtonHovered] = Mix(hover, accent, 0.12f);
    c[ImGuiCol_ButtonActive] = Mix(raised, accent, 0.30f);

    // Headers / selectables — the primary "pick an item" feedback across the
    // editor (Hierarchy tree, Inspector headers, lists), so keep it clearly read.
    c[ImGuiCol_Header] = surf;
    c[ImGuiCol_HeaderHovered] = hover;
    c[ImGuiCol_HeaderActive] = hover;

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
    c[ImGuiCol_Tab] = Color("MENU_BAR_BG");
    c[ImGuiCol_TabHovered] = hover;
    c[ImGuiCol_TabSelected] = bg;
    // A focused editor panel uses the canonical Infernux selection red at
    // full opacity. The dimmed overline below intentionally keeps its current
    // subdued accent treatment for visible but unfocused dock tabs.
    c[ImGuiCol_TabSelectedOverline] = Alpha(panelFocus, 1.0f);
    c[ImGuiCol_TabDimmed] = Color("MENU_BAR_BG");
    c[ImGuiCol_TabDimmedSelected] = bg;
    c[ImGuiCol_TabDimmedSelectedOverline] = Alpha(accent, 0.60f);

    // Docking
    c[ImGuiCol_DockingPreview] = Alpha(accent, 0.25f);
    c[ImGuiCol_DockingEmptyBg] = Mix(bg, ImVec4(0, 0, 0, 1), 0.4f);

    // Plots
    c[ImGuiCol_PlotLines] = Mix(surf, text, 0.45f);
    c[ImGuiCol_PlotHistogram] = accent;

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
