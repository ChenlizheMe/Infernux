#include "MenuBarPanel.h"
#include <function/renderer/gui/EditorMenuItem.h>
#include <function/renderer/gui/InxGUISemantics.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <tuple>

namespace infernux
{
namespace
{

bool BeginSemanticMenu(InxGUIContext *ctx, const std::string &label, const std::string &semanticId, bool enabled = true)
{
    const bool open = ImGui::BeginMenu(label.c_str(), enabled);
    if (ctx && InxGUISemantics::IsCaptureEnabled())
        ctx->RecordSemanticItem("menu", label, enabled, semanticId, open);
    return open;
}

bool SemanticMenuItem(InxGUIContext *ctx, const std::string &label, const std::string &shortcut, bool selected,
                      bool enabled, const std::string &semanticId)
{
    const bool clicked =
        EditorMenuItem(label.c_str(), shortcut.empty() ? nullptr : shortcut.c_str(), selected, enabled);
    if (ctx && InxGUISemantics::IsCaptureEnabled())
        ctx->RecordSemanticItem("menu_item", label, enabled, semanticId);
    return clicked;
}

} // namespace

// ════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════

MenuBarPanel::MenuBarPanel() = default;

void MenuBarPanel::InvalidateWindowTypeCache()
{
    m_windowTypesDirty = true;
}

// ════════════════════════════════════════════════════════════════════
// Translation helper
// ════════════════════════════════════════════════════════════════════

std::string MenuBarPanel::T(const std::string &key) const
{
    if (translate)
        return translate(key);
    auto dot = key.rfind('.');
    return (dot != std::string::npos) ? key.substr(dot + 1) : key;
}

std::vector<std::string> SplitMenuPath(const std::string &path)
{
    std::vector<std::string> result;
    size_t start = 0;
    while (start <= path.size()) {
        const size_t slash = path.find('/', start);
        result.push_back(path.substr(start, slash == std::string::npos ? std::string::npos : slash - start));
        if (slash == std::string::npos)
            break;
        start = slash + 1;
    }
    return result;
}

std::string MenuBarPanel::ResolveMenuLabel(const std::string &key, const std::string &literal) const
{
    if (hasTranslation && hasTranslation(key))
        return T(key);
    return literal;
}

// ════════════════════════════════════════════════════════════════════
// Render
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::OnRender(InxGUIContext *ctx)
{
    // Check for window close request (SDL_EVENT_QUIT intercepted by C++)
    if (isCloseRequested && onRequestClose) {
        if (isCloseRequested())
            onRequestClose();
    }

    // These authored overrides replace the already-scaled global style.
    const float dpi = ctx->GetDpiScale();
    const auto scaled = [dpi](ImVec2 size) { return ImVec2(size.x * dpi, size.y * dpi); };
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, scaled(EditorTheme::TOOLBAR_FRAME_PAD));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, scaled(EditorTheme::TOOLBAR_ITEM_SPC));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, scaled(EditorTheme::TOOLBAR_WIN_PAD));
    ImGui::PushStyleColor(ImGuiCol_MenuBarBg, EditorTheme::MENU_BAR_BG);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, EditorTheme::POPUP_BG);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, EditorTheme::HEADER_HOVERED);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, EditorTheme::HEADER_ACTIVE);

    if (ImGui::BeginMainMenuBar()) {
        if (InxGUISemantics::IsCaptureEnabled())
            ctx->RecordSemanticWindow("menu_bar", "Main Menu", "menu_bar");
        RenderProjectMenu(ctx);
        RenderEditMenu(ctx);
        RenderSceneMenu(ctx);
        RenderDynamicMenus(ctx);
        RenderWindowMenu(ctx);
        ImGui::EndMainMenuBar();
    }

    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(3);

    // Utility settings are WindowManager-owned panel surfaces; only global
    // confirmation overlays remain separately rendered from Python.
}

// ════════════════════════════════════════════════════════════════════
// Project menu
// ════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::RenderProjectMenu(InxGUIContext *ctx)
{
    if (!BeginSemanticMenu(ctx, T("menu.project"), "menu.project"))
        return;

    const bool canNew = CanExecuteCommand("file.new_scene");
    if (SemanticMenuItem(ctx, T("menu.new_scene"), "Ctrl+N", false, canNew, "menu.project.new_scene"))
        ExecuteCommand("file.new_scene", "menu");
    const bool canSave = CanExecuteCommand("file.save");
    if (SemanticMenuItem(ctx, T("menu.save_scene"), "Ctrl+S", false, canSave, "menu.project.save_scene"))
        ExecuteCommand("file.save", "menu");
    const bool canSaveAs = CanExecuteCommand("file.save_as");
    if (SemanticMenuItem(ctx, T("menu.save_scene_as"), "Ctrl+Shift+S", false, canSaveAs, "menu.project.save_scene_as"))
        ExecuteCommand("file.save_as", "menu");

    ImGui::Separator();

    // Build Settings toggle
    const bool canBuildSettings = CanExecuteCommand("window.toggle.build_settings");
    const bool buildSettingsOpen = IsCommandChecked("window.toggle.build_settings");
    if (SemanticMenuItem(ctx, T("menu.build_settings"), "", buildSettingsOpen, canBuildSettings,
                         "menu.project.build_settings"))
        ExecuteCommand("window.toggle.build_settings", "menu");

    // Physics Layer Matrix toggle
    const bool canPhysicsLayers = CanExecuteCommand("window.toggle.physics_layers");
    const bool physicsLayersOpen = IsCommandChecked("window.toggle.physics_layers");
    if (SemanticMenuItem(ctx, T("menu.physics_layer_matrix"), "", physicsLayersOpen, canPhysicsLayers,
                         "menu.project.physics_layer_matrix"))
        ExecuteCommand("window.toggle.physics_layers", "menu");

    ImGui::Separator();

    // Preferences toggle
    const bool canPreferences = CanExecuteCommand("window.toggle.preferences");
    const bool preferencesOpen = IsCommandChecked("window.toggle.preferences");
    if (SemanticMenuItem(ctx, T("menu.preferences"), "", preferencesOpen, canPreferences, "menu.project.preferences"))
        ExecuteCommand("window.toggle.preferences", "menu");

    ImGui::EndMenu();
}

void MenuBarPanel::RenderEditMenu(InxGUIContext *ctx)
{
    if (!BeginSemanticMenu(ctx, T("menu.edit"), "menu.edit"))
        return;

    const bool canUndoCommand = CanExecuteCommand("edit.undo");
    if (SemanticMenuItem(ctx, T("menu.undo"), "Ctrl+Z", false, canUndoCommand, "menu.edit.undo"))
        ExecuteCommand("edit.undo", "menu");
    const bool canRedoCommand = CanExecuteCommand("edit.redo");
    if (SemanticMenuItem(ctx, T("menu.redo"), "Ctrl+Shift+Z", false, canRedoCommand, "menu.edit.redo"))
        ExecuteCommand("edit.redo", "menu");

    ImGui::Separator();

    for (const auto &[commandId, labelKey, shortcut, semanticId] :
         {std::tuple{"edit.copy", "menu.copy", "Ctrl+C", "menu.edit.copy"},
          std::tuple{"edit.cut", "menu.cut", "Ctrl+X", "menu.edit.cut"},
          std::tuple{"edit.paste", "menu.paste", "Ctrl+V", "menu.edit.paste"},
          std::tuple{"edit.rename", "menu.rename", "F2", "menu.edit.rename"},
          std::tuple{"edit.delete", "menu.delete", "Delete", "menu.edit.delete"}}) {
        const bool enabled = CanExecuteCommand(commandId);
        if (SemanticMenuItem(ctx, T(labelKey), shortcut, false, enabled, semanticId))
            ExecuteCommand(commandId, "menu");
    }

    ImGui::Separator();
    const bool canOpenPalette = CanExecuteCommand("command_palette.open");
    if (SemanticMenuItem(ctx, T("menu.command_palette"), "Ctrl+Shift+P", false, canOpenPalette,
                         "menu.edit.command_palette"))
        ExecuteCommand("command_palette.open", "menu");

    ImGui::EndMenu();
}

bool MenuBarPanel::ExecuteCommand(const std::string &commandId, const std::string &source,
                                  const std::string &argument) const
{
    return executeCommand && executeCommand(commandId, source, argument);
}

bool MenuBarPanel::CanExecuteCommand(const std::string &commandId, const std::string &argument) const
{
    return canExecuteCommand && canExecuteCommand(commandId, argument);
}

bool MenuBarPanel::IsCommandChecked(const std::string &commandId, const std::string &argument) const
{
    return isCommandChecked && isCommandChecked(commandId, argument);
}

// ════════════════════════════════════════════════════════════════════
// Scene menu — per-scene settings (environment / lighting)
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::RenderSceneMenu(InxGUIContext *ctx)
{
    if (!BeginSemanticMenu(ctx, T("menu.scene"), "menu.scene"))
        return;

    const bool canEnvironment = CanExecuteCommand("window.toggle.environment");
    const bool environmentOpen = IsCommandChecked("window.toggle.environment");
    if (SemanticMenuItem(ctx, T("menu.environment_settings"), "", environmentOpen, canEnvironment,
                         "menu.scene.environment_settings"))
        ExecuteCommand("window.toggle.environment", "menu");

    ImGui::EndMenu();
}

// ════════════════════════════════════════════════════════════════════
// Window menu
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::RenderWindowMenu(InxGUIContext *ctx)
{
    if (!BeginSemanticMenu(ctx, T("menu.window"), "menu.window"))
        return;

    RefreshWindowTypeCache();
    if (!m_cachedWindowTypes.empty()) {
        bool hasItems = false;
        for (const auto &info : m_cachedWindowTypes) {
            const auto segments = SplitMenuPath(info.menuPath);
            hasItems = hasItems || (!segments.empty() && segments.front() == "Window");
        }

        if (hasItems)
            RenderMenuContents(ctx, "Window", m_cachedWindowTypes);
        else
            SemanticMenuItem(ctx, T("menu.no_windows"), "", false, false, "menu.window.none");
    } else {
        SemanticMenuItem(ctx, T("menu.no_wm"), "", false, false, "menu.window.unavailable");
    }

    ImGui::Separator();

    const bool canResetLayout = CanExecuteCommand("window.reset_layout");
    if (SemanticMenuItem(ctx, T("menu.reset_layout"), "", false, canResetLayout, "menu.window.reset_layout"))
        ExecuteCommand("window.reset_layout", "menu");

    ImGui::EndMenu();
}

// ════════════════════════════════════════════════════════════════════
// Dynamic menus (everything between Project and Window)
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::RenderDynamicMenus(InxGUIContext *ctx)
{
    RefreshWindowTypeCache();
    if (m_cachedWindowTypes.empty())
        return;

    // Render each top-level menu.
    for (const auto &top : m_cachedTopMenus) {
        std::string key;
        for (const auto &info : m_cachedWindowTypes) {
            const auto segments = SplitMenuPath(info.menuPath);
            if (!segments.empty() && segments.front() == top && !info.menuPathKeys.empty()) {
                key = info.menuPathKeys.front();
                if (!key.empty())
                    break;
            }
        }
        RenderMenuGroup(ctx, top, ResolveMenuLabel(key, top), m_cachedWindowTypes);
    }
}

void MenuBarPanel::RefreshWindowTypeCache()
{
    if (!m_windowTypesDirty)
        return;
    if (!getRegisteredTypes)
        return;

    m_windowTypesDirty = false;
    m_cachedWindowTypes = getRegisteredTypes();
    m_cachedTopMenus.clear();
    for (const auto &info : m_cachedWindowTypes) {
        if (info.menuPath.empty() || info.menuPath == "Window")
            continue;
        std::string top = info.menuPath;
        const auto slash = top.find('/');
        if (slash != std::string::npos)
            top.resize(slash);
        if (top == "Window")
            continue;
        if (std::find(m_cachedTopMenus.begin(), m_cachedTopMenus.end(), top) == m_cachedTopMenus.end())
            m_cachedTopMenus.push_back(std::move(top));
    }
}

void MenuBarPanel::RenderMenuGroup(InxGUIContext *ctx, const std::string &topMenu, const std::string &translatedLabel,
                                   const std::vector<WindowTypeInfo> &types)
{
    if (!BeginSemanticMenu(ctx, translatedLabel, "menu." + topMenu))
        return;

    RenderMenuContents(ctx, topMenu, types);
    ImGui::EndMenu();
}

void MenuBarPanel::RenderMenuContents(InxGUIContext *ctx, const std::string &topMenu,
                                      const std::vector<WindowTypeInfo> &types)
{
    struct Entry
    {
        std::string typeId;
        std::string displayName;
        std::string titleKey;
    };

    struct MenuNode
    {
        std::string label;
        std::string translationKey;
        std::vector<Entry> entries;
        std::vector<MenuNode> children;
    };

    MenuNode root;
    root.label = topMenu;

    for (const auto &info : types) {
        const auto segments = SplitMenuPath(info.menuPath);
        if (segments.empty() || segments.front() != topMenu)
            continue;

        MenuNode *node = &root;
        for (size_t index = 1; index < segments.size(); ++index) {
            const auto found = std::find_if(node->children.begin(), node->children.end(),
                                            [&](const MenuNode &child) { return child.label == segments[index]; });
            const std::string key = index < info.menuPathKeys.size() ? info.menuPathKeys[index] : "";
            if (found == node->children.end()) {
                node->children.push_back(MenuNode{segments[index], key, {}, {}});
                node = &node->children.back();
            } else {
                node = &*found;
                if (node->translationKey.empty())
                    node->translationKey = key;
            }
        }
        node->entries.push_back(Entry{info.typeId, info.displayName, info.titleKey});
    }

    const auto renderEntry = [&](const Entry &entry) {
        const bool canOpen = CanExecuteCommand("window.open", entry.typeId);
        const bool isOpen = IsCommandChecked("window.open", entry.typeId);
        const std::string label = ResolveMenuLabel(entry.titleKey, entry.displayName);
        if (SemanticMenuItem(ctx, label, "", isOpen, canOpen, "window." + entry.typeId))
            ExecuteCommand("window.open", "menu", entry.typeId);
    };

    std::function<void(const MenuNode &, const std::string &)> renderChildren;
    renderChildren = [&](const MenuNode &node, const std::string &semanticPath) {
        for (const auto &entry : node.entries)
            renderEntry(entry);
        for (const auto &child : node.children) {
            const std::string childSemantic = semanticPath + "." + child.label;
            const std::string label = ResolveMenuLabel(child.translationKey, child.label);
            if (BeginSemanticMenu(ctx, label, childSemantic)) {
                renderChildren(child, childSemantic);
                ImGui::EndMenu();
            }
        }
    };

    renderChildren(root, "menu." + topMenu);
}

} // namespace infernux
