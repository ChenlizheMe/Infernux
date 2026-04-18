#include "MenuBarPanel.h"

#include <algorithm>
#include <cctype>

namespace infernux
{

// ════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════

MenuBarPanel::MenuBarPanel() = default;

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

// ════════════════════════════════════════════════════════════════════
// Render
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::OnRender(InxGUIContext *ctx)
{
    // Handle global shortcuts before menu logic
    HandleShortcuts(ctx);

    // Check for window close request (SDL_EVENT_QUIT intercepted by C++)
    if (isCloseRequested && onRequestClose) {
        if (isCloseRequested())
            onRequestClose();
    }

    // Style overrides
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, EditorTheme::TOOLBAR_FRAME_PAD);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, EditorTheme::TOOLBAR_ITEM_SPC);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, EditorTheme::TOOLBAR_WIN_PAD);
    ImGui::PushStyleColor(ImGuiCol_MenuBarBg, EditorTheme::MENU_BAR_BG);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, EditorTheme::POPUP_BG);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, EditorTheme::HEADER_HOVERED);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, EditorTheme::HEADER_ACTIVE);

    if (ImGui::BeginMainMenuBar()) {
        RenderProjectMenu(ctx);
        RenderDynamicMenus(ctx);
        RenderWindowMenu(ctx);
        ImGui::EndMainMenuBar();
    }

    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(3);

    // Note: floating sub-panels (BuildSettings, Preferences, PhysicsLayerMatrix)
    // and save-confirmation popup are rendered from Python side.
}

// ════════════════════════════════════════════════════════════════════
// Keyboard shortcuts
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::HandleShortcuts(InxGUIContext *ctx)
{
    bool ctrl = ctx->IsKeyDown(KEY_LEFT_CTRL) || ctx->IsKeyDown(KEY_RIGHT_CTRL);
    if (!ctrl)
        return;

    if (ctx->IsKeyPressed(KEY_S) && onSave)
        onSave();

    if (ctx->IsKeyPressed(KEY_N) && onNewScene)
        onNewScene();

    if (ctx->IsKeyPressed(KEY_Z)) {
        if (canUndo && canUndo() && onUndo)
            onUndo();
    }

    if (ctx->IsKeyPressed(KEY_Y)) {
        if (canRedo && canRedo() && onRedo)
            onRedo();
    }
}

// ════════════════════════════════════════════════════════════════════
// Project menu
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::RenderProjectMenu(InxGUIContext *ctx)
{
    if (!ImGui::BeginMenu(T("menu.project").c_str()))
        return;

    // Build Settings toggle
    bool bsOpen = isBuildSettingsOpen ? isBuildSettingsOpen() : false;
    if (ImGui::MenuItem(T("menu.build_settings").c_str(), "", bsOpen, true)) {
        if (toggleBuildSettings)
            toggleBuildSettings();
    }

    // Physics Layer Matrix toggle
    bool plOpen = isPhysicsLayerMatrixOpen ? isPhysicsLayerMatrixOpen() : false;
    if (ImGui::MenuItem(T("menu.physics_layer_matrix").c_str(), "", plOpen, true)) {
        if (togglePhysicsLayerMatrix)
            togglePhysicsLayerMatrix();
    }

    ImGui::Separator();

    // Preferences toggle
    bool prefOpen = isPreferencesOpen ? isPreferencesOpen() : false;
    if (ImGui::MenuItem(T("menu.preferences").c_str(), "", prefOpen, true)) {
        if (togglePreferences)
            togglePreferences();
    }

    ImGui::EndMenu();
}

// ════════════════════════════════════════════════════════════════════
// Window menu
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::RenderWindowMenu(InxGUIContext *ctx)
{
    if (!ImGui::BeginMenu(T("menu.window").c_str()))
        return;

    if (getRegisteredTypes && getOpenWindows) {
        auto types = getRegisteredTypes();
        auto openWins = getOpenWindows();

        bool hasItems = false;
        for (const auto &info : types) {
            if (info.menuPath != "Window")
                continue;
            hasItems = true;

            bool isOpen = false;
            auto it = openWins.find(info.typeId);
            if (it != openWins.end())
                isOpen = it->second;

            bool canCreate = !(info.singleton && isOpen);

            if (ImGui::MenuItem(info.displayName.c_str(), "", isOpen, canCreate)) {
                if (isOpen) {
                    if (closeWindow)
                        closeWindow(info.typeId);
                } else {
                    if (openWindow)
                        openWindow(info.typeId);
                }
            }
        }

        if (!hasItems) {
            ImGui::MenuItem(T("menu.no_windows").c_str(), "", false, false);
        }
    } else {
        ImGui::MenuItem(T("menu.no_wm").c_str(), "", false, false);
    }

    ImGui::Separator();

    if (ImGui::MenuItem(T("menu.reset_layout").c_str(), "", false, true)) {
        if (resetLayout)
            resetLayout();
    }

    ImGui::EndMenu();
}

// ════════════════════════════════════════════════════════════════════
// Dynamic menus (everything between Project and Window)
// ════════════════════════════════════════════════════════════════════

void MenuBarPanel::RenderDynamicMenus(InxGUIContext *ctx)
{
    if (!getRegisteredTypes || !getOpenWindows)
        return;

    auto types = getRegisteredTypes();
    auto openWins = getOpenWindows();

    // Discover unique top-level menu names (everything except "Window").
    // Preserve insertion order so menus appear in panel registration order.
    std::vector<std::string> topMenus;
    for (const auto &info : types) {
        if (info.menuPath == "Window")
            continue;
        std::string top = info.menuPath;
        auto slash = top.find('/');
        if (slash != std::string::npos)
            top = top.substr(0, slash);

        bool found = false;
        for (const auto &t : topMenus)
            if (t == top) { found = true; break; }
        if (!found)
            topMenus.push_back(top);
    }

    // Render each top-level menu.
    for (const auto &top : topMenus) {
        // Build i18n key: "Animation" -> "menu.animation"
        std::string key = "menu." + top;
        for (auto &c : key)
            if (c == ' ') c = '_';
            else c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        std::string label = T(key);
        // Fallback: if T() returned the key tail, use original name
        if (label == key.substr(key.rfind('.') + 1))
            label = top;

        RenderMenuGroup(top, label, types, openWins);
    }
}

void MenuBarPanel::RenderMenuGroup(
    const std::string &topMenu,
    const std::string &translatedLabel,
    const std::vector<WindowTypeInfo> &types,
    const std::map<std::string, bool> &openWins)
{
    // Collect entries belonging to this top-level menu.
    struct Entry {
        std::string subMenu;     // "" = direct child, else sub-menu label
        std::string typeId;
        std::string displayName;
        bool singleton;
    };

    std::vector<Entry> entries;
    std::vector<std::string> subMenuOrder;

    const size_t topLen = topMenu.size();

    for (const auto &info : types) {
        // Must start with topMenu
        if (info.menuPath.rfind(topMenu, 0) != 0)
            continue;
        // Must be exactly topMenu or topMenu/...
        if (info.menuPath.size() > topLen && info.menuPath[topLen] != '/')
            continue;

        Entry e;
        e.typeId      = info.typeId;
        e.displayName = info.displayName;
        e.singleton   = info.singleton;

        if (info.menuPath.size() > topLen + 1)
            e.subMenu = info.menuPath.substr(topLen + 1);

        entries.push_back(e);

        if (!e.subMenu.empty()) {
            bool found = false;
            for (const auto &s : subMenuOrder)
                if (s == e.subMenu) { found = true; break; }
            if (!found)
                subMenuOrder.push_back(e.subMenu);
        }
    }

    if (entries.empty())
        return;

    if (!ImGui::BeginMenu(translatedLabel.c_str()))
        return;

    // Lambda: render a single menu-item toggle
    auto renderItem = [&](const Entry &e) {
        bool isOpen = false;
        auto it = openWins.find(e.typeId);
        if (it != openWins.end())
            isOpen = it->second;
        bool canCreate = !(e.singleton && isOpen);

        if (ImGui::MenuItem(e.displayName.c_str(), "", isOpen, canCreate)) {
            if (isOpen) {
                if (closeWindow) closeWindow(e.typeId);
            } else {
                if (openWindow) openWindow(e.typeId);
            }
        }
    };

    // Top-level items (menuPath == topMenu exactly)
    for (const auto &e : entries) {
        if (e.subMenu.empty())
            renderItem(e);
    }

    // Sub-menus
    for (const auto &sm : subMenuOrder) {
        // Build i18n key: e.g. "Animation" + "2D Animation" -> "menu.animation_2d_animation"
        std::string smKey = "menu." + topMenu + "_" + sm;
        for (auto &c : smKey)
            if (c == ' ') c = '_';
            else c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        std::string smLabel = T(smKey);
        if (smLabel == smKey.substr(smKey.rfind('.') + 1))
            smLabel = sm;

        if (ImGui::BeginMenu(smLabel.c_str())) {
            for (const auto &e : entries) {
                if (e.subMenu == sm)
                    renderItem(e);
            }
            ImGui::EndMenu();
        }
    }

    ImGui::EndMenu();
}

} // namespace infernux
