#pragma once

#include <imgui.h>
#include <cstdint>
#include <filesystem>
#include <platform/filesystem/InxPath.h>
#include <string>

// CPU UI fixtures explicitly load the bundled monospace face. ImGui's
// embedded fonts are excluded, including in standalone test programs.
inline ImFont *InstallImGuiTestFont(float size = 16.0f)
{
    const auto root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
    const auto path = infernux::FromFsPath(root / "python/infernux/resources/fonts/MapleMono-CN-Regular.ttf");
    auto &io = ImGui::GetIO();
    io.FontDefault = io.Fonts->AddFontFromFileTTF(path.c_str(), size);
    return io.FontDefault;
}

inline ImGuiContext *CreateImGuiTestContext()
{
    auto *context = ImGui::CreateContext();
    InstallImGuiTestFont();
    return context;
}

template <typename GuiContext> inline void ConfigureImGuiTestGUI(GuiContext &context)
{
    GuiContext::s_monoFont = ImGui::GetIO().FontDefault;
    // CPU-only draw tests have no uploaded icon descriptors.
    context.resolveEditorIcon = [](const std::string &) { return uint64_t{0}; };
}
