#pragma once

#include <imgui.h>
#include <filesystem>
#include <platform/filesystem/InxPath.h>

// Native UI tests use the same bundled face as the editor. ImGui's embedded
// fonts are excluded from the build, including in standalone test programs.
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
