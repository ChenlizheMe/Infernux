#include <imgui.h>
#include <imgui_internal.h>

#include <cassert>

static void CheckTreePop(bool header)
{
    auto *context = ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(400, 300);
    io.DeltaTime = 1.0f / 60.0f;
    io.ConfigErrorRecoveryEnableAssert = false;
    io.ConfigErrorRecoveryEnableDebugLog = false;
    io.ConfigErrorRecoveryEnableTooltip = false;
    unsigned char *pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    int errors = 0;
    context->ErrorCallbackUserData = &errors;
    context->ErrorCallback = [](ImGuiContext *, void *user, const char *) {
        ++*static_cast<int *>(user);
    };
    for (int frame = 0; frame < 80; ++frame) {
        ImGui::NewFrame();
        ImGui::Begin("Plugin panel");
        if (frame == 0) {
            ImGui::SetNextItemOpen(header);
            if (header)
                ImGui::CollapsingHeader("Header");
            else
                assert(!ImGui::TreeNode("Closed tree"));
            const auto cursor = ImGui::GetCursorPos();
            const int idDepth = context->CurrentWindow->IDStack.Size;
            ImGui::TreePop();
            assert(context->CurrentWindow->DC.TreeDepth == 0);
            assert(context->CurrentWindow->IDStack.Size == idDepth);
            assert(ImGui::GetCursorPosX() == cursor.x);
        }
        ImGui::BeginGroup();
        ImGui::CollapsingHeader("Staged Changes");
        ImGui::CollapsingHeader("Changes");
        ImGui::EndGroup();
        ImGui::SetNextItemOpen(true);
        if (ImGui::TreeNode("Valid tree"))
            ImGui::TreePop();
        ImGui::End();
        ImGui::Render();
        assert(errors == 1);
    }
    ImGui::DestroyContext();
}

int main()
{
    CheckTreePop(false);
    CheckTreePop(true);
}
