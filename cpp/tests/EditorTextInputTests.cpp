#include <function/renderer/gui/InxGUIContext.h>
#include <imgui_internal.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void Check(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct Fixture
{
    infernux::InxGUIContext context;
    Fixture()
    {
        ImGui::CreateContext();
        auto &io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.DisplaySize = ImVec2(900, 600);
        io.DeltaTime = 1.0f / 60.0f;
        io.ConfigInputTrickleEventQueue = false;
        unsigned char *pixels;
        int width, height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    }
    ~Fixture()
    {
        ImGui::DestroyContext();
    }
    void Begin()
    {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(850, 540), ImGuiCond_Always);
        ImGui::Begin("Text editing", nullptr, ImGuiWindowFlags_NoSavedSettings);
    }
    void End()
    {
        ImGui::End();
        ImGui::Render();
    }
};

void VerifyBatchIdle(bool multiline, const std::string &value, bool mixed)
{
    Fixture fixture;
    infernux::PropertyDesc descriptor;
    descriptor.type = infernux::PropertyDesc::String;
    descriptor.widgetId = "##dialogue";
    descriptor.label = "Dialogue";
    descriptor.sVal = value;
    descriptor.multiline = multiline;
    descriptor.mixed = mixed;
    for (int frame = 0; frame < 3; ++frame) {
        fixture.Begin();
        int active = -2, deactivated = -2;
        const auto edits = fixture.context.RenderPropertyBatch({descriptor}, 140, &active, &deactivated);
        Check(edits.empty(), "An idle String batch manufactured an edit");
        Check(active == -1 && deactivated == -1, "Idle text unexpectedly owns editing");
        fixture.End();
    }
}

// Exercise genuine mouse/key/text events through each public native input path.
void VerifyTextGrowth(int route)
{
    Fixture fixture;
    std::string value(5000, 'a');
    value += "原始对话";
    const std::string initial = value;
    const std::string addition = std::string(5000, 'z') + "新增线索";
    int changes = 0;
    ImVec2 itemMin, itemMax;
    auto frame = [&] {
        fixture.Begin();
        bool edited = false;
        if (route < 2) {
            infernux::PropertyDesc descriptor;
            descriptor.type = infernux::PropertyDesc::String;
            descriptor.widgetId = "##dialogue";
            descriptor.label = "Dialogue";
            descriptor.multiline = route == 1;
            descriptor.sVal = value;
            const auto result = fixture.context.RenderPropertyBatch({descriptor}, 140);
            Check(result.size() <= 1, "One input generated duplicate changes");
            if (!result.empty()) {
                value = result.front().sVal;
                edited = true;
            }
        } else if (route == 2) {
            edited = fixture.context.TextInput("##dialogue", value);
        } else if (route == 3) {
            edited = fixture.context.TextArea("##dialogue", value);
        } else {
            edited = fixture.context.InputTextWithHint("##dialogue", "Dialogue", value);
        }
        itemMin = ImGui::GetItemRectMin();
        itemMax = ImGui::GetItemRectMax();
        changes += edited;
        fixture.End();
    };
    frame();
    frame();
    Check(value == initial && changes == 0, "Displaying full text changed it");
    auto &io = ImGui::GetIO();
    io.AddMousePosEvent(itemMin.x + (itemMax.x - itemMin.x) * 0.75f, itemMin.y + 8);
    frame();
    io.AddMouseButtonEvent(0, true);
    frame();
    io.AddMouseButtonEvent(0, false);
    frame();
    Check(ImGui::GetCurrentContext()->ActiveId != 0, "Text input did not activate");
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    io.AddKeyEvent(ImGuiKey_End, true);
    frame();
    io.AddKeyEvent(ImGuiKey_End, false);
    io.AddKeyEvent(ImGuiMod_Ctrl, false);
    frame();
    io.AddInputCharactersUTF8(addition.c_str());
    frame();
    Check(value == initial + addition, "Editing lost the full UTF-8 prefix or failed to grow");
    Check(changes == 1, "One input event did not produce exactly one change");
    frame();
    Check(changes == 1, "An idle follow-up frame repeated a text edit");
}
} // namespace

int main()
try {
    int cases = 0;
    for (bool multiline : {false, true})
        for (const std::string &value : {std::string(), std::string(255, 'a'), std::string(4095, 'b'),
                                         std::string(5000, 'c'), std::string(4094, 'd') + "谜题线索"})
            for (bool mixed : {false, true}) {
                VerifyBatchIdle(multiline, value, mixed);
                ++cases;
            }
    for (int route = 0; route < 5; ++route) {
        VerifyTextGrowth(route);
        ++cases;
    }
    std::cout << cases << " text editing cases passed\n";
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
