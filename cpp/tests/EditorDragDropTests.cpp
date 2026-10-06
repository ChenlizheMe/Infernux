#include <function/renderer/gui/InxGUIContext.h>
#include <imgui.h>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace
{
using Value = std::variant<uint64_t, std::string>;

void Check(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct Fixture
{
    Fixture()
    {
        ImGui::CreateContext();
        auto &io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        io.DisplaySize = ImVec2(700, 400);
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
};

void VerifyDelivery(const Value &value, bool any, bool rejectWrongKind, bool previewMode = false, bool opaque = false)
{
    Fixture fixture;
    // Panel contexts need not be the same object. The type belongs to the
    // payload, never to a cached source/receiver wrapper instance.
    infernux::InxGUIContext sourceContext, targetContext;
    const bool integer = std::holds_alternative<uint64_t>(value);
    const std::string payloadType = integer ? "GOID" : "SCRIPT_FILE";
    int sourceFrames = 0, targetFrames = 0, deliveries = 0, previews = 0;
    for (int frame = 0; frame < 8; ++frame) {
        auto &io = ImGui::GetIO();
        io.AddMousePosEvent(frame < 3 ? 60.f : (frame == 3 ? 90.f : 300.f), 60.f);
        if (frame == 2)
            io.AddMouseButtonEvent(0, true);
        if (frame == 6)
            io.AddMouseButtonEvent(0, false);
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(600, 300), ImGuiCond_Always);
        ImGui::Begin("Payload contract", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoSavedSettings);
        ImGui::SetCursorPos(ImVec2(20, 30));
        ImGui::Button("Source", ImVec2(120, 40));
        if (sourceContext.BeginDragDropSource()) {
            ++sourceFrames;
            if (opaque) {
                const auto &data = std::get<std::string>(value);
                ImGui::SetDragDropPayload(payloadType.c_str(), data.data(), data.size());
            } else {
                std::visit([&](const auto &data) { sourceContext.SetDragDropPayload(payloadType, data); }, value);
            }
            ImGui::TextUnformatted("Move asset");
            sourceContext.EndDragDropSource();
        }
        ImGui::SetCursorPos(ImVec2(250, 30));
        ImGui::Button("Target", ImVec2(120, 40));
        if (targetContext.BeginDragDropTarget()) {
            ++targetFrames;
            uint64_t number = 123;
            std::string text = "unchanged", type;
            bool isInteger = false;
            Check(!targetContext.AcceptDragDropPayload("OTHER_TYPE", &text), "Accepted a different payload type");
            if (rejectWrongKind) {
                const bool wrong = integer ? targetContext.AcceptDragDropPayload(payloadType, &text)
                                           : targetContext.AcceptDragDropPayload(payloadType, &number);
                Check(!wrong && number == 123 && text == "unchanged", "Wrong typed receiver consumed a value");
            }
            bool accepted;
            if (previewMode) {
                accepted =
                    targetContext.AcceptDragDropPayload(payloadType, &text, ImGuiDragDropFlags_AcceptBeforeDelivery);
                type = payloadType;
            } else if (any) {
                accepted = targetContext.AcceptAnyDragDropPayload(&type, &number, &text, &isInteger);
            } else {
                // Same dispatch order as the public Python binding.
                isInteger = targetContext.AcceptDragDropPayload(payloadType, &number);
                accepted = isInteger || targetContext.AcceptDragDropPayload(payloadType, &text);
                type = payloadType;
            }
            if (opaque) {
                Check(!accepted, "Infernux accepted an opaque native payload as a string/integer");
                // A private native payload (e.g. component reorder) still has
                // its own receiver; our typed API must leave it untouched.
                if (const auto *raw = ImGui::AcceptDragDropPayload(payloadType.c_str())) {
                    ++deliveries;
                    Check(frame == 6, "Opaque payload was delivered before release");
                    Check(std::string(static_cast<const char *>(raw->Data), raw->DataSize) ==
                              std::get<std::string>(value),
                          "Typed receiver corrupted opaque native data");
                }
            }
            if (accepted) {
                const auto *payload = ImGui::GetDragDropPayload();
                if (payload->IsDelivery()) {
                    ++deliveries;
                    Check(frame == 6, "Payload was delivered before the mouse release");
                } else {
                    Check(previewMode && frame < 6, "Ordinary accept delivered a preview");
                    previews += payload->IsPreview();
                }
                Check(type == payloadType && isInteger == integer, "Payload kind or type changed in transit");
                Check(integer ? number == std::get<uint64_t>(value) : text == std::get<std::string>(value),
                      "Payload value changed in transit");
            }
            targetContext.EndDragDropTarget();
        }
        ImGui::End();
        ImGui::Render();
    }
    Check(sourceFrames > 0 && targetFrames > 0 && deliveries == 1, "Pointer sequence must deliver exactly once");
    Check(!previewMode || previews > 0, "Preview receiver did not receive a hover preview");
}
} // namespace

int main()
{
    const std::vector<Value> values = {std::string("choice"),
                                       std::string("choiceA"),
                                       std::string("choiceAB"),
                                       std::string("中文a"),
                                       std::string(),
                                       std::string("a\0bcdef", 7),
                                       std::string(5000, 'x'),
                                       uint64_t(0),
                                       uint64_t(7),
                                       std::numeric_limits<uint64_t>::max()};
    int failures = 0, cases = 0;
    for (const auto &value : values) {
        for (bool any : {false, true}) {
            for (bool rejectWrongKind : {false, true}) {
                ++cases;
                try {
                    VerifyDelivery(value, any, rejectWrongKind);
                } catch (const std::exception &error) {
                    ++failures;
                    std::cerr << "case " << cases << ": " << error.what() << '\n';
                }
            }
        }
        if (std::holds_alternative<std::string>(value)) {
            ++cases;
            try {
                VerifyDelivery(value, false, true, true);
            } catch (const std::exception &error) {
                ++failures;
                std::cerr << "preview case " << cases << ": " << error.what() << '\n';
            }
        }
    }
    // Unknown native types, truncated integers and invalid tags are never
    // inferred as text. They retain their own raw native receive contract.
    for (const std::string &bytes : {std::string("12345678"), std::string(24, '\0'), std::string("INX\1short"),
                                     std::string("INX\1"), std::string("INX\3unknown"), std::string("I")}) {
        ++cases;
        try {
            VerifyDelivery(bytes, true, true, false, true);
        } catch (const std::exception &error) {
            ++failures;
            std::cerr << "opaque case " << cases << ": " << error.what() << '\n';
        }
    }
    std::cout << cases - failures << '/' << cases << " real-pointer drag/drop cases passed\n";
    return failures ? 1 : 0;
}
