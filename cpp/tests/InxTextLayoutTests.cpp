#include <function/renderer/gui/InxTextLayout.h>

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace
{
struct AllocationAudit
{
    ImGuiMemAllocFunc allocate;
    ImGuiMemFreeFunc free;
    void *context;
    std::unordered_set<void *> live;

    static void *Allocate(size_t size, void *user)
    {
        auto &audit = *static_cast<AllocationAudit *>(user);
        void *pointer = audit.allocate(size, audit.context);
        if (pointer)
            audit.live.insert(pointer);
        return pointer;
    }
    static void Free(void *pointer, void *user)
    {
        auto &audit = *static_cast<AllocationAudit *>(user);
        if (pointer)
            assert(audit.live.erase(pointer) == 1);
        audit.free(pointer, audit.context);
    }
};

// Independent glyph decoding/measurement is the draw oracle. The production
// renderer must consume layout advances without changing the resulting mesh.
void ReferenceDraw(ImDrawList &draw, const infernux::textlayout::TextLayoutResult &layout, float alignX, float alignY,
                   float spacing)
{
    const char *base = layout.text.data();
    for (size_t index = 0; index < layout.lines.size(); ++index) {
        const auto &line = layout.lines[index];
        float x = 7.25f + (220.f - line.width) * alignX;
        const float y = 9.75f + (180.f - layout.totalHeight) * alignY + index * layout.lineAdvance;
        int count = 0;
        float width = 0;
        for (const char *cursor = base + line.startOffset; cursor < base + line.endOffset;) {
            const char *start = cursor;
            unsigned int codepoint = 0;
            const int consumed = ImTextCharFromUtf8(&codepoint, cursor, base + line.endOffset);
            if (consumed <= 0)
                break;
            cursor += consumed;
            if (codepoint == '\r' || codepoint == '\n')
                continue;
            auto *font = infernux::textlayout::ResolveGlyphFont(layout, static_cast<ImWchar>(codepoint));
            const float size = layout.logicalFontSize * infernux::textlayout::ResolveEmRasterScale(font);
            const float advance = font->CalcTextSizeA(size, FLT_MAX, 0, start, cursor).x;
            if (count++) {
                x += spacing;
                width += spacing;
            }
            font->RenderChar(&draw, size, ImVec2(x, y), IM_COL32_WHITE, static_cast<ImWchar>(codepoint));
            x += advance;
            width += advance;
        }
        assert(std::abs(width - line.width) < .002f);
    }
}

void WriteU16(unsigned char *target, uint16_t value)
{
    target[0] = static_cast<unsigned char>(value >> 8u);
    target[1] = static_cast<unsigned char>(value);
}

void WriteU32(unsigned char *target, uint32_t value)
{
    target[0] = static_cast<unsigned char>(value >> 24u);
    target[1] = static_cast<unsigned char>(value >> 16u);
    target[2] = static_cast<unsigned char>(value >> 8u);
    target[3] = static_cast<unsigned char>(value);
}
} // namespace

int main()
{
    std::array<unsigned char, 128> font{};
    WriteU16(font.data() + 4u, 2u);

    unsigned char *headRecord = font.data() + 12u;
    WriteU32(headRecord, 0x68656164u);
    WriteU32(headRecord + 8u, 64u);
    WriteU32(headRecord + 12u, 20u);

    unsigned char *hheaRecord = font.data() + 28u;
    WriteU32(hheaRecord, 0x68686561u);
    WriteU32(hheaRecord + 8u, 96u);
    WriteU32(hheaRecord + 12u, 8u);

    WriteU16(font.data() + 64u + 18u, 1000u);
    WriteU16(font.data() + 96u + 4u, 1100u);
    WriteU16(font.data() + 96u + 6u, static_cast<uint16_t>(-400));

    const float scale = infernux::textlayout::ReadSfntEmRasterScale(font.data(), static_cast<int>(font.size()));
    assert(scale > 1.499f && scale < 1.501f);
    assert(infernux::textlayout::ReadSfntEmRasterScale(nullptr, 0) == 1.0f);

    AllocationAudit audit;
    ImGui::GetAllocatorFunctions(&audit.allocate, &audit.free, &audit.context);
    ImGui::SetAllocatorFunctions(AllocationAudit::Allocate, AllocationAudit::Free, &audit);
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = ImVec2(1280, 720);
    io.DeltaTime = 1.f / 60;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_RendererHasVtxOffset;
    io.Fonts->AddFontDefault();
    ImGui::NewFrame();
    std::vector<std::string> errors;
    auto &logger = infernux::InxLog::GetInstance();
    const size_t sink =
        logger.AddSink([&errors](infernux::LogLevel level, const char *, int, const std::string &message, bool) {
            if (level == infernux::LOG_ERROR)
                errors.push_back(message);
        });
    infernux::textlayout::ClearFontCache();
    constexpr const char *missingPath = "Z:/infernux-tests/font-that-does-not-exist.ttf";
    assert(infernux::textlayout::ResolveFont(missingPath) == nullptr);
    assert(infernux::textlayout::ResolveFont(missingPath) == nullptr);
    assert(errors.size() == 1u);
    assert(errors[0].find("does not exist") != std::string::npos);
    infernux::textlayout::TextLayoutParams fallbackParams{};
    fallbackParams.text = "fallback";
    fallbackParams.fontSize = 18.0f;
    fallbackParams.fallbackFontPaths = {missingPath};
    const auto missingFallback = infernux::textlayout::LayoutText(fallbackParams);
    assert(missingFallback.font == nullptr);
    assert(missingFallback.lines.empty());
    assert(errors.size() == 1u);
    fallbackParams.fallbackFontPaths = {""};
    const auto emptyFallback = infernux::textlayout::LayoutText(fallbackParams);
    assert(emptyFallback.font == nullptr);
    assert(errors.size() == 2u);
    const auto repeatedEmptyFallback = infernux::textlayout::LayoutText(fallbackParams);
    assert(repeatedEmptyFallback.font == nullptr);
    assert(errors.size() == 2u);

    const std::filesystem::path repositoryRoot =
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
    const std::string latinFont = (repositoryRoot / "external/imgui/misc/fonts/Roboto-Medium.ttf").string();
    const std::string cjkFont = (repositoryRoot / "python/Infernux/resources/fonts/PingFangSC-Regular.ttf").string();
    assert(std::filesystem::exists(latinFont));
    assert(std::filesystem::exists(cjkFont));
    infernux::textlayout::TextLayoutParams explicitChain{};
    explicitChain.text = "A\xE4\xB8\xAD";
    explicitChain.fontPath = latinFont;
    explicitChain.fontSize = 32.0f;
    explicitChain.fallbackFontPaths = {cjkFont};
    const auto chainedLayout = infernux::textlayout::LayoutText(explicitChain);
    assert(chainedLayout.font != nullptr);
    assert(chainedLayout.fallbackFonts.size() == 1u);
    assert(infernux::textlayout::ResolveGlyphFont(chainedLayout, static_cast<ImWchar>('A')) == chainedLayout.font);
    assert(infernux::textlayout::ResolveGlyphFont(chainedLayout, static_cast<ImWchar>(0x4E2D)) ==
           chainedLayout.fallbackFonts[0]);
    assert(chainedLayout.totalWidth > 0.0f);
    {
        ImDrawList drawList(ImGui::GetDrawListSharedData());
        auto resetDraw = [&] {
            drawList._ResetForNewFrame();
            drawList.PushTextureID(io.Fonts->TexRef);
            drawList.PushClipRectFullScreen();
        };
        // CPU-only layout + glyph tessellation, not GPU or total frame time.
        for (bool fallback : {false, true}) {
            auto params = explicitChain;
            params.fontSize = 18;
            params.text = fallback ? "CPU / GPU - UI 041 / 中文文本换行测量" : "CPU / GPU - UI 041 / dynamic label";
            if (!fallback)
                params.fallbackFontPaths.clear();
            params.wrapWidth = 220;
            params.letterSpacing = .35f;
            std::vector<double> samples;
            for (int frame = 0; frame < 20; ++frame) {
                resetDraw();
                auto start = std::chrono::steady_clock::now();
                for (int label = 0; label < 1000; ++label) {
                    const auto layout = infernux::textlayout::LayoutText(params);
                    infernux::textlayout::RenderTextBox(&drawList, 0, 0, 220, 80, layout, IM_COL32_WHITE, .5f, .5f,
                                                        params.letterSpacing);
                }
                if (frame >= 4)
                    samples.push_back(
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                assert(drawList.VtxBuffer.Size > 0);
            }
            std::sort(samples.begin(), samples.end());
            std::cout << "TEXT_LAYOUT_BENCH fallback=" << fallback
                      << " labels=1000 p50_ms=" << samples[samples.size() / 2] << std::endl;
        }
        ImDrawList reference(ImGui::GetDrawListSharedData());
        auto resetReference = [&] {
            reference._ResetForNewFrame();
            reference.PushTextureID(io.Fonts->TexRef);
            reference.PushClipRectFullScreen();
        };
        const std::vector<std::string> strings{
            "",
            "\n",
            "\r\n",
            "\n\n",
            "abc\n",
            "a\r\nb\rc",
            " aa   bb cc ",
            "\t\t alpha\t beta\t",
            "longwordwithoutbreaks",
            "path/to-file.ext, next!",
            "英文字母 A 中文混排。下一行！",
            "a 中\r文  b\n\t末尾\n",
            "emoji \xF0\x9F\x98\x80",
            std::string("before\0after", 12),
            std::string("broken \xE4\xB8", 9),
            std::string(4096, ' ') + "after a long whitespace run",
        };
        size_t cases = 0;
        for (const auto &text : strings) {
            for (float wrap : {0.f, 1.f, 23.f, 85.f, 220.f}) {
                for (float spacing : {-1.f, 0.f, .35f, 2.f}) {
                    for (float size : {12.f, 18.f, 32.f}) {
                        auto params = explicitChain;
                        params.text = text;
                        params.wrapWidth = wrap;
                        params.letterSpacing = spacing;
                        params.fontSize = size;
                        params.lineHeight = .75f;
                        const auto layout = infernux::textlayout::LayoutText(params);
                        assert(layout.font && layout.logicalFontSize == size);
                        float widest = 0;
                        for (const auto &line : layout.lines) {
                            assert(line.startOffset <= line.endOffset && line.endOffset <= text.size());
                            assert(line.glyphStart <= line.glyphEnd && line.glyphEnd <= layout.glyphs.size());
                            widest = std::max(widest, line.width);
                        }
                        assert(std::abs(layout.totalWidth - widest) < .002f);
                        if (!layout.lines.empty())
                            assert(std::abs(layout.totalHeight -
                                            (size + layout.lineAdvance * (layout.lines.size() - 1))) < .002f);
                        const float alignment = static_cast<float>(cases % 3) * .5f;
                        resetReference();
                        ReferenceDraw(reference, layout, alignment, 1 - alignment, spacing); // warm atlas
                        resetDraw();
                        infernux::textlayout::RenderTextBox(&drawList, 7.25f, 9.75f, 227.25f, 189.75f, layout,
                                                            IM_COL32_WHITE, alignment, 1 - alignment, spacing);
                        resetReference();
                        ReferenceDraw(reference, layout, alignment, 1 - alignment, spacing);
                        assert(drawList.VtxBuffer.Size == reference.VtxBuffer.Size);
                        assert(drawList.IdxBuffer.Size == reference.IdxBuffer.Size);
                        for (int vertex = 0; vertex < drawList.VtxBuffer.Size; ++vertex) {
                            const auto &actual = drawList.VtxBuffer[vertex];
                            const auto &expected = reference.VtxBuffer[vertex];
                            assert(std::abs(actual.pos.x - expected.pos.x) < .002f);
                            assert(std::abs(actual.pos.y - expected.pos.y) < .002f);
                            assert(actual.uv.x == expected.uv.x && actual.uv.y == expected.uv.y);
                            assert(actual.col == expected.col);
                        }
                        ++cases;
                    }
                }
            }
        }
        std::cout << "TEXT_LAYOUT_GEOMETRY_EQUIVALENCE cases=" << cases << std::endl;
    } // Draw-list storage is released before the atlas/context.

    const int fontCount = io.Fonts->Fonts.Size;
    const auto oldDirectory = std::filesystem::current_path();
    std::filesystem::current_path(infernux::ToFsPath(latinFont).parent_path());
    assert(infernux::textlayout::ResolveFont("Roboto-Medium.ttf") == chainedLayout.font);
    assert(infernux::textlayout::ResolveFont("./Roboto-Medium.ttf") == chainedLayout.font);
    std::filesystem::current_path(infernux::ToFsPath(cjkFont).parent_path());
    assert(infernux::textlayout::ResolveFont("Roboto-Medium.ttf") == nullptr);
    assert(infernux::textlayout::ResolveFont("PingFangSC-Regular.ttf") == chainedLayout.fallbackFonts[0]);
    std::filesystem::current_path(oldDirectory);
    assert(io.Fonts->Fonts.Size == fontCount);
    ImGui::EndFrame();
    infernux::textlayout::ClearFontCache();
    assert(infernux::textlayout::GetFontCache().empty());
    assert(infernux::textlayout::GetMissingFonts().empty());
    io.Fonts->Clear();
    io.Fonts->AddFontDefault();
    ImGui::NewFrame();
    const auto reloaded = infernux::textlayout::LayoutText(explicitChain);
    assert(reloaded.font && reloaded.fallbackFonts.size() == 1);
    assert(std::abs(reloaded.totalWidth - chainedLayout.totalWidth) < .002f);
    assert(reloaded.totalHeight == chainedLayout.totalHeight);
    ImGui::EndFrame();
    logger.RemoveSink(sink);
    infernux::textlayout::ClearFontCache();
    ImGui::DestroyContext();
    assert(audit.live.empty());
    ImGui::SetAllocatorFunctions(audit.allocate, audit.free, audit.context);
    return 0;
}
