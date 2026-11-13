#include "StatusBarPanel.h"
#include "ConsolePanel.h"
#include <function/editor/EditorThemeRegistry.h>
#include <function/renderer/gui/InxGUISemantics.h>

#include <algorithm>
#include <cmath>
#include <cstring>

namespace infernux
{

namespace
{

enum class StatusGlyph
{
    None,
    Error,
    Warning,
    Info,
    Check,
};

// Severity reads by shape as well as colour: error octagon, warning
// triangle, info square, success check. `size` is the glyph's square extent.
void DrawStatusGlyph(ImDrawList *draw, const ImVec2 &center, float size, StatusGlyph glyph, ImU32 color, float dpi)
{
    const float r = size * 0.5f;
    switch (glyph) {
    case StatusGlyph::Error: {
        constexpr float step = 3.14159265f / 4.0f;
        for (int i = 0; i < 8; ++i) {
            const float angle = step * (static_cast<float>(i) + 0.5f);
            draw->PathLineTo(ImVec2(center.x + std::cos(angle) * r, center.y + std::sin(angle) * r));
        }
        draw->PathFillConvex(color);
        break;
    }
    case StatusGlyph::Warning:
        draw->AddTriangleFilled(ImVec2(center.x, center.y - r), ImVec2(center.x + r, center.y + r * 0.8f),
                                ImVec2(center.x - r, center.y + r * 0.8f), color);
        break;
    case StatusGlyph::Info:
        draw->AddRectFilled(ImVec2(center.x - r * 0.8f, center.y - r * 0.8f),
                            ImVec2(center.x + r * 0.8f, center.y + r * 0.8f), color);
        break;
    case StatusGlyph::Check: {
        const ImVec2 points[3] = {ImVec2(center.x - r, center.y), ImVec2(center.x - r * 0.3f, center.y + r * 0.65f),
                                  ImVec2(center.x + r, center.y - r * 0.7f)};
        draw->AddPolyline(points, 3, color, ImDrawFlags_None, (std::max)(1.0f, 1.5f * dpi));
        break;
    }
    case StatusGlyph::None:
        break;
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════
// Construction
// ════════════════════════════════════════════════════════════════════

StatusBarPanel::StatusBarPanel() = default;

// ════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════

void StatusBarPanel::SetConsolePanel(ConsolePanel *panel)
{
    m_console = panel;
}

void StatusBarPanel::SetEngineStatus(const std::string &text, float progress, const std::string &kind)
{
    m_statusText = text;
    m_statusProgress = progress;
    m_statusKind = kind;
}

// ════════════════════════════════════════════════════════════════════
// Render
// ════════════════════════════════════════════════════════════════════

void StatusBarPanel::OnRender(InxGUIContext *ctx)
{
    float x0, y0, dispW, dispH;
    ctx->GetMainViewportBounds(&x0, &y0, &dispW, &dispH);
    if (dispW <= 0.0f || dispH <= 0.0f)
        return;

    if (syncFromEngine)
        syncFromEngine();

    float dpi = ctx->GetDpiScale();
    float height = EditorTheme::STATUS_BAR_BASE_HEIGHT * dpi;
    ctx->PushUIFont(EditorThemeRegistry::Float("EDITOR_SMALL_FONT_SIZE"));

    ctx->SetNextWindowPos(x0, y0 + dispH - height, ImGuiCond_Always, 0.0f, 0.0f);
    ctx->SetNextWindowSize(dispW, height, ImGuiCond_Always);

    // Style overrides (before Begin)
    ImGui::PushStyleColor(ImGuiCol_WindowBg, EditorThemeRegistry::Color("STATUS_BAR_BG"));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                        ImVec2(EditorTheme::STATUS_BAR_WIN_PAD.x * dpi, EditorTheme::STATUS_BAR_WIN_PAD.y * dpi));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        ImVec2(EditorTheme::STATUS_BAR_ITEM_SPC.x * dpi, EditorTheme::STATUS_BAR_ITEM_SPC.y * dpi));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(EditorTheme::STATUS_BAR_FRAME_PAD.x * dpi, EditorTheme::STATUS_BAR_FRAME_PAD.y * dpi));

    constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                                       ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("##InxStatusBar", nullptr, flags)) {
        RenderContent(ctx, dispW);
    }
    ImGui::End();

    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(1);
    ctx->PopFont();
}

// ════════════════════════════════════════════════════════════════════
// Content rendering
// ════════════════════════════════════════════════════════════════════

void StatusBarPanel::RenderContent(InxGUIContext *ctx, float dispW)
{
    const bool statusActive = !m_statusText.empty();
    const float dpi = ctx->GetDpiScale();
    const float barHeight = (EditorTheme::STATUS_BAR_BASE_HEIGHT - 8.0f) * dpi;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float availableWidth = (std::max)(0.0f, ImGui::GetContentRegionAvail().x);
    std::string sessionText;
    float sessionWidth = 0.0f;
    if (availableWidth >= 720.0f * dpi) {
        char frameTime[48];
        const float fps = ImGui::GetIO().Framerate;
        snprintf(frameTime, sizeof(frameTime), "UI %.0f FPS / %.1f ms", fps, fps > 0.0f ? 1000.0f / fps : 0.0f);
        sessionText = m_sessionLabel.empty() ? frameTime : m_sessionLabel + "  |  " + frameTime;
        ctx->PushMonoFont();
        sessionWidth = (std::min)(availableWidth * 0.4f, ImGui::CalcTextSize(sessionText.c_str()).x + 20.0f * dpi);
        ctx->PopFont();
    }
    const float preferredStatusWidth = (std::min)((std::max)(210.0f * dpi, availableWidth * 0.36f), 430.0f * dpi);
    const float statusWidth =
        statusActive ? (std::min)(preferredStatusWidth, (std::max)(0.0f, availableWidth - 80.0f * dpi)) : 0.0f;
    const float consoleWidth = (std::max)(80.0f * dpi, availableWidth - statusWidth - sessionWidth);

    if (m_console) {
        const uint64_t revision = m_console->GetRevision();
        if (revision != m_consoleRevision) {
            m_console->GetStatusBarSnapshot(m_latestMessage, m_latestLevel, m_infoCount, m_warningCount, m_errorCount,
                                            m_latestUid);
            m_consoleRevision = m_console->GetRevision();
        }
    }

    ImGui::SetCursorScreenPos(origin);
    ImGui::InvisibleButton("##StatusBarConsole", ImVec2(consoleWidth, barHeight));
    const bool consoleHovered = ImGui::IsItemHovered();
    if (ImGui::IsItemClicked() && m_console && executeCommand)
        executeCommand("console.open_entry", "pointer", std::to_string(m_latestUid));
    if (InxGUISemantics::IsCaptureEnabled())
        ctx->RecordSemanticItem("status_console", "Console", true, "status.console");

    ImDrawList *draw = ImGui::GetWindowDrawList();
    const float right = origin.x + availableWidth;
    const float consoleRight = origin.x + consoleWidth;
    const float textY = origin.y + (barHeight - ImGui::GetTextLineHeight()) * 0.5f;
    draw->AddLine(ImVec2(origin.x, origin.y - 4.0f * dpi), ImVec2(right, origin.y - 4.0f * dpi),
                  ImGui::ColorConvertFloat4ToU32(EditorTheme::STATUS_BAR_BORDER));
    if (consoleHovered)
        draw->AddRectFilled(origin, ImVec2(consoleRight, origin.y + barHeight),
                            ImGui::ColorConvertFloat4ToU32(EditorThemeRegistry::Color("BTN_SB_HOVERED")));

    // Left: error / warning / info counts (glyph + number), then the latest
    // message. Zero counts stay readable in text-disabled.
    struct SeverityCount
    {
        StatusGlyph glyph;
        int count;
        const char *colorKey;
    };
    const SeverityCount counts[] = {
        {StatusGlyph::Error, m_errorCount, "LOG_ERROR"},
        {StatusGlyph::Warning, m_warningCount, "LOG_WARNING"},
        {StatusGlyph::Info, m_infoCount, "LOG_INFO"},
    };
    const float glyphSize = 9.0f * dpi;
    const float glyphCenterY = origin.y + barHeight * 0.5f;
    const ImVec4 zeroColor = EditorThemeRegistry::Color("TEXT_DISABLED");
    float x = origin.x + 9.0f * dpi;
    for (const SeverityCount &entry : counts) {
        const ImU32 color =
            ImGui::ColorConvertFloat4ToU32(entry.count > 0 ? EditorThemeRegistry::Color(entry.colorKey) : zeroColor);
        DrawStatusGlyph(draw, ImVec2(x + glyphSize * 0.5f, glyphCenterY), glyphSize, entry.glyph, color, dpi);
        x += glyphSize + 4.0f * dpi;
        char number[16];
        snprintf(number, sizeof(number), "%d", entry.count);
        draw->AddText(ImVec2(x, textY), color, number);
        x += ImGui::CalcTextSize(number).x + 12.0f * dpi;
    }

    const float messageX = x + 4.0f * dpi;
    const ImVec4 messageColor = LevelColorForString(m_latestLevel);
    const std::string &summary = m_latestMessage.empty() && m_console ? m_console->GetTitle() : m_latestMessage;
    draw->PushClipRect(ImVec2(messageX, origin.y),
                       ImVec2((std::max)(messageX, consoleRight - 9.0f * dpi), origin.y + barHeight), true);
    draw->AddText(ImVec2(messageX, textY), ImGui::ColorConvertFloat4ToU32(messageColor), summary.c_str());
    draw->PopClipRect();

    if (statusActive)
        RenderEngineStatus(consoleRight, origin.y, statusWidth, barHeight, dpi, m_statusText, m_statusProgress,
                           m_statusKind);

    if (sessionWidth > 0.0f) {
        const float left = right - sessionWidth;
        ctx->PushMonoFont();
        const float textWidth = ImGui::CalcTextSize(sessionText.c_str()).x;
        draw->PushClipRect(ImVec2(left, origin.y), ImVec2(right, origin.y + barHeight), true);
        draw->AddText(ImVec2((std::max)(left + 8.0f * dpi, right - textWidth - 8.0f * dpi), textY),
                      ImGui::ColorConvertFloat4ToU32(zeroColor), sessionText.c_str());
        draw->PopClipRect();
        ctx->PopFont();
    }

    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + barHeight));
    // SetCursorScreenPos alone is not a layout item. Submit a zero-sized item
    // so current ImGui versions can account for the explicit bar boundary.
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
}

void StatusBarPanel::RenderEngineStatus(float x, float y, float width, float height, float dpi, const std::string &text,
                                        float progress, const std::string &kind)
{
    const bool determinate = kind == "progress" && progress >= 0.0f && progress < 1.0f;
    const ImVec4 progressLabelColor = EditorThemeRegistry::Color("STATUS_PROGRESS_LABEL_CLR");
    ImVec4 statusColor = progressLabelColor;
    StatusGlyph statusGlyph = StatusGlyph::None;
    if (kind == "success") {
        statusColor = EditorThemeRegistry::Color("SUCCESS_TEXT");
        statusGlyph = StatusGlyph::Check;
    } else if (kind == "warning") {
        statusColor = EditorThemeRegistry::Color("LOG_WARNING");
        statusGlyph = StatusGlyph::Warning;
    } else if (kind == "error") {
        statusColor = EditorThemeRegistry::Color("LOG_ERROR");
        statusGlyph = StatusGlyph::Error;
    } else if (kind == "activity") {
        const float pulse = 0.68f + 0.22f * static_cast<float>(std::sin(ImGui::GetTime() * 4.0));
        statusColor = EditorThemeRegistry::Color("ROLE_TEXT_DIM");
        statusColor.w = pulse;
        statusGlyph = StatusGlyph::Info;
    }

    ImDrawList *draw = ImGui::GetWindowDrawList();
    draw->AddLine(ImVec2(x, y), ImVec2(x, y + height), ImGui::ColorConvertFloat4ToU32(EditorTheme::STATUS_BAR_BORDER));
    draw->AddRectFilled(ImVec2(x + 1.0f * dpi, y), ImVec2(x + width, y + height),
                        ImGui::ColorConvertFloat4ToU32(EditorTheme::STATUS_ACTIVITY_BG));

    const float textY = y + (height - ImGui::GetTextLineHeight()) * 0.5f;
    float textX = x + 11.0f * dpi;
    if (statusGlyph != StatusGlyph::None) {
        const float glyphSize = 9.0f * dpi;
        DrawStatusGlyph(draw, ImVec2(textX + glyphSize * 0.5f, y + height * 0.5f), glyphSize, statusGlyph,
                        ImGui::ColorConvertFloat4ToU32(statusColor), dpi);
        textX += glyphSize + 7.0f * dpi;
    }

    char percentage[16] = {};
    float percentageWidth = 0.0f;
    if (determinate) {
        snprintf(percentage, sizeof(percentage), "%d%%",
                 static_cast<int>(std::round((std::min)((std::max)(progress, 0.0f), 1.0f) * 100.0f)));
        percentageWidth = ImGui::CalcTextSize(percentage).x;
        draw->AddText(ImVec2(x + width - percentageWidth - 10.0f * dpi, textY),
                      ImGui::ColorConvertFloat4ToU32(progressLabelColor), percentage);
    }

    const float labelRight = x + width - (determinate ? percentageWidth + 20.0f * dpi : 9.0f * dpi);
    draw->PushClipRect(ImVec2(textX, y), ImVec2((std::max)(textX, labelRight), y + height), true);
    draw->AddText(ImVec2(textX, textY), ImGui::ColorConvertFloat4ToU32(statusColor), text.c_str());
    draw->PopClipRect();

    const float lineHeight = 2.0f * dpi;
    const float lineY = y;
    if (determinate) {
        const float clamped = (std::min)((std::max)(progress, 0.0f), 1.0f);
        draw->AddRectFilled(ImVec2(x + 1.0f * dpi, lineY), ImVec2(x + width, lineY + lineHeight),
                            ImGui::ColorConvertFloat4ToU32(EditorThemeRegistry::Color("STATUS_PROGRESS_BG")));
        draw->AddRectFilled(ImVec2(x + 1.0f * dpi, lineY),
                            ImVec2(x + 1.0f * dpi + (width - 1.0f * dpi) * clamped, lineY + lineHeight),
                            ImGui::ColorConvertFloat4ToU32(EditorThemeRegistry::Color("STATUS_PROGRESS_CLR")));
    } else if (kind == "activity") {
        const float trackWidth = (std::max)(40.0f * dpi, width * 0.28f);
        const float travel = (std::max)(1.0f * dpi, width - trackWidth - 2.0f * dpi);
        const float phase = static_cast<float>(std::fmod(ImGui::GetTime() * 92.0 * dpi, static_cast<double>(travel)));
        draw->AddRectFilled(ImVec2(x + 1.0f * dpi + phase, lineY),
                            ImVec2(x + 1.0f * dpi + phase + trackWidth, lineY + lineHeight),
                            ImGui::ColorConvertFloat4ToU32(statusColor));
    }
}

ImVec4 StatusBarPanel::LevelColorForString(const std::string &level) const
{
    if (level == "error")
        return EditorThemeRegistry::Color("LOG_ERROR");
    if (level == "warning")
        return EditorThemeRegistry::Color("LOG_WARNING");
    return EditorThemeRegistry::Color("LOG_INFO");
}

} // namespace infernux
