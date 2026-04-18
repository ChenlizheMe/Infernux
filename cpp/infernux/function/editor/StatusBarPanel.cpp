#include "StatusBarPanel.h"
#include "ConsolePanel.h"

#include <algorithm>
#include <cstring>

namespace infernux
{

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

void StatusBarPanel::SetLatestMessage(const std::string &message, const std::string &level)
{
    // Keep only the first line
    auto nl = message.find('\n');
    m_latestMsg = (nl != std::string::npos) ? message.substr(0, nl) : message;
    m_latestLevel = level;
}

void StatusBarPanel::ClearCounts()
{
    m_warnCount = 0;
    m_errorCount = 0;
    m_latestMsg.clear();
    m_latestLevel = "info";
}

void StatusBarPanel::SetEngineStatus(const std::string &text, float progress)
{
    m_statusText = text;
    m_statusProgress = progress;
}

void StatusBarPanel::IncrementWarnCount()
{
    ++m_warnCount;
}

void StatusBarPanel::IncrementErrorCount()
{
    ++m_errorCount;
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

    float dpi = ctx->GetDpiScale();
    float height = EditorTheme::STATUS_BAR_BASE_HEIGHT * dpi;

    ctx->SetNextWindowPos(x0, y0 + dispH - height, ImGuiCond_Always, 0.0f, 0.0f);
    ctx->SetNextWindowSize(dispW, height, ImGuiCond_Always);

    // Style overrides (before Begin)
    ImGui::PushStyleColor(ImGuiCol_WindowBg, EditorTheme::STATUS_BAR_BG);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, EditorTheme::STATUS_BAR_WIN_PAD);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, EditorTheme::STATUS_BAR_ITEM_SPC);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, EditorTheme::STATUS_BAR_FRAME_PAD);

    constexpr ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
                                       ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
                                       ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings;

    if (ImGui::Begin("##InxStatusBar", nullptr, flags)) {
        RenderContent(ctx, dispW);
    }
    ImGui::End();

    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(1);
}

// ════════════════════════════════════════════════════════════════════
// Content rendering
// ════════════════════════════════════════════════════════════════════

void StatusBarPanel::RenderContent(InxGUIContext *ctx, float dispW)
{
    bool statusActive = !m_statusText.empty();
    float leftZoneW = statusActive ? dispW * (1.0f - EditorTheme::STATUS_PROGRESS_FRACTION) : dispW;
    float dpi = ctx->GetDpiScale();
    float barH = EditorTheme::STATUS_BAR_BASE_HEIGHT * dpi - 8.0f;

    // ── Left zone: clickable area → opens console ────────────────
    float clickW = (std::max)(leftZoneW - 8.0f, 100.0f);

    ImGui::PushStyleColor(ImGuiCol_Button, EditorTheme::BTN_GHOST);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, EditorTheme::BTN_SB_HOVERED);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, EditorTheme::BTN_SB_ACTIVE);

    if (ImGui::InvisibleButton("##StatusBarClick", ImVec2(clickW, barH))) {
        if (m_console)
            m_console->SelectLatestEntry();
    }
    ImGui::PopStyleColor(3);

    // Overlay text on top of the invisible button
    ImGui::SameLine(6.0f);

    // ── Level icon + message ─────────────────────────────────────
    // Prefer the native console's last *visible* line (matches Clear / filters).
    std::string lineMsg = m_latestMsg;
    std::string lineLevel = m_latestLevel;
    if (m_console) {
        m_console->GetLastVisibleForStatusBar(lineMsg, lineLevel);
    }

    const ImVec4 &clr = LevelColorForString(lineLevel);
    ImGui::PushStyleColor(ImGuiCol_Text, clr);

    std::string icon;
    if (lineLevel == "error")
        icon = std::string(EditorTheme::ICON_ERROR) + " ";
    else if (lineLevel == "warning")
        icon = std::string(EditorTheme::ICON_WARNING) + " ";

    // Truncate
    std::string msg = lineMsg;
    int maxChars = (std::max)(10, static_cast<int>((leftZoneW - 160.0f) / 8.0f));
    if (static_cast<int>(msg.size()) > maxChars) {
        msg = msg.substr(0, maxChars - 1) + "\xe2\x80\xa6"; // …
    }

    ImGui::TextUnformatted((icon + msg).c_str());
    ImGui::PopStyleColor(1);

    // ── Right counters: [W] N  [E] N ────────────────────────────
    float counterX = leftZoneW - 130.0f;
    if (counterX > 0.0f) {
        ImGui::SameLine(counterX);

        // Warn count
        if (m_warnCount > 0)
            ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::LOG_WARNING);
        else
            ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::LOG_DIM);

        char warnBuf[64];
        snprintf(warnBuf, sizeof(warnBuf), "%s %d", EditorTheme::ICON_WARNING, m_warnCount);
        ImGui::TextUnformatted(warnBuf);
        ImGui::PopStyleColor(1);

        ImGui::SameLine(0.0f, 12.0f);

        // Error count
        if (m_errorCount > 0)
            ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::LOG_ERROR);
        else
            ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::LOG_DIM);

        char errBuf[64];
        snprintf(errBuf, sizeof(errBuf), "%s %d", EditorTheme::ICON_ERROR, m_errorCount);
        ImGui::TextUnformatted(errBuf);
        ImGui::PopStyleColor(1);
    }

    // ── Right zone: engine status + progress ─────────────────────
    if (statusActive) {
        RenderEngineStatus(ctx, dispW, leftZoneW, m_statusText, m_statusProgress);
    }
}

void StatusBarPanel::RenderEngineStatus(InxGUIContext *ctx, float dispW, float leftZoneW, const std::string &text,
                                        float progress)
{
    ImGui::SameLine(leftZoneW + 8.0f);
    float zoneW = dispW - leftZoneW - 16.0f;

    // Determinate progress bar
    if (progress >= 0.0f) {
        float barW = (std::min)(zoneW * 0.4f, 80.0f);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, EditorTheme::STATUS_PROGRESS_BG);
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, EditorTheme::STATUS_PROGRESS_CLR);
        ImGui::SetNextItemWidth(barW);
        float clamped = (std::min)((std::max)(progress, 0.0f), 1.0f);
        ImGui::ProgressBar(clamped, ImVec2(barW, 0.0f), "");
        ImGui::PopStyleColor(2);
        ImGui::SameLine(0.0f, 6.0f);
    }

    // Truncate text
    float remainingW = zoneW - (progress >= 0.0f ? 90.0f : 0.0f);
    int maxChars = (std::max)(6, static_cast<int>(remainingW / 8.0f));
    std::string label = text;
    if (static_cast<int>(label.size()) > maxChars) {
        label = label.substr(0, maxChars - 1) + "\xe2\x80\xa6"; // …
    }

    ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::STATUS_PROGRESS_LABEL_CLR);
    ImGui::TextUnformatted(label.c_str());
    ImGui::PopStyleColor(1);
}

const ImVec4 &StatusBarPanel::LevelColor() const
{
    return LevelColorForString(m_latestLevel);
}

const ImVec4 &StatusBarPanel::LevelColorForString(const std::string &level) const
{
    if (level == "error")
        return EditorTheme::LOG_ERROR;
    if (level == "warning")
        return EditorTheme::LOG_WARNING;
    return EditorTheme::LOG_INFO;
}

} // namespace infernux
