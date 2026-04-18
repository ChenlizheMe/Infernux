#include "ConsolePanel.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace infernux
{

// ════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ════════════════════════════════════════════════════════════════════

ConsolePanel::ConsolePanel() : EditorPanel("Console", "console")
{
    // Subscribe to INXLOG — receives ALL C++ log messages.
    m_sinkId = InxLog::GetInstance().AddSink(
        [this](LogLevel level, const char *file, int line, const std::string &message, bool internalOnly) {
            OnLogMessage(level, file, line, message, internalOnly);
        });
}

ConsolePanel::~ConsolePanel()
{
    InxLog::GetInstance().RemoveSink(m_sinkId);
}

// ════════════════════════════════════════════════════════════════════
// INXLOG sink callback (may be called from ANY thread)
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::OnLogMessage(LogLevel level, const char *file, int line, const std::string &message,
                                bool internalOnly)
{
    if (internalOnly || IsInternalNoise(message))
        return;

    LogEntry entry;
    entry.level = level;
    entry.message = message;
    entry.sourceFile = file ? file : "";
    entry.sourceLine = line;
    entry.timestamp = CurrentTimestamp();

    // Cache first line for display
    auto nl = entry.message.find('\n');
    entry.firstLine = (nl != std::string::npos) ? entry.message.substr(0, nl) : entry.message;

    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        m_pendingLogs.push_back(std::move(entry));
    }
}

// ════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::LogFromPython(LogLevel level, const std::string &message, const std::string &stackTrace,
                                 const std::string &sourceFile, int sourceLine)
{
    LogEntry entry;
    entry.level = level;
    entry.message = message;
    entry.stackTrace = stackTrace;
    entry.sourceFile = sourceFile;
    entry.sourceLine = sourceLine;
    entry.timestamp = CurrentTimestamp();

    auto nl = entry.message.find('\n');
    entry.firstLine = (nl != std::string::npos) ? entry.message.substr(0, nl) : entry.message;

    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        m_pendingLogs.push_back(std::move(entry));
    }
}

void ConsolePanel::Clear()
{
    std::lock_guard<std::mutex> lock(m_logMutex);
    m_logs.clear();
    m_pendingLogs.clear();
    m_selectedIndex = -1;
    m_nextUid = 1;
    m_cacheDirty = true;
    m_cachedInfoCount = 0;
    m_cachedWarnCount = 0;
    m_cachedErrorCount = 0;
    m_visible.clear();
}

int ConsolePanel::GetInfoCount() const
{
    int infoCount = 0;
    int warnCount = 0;
    int errorCount = 0;
    GetCountSnapshot(infoCount, warnCount, errorCount);
    return infoCount;
}

int ConsolePanel::GetWarningCount() const
{
    int infoCount = 0;
    int warnCount = 0;
    int errorCount = 0;
    GetCountSnapshot(infoCount, warnCount, errorCount);
    return warnCount;
}

int ConsolePanel::GetErrorCount() const
{
    int infoCount = 0;
    int warnCount = 0;
    int errorCount = 0;
    GetCountSnapshot(infoCount, warnCount, errorCount);
    return errorCount;
}

void ConsolePanel::SelectLatestEntry()
{
    m_isOpen = true;
    m_selectedIndex = -2; // sentinel: "select last" — resolved in RenderBody
    ImGui::SetWindowFocus((m_title + "###" + m_windowId).c_str());
}

void ConsolePanel::GetLastVisibleForStatusBar(std::string &outMsg, std::string &outLevel)
{
    outMsg.clear();
    outLevel = "info";
    EnsureCache();
    if (m_visible.empty())
        return;
    const VisibleEntry &ve = m_visible.back();
    if (ve.logIndex >= m_logs.size())
        return;
    const LogEntry &log = m_logs[ve.logIndex];
    outMsg = log.firstLine;
    switch (log.level) {
    case LOG_WARN:
        outLevel = "warning";
        break;
    case LOG_ERROR:
    case LOG_FATAL:
        outLevel = "error";
        break;
    default:
        outLevel = "info";
        break;
    }
}

// ════════════════════════════════════════════════════════════════════
// Render
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::OnRenderContent(InxGUIContext *ctx)
{
    FlushPendingLogs();
    EnsureCache();
    RenderToolbar(ctx);
    ImGui::Separator();
    RenderBody(ctx);
}

// ════════════════════════════════════════════════════════════════════
// Flush pending logs (main thread only)
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::FlushPendingLogs()
{
    std::vector<LogEntry> incoming;
    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        if (m_pendingLogs.empty())
            return;
        incoming.swap(m_pendingLogs);
    }

    for (auto &entry : incoming) {
        entry.uid = m_nextUid++;
        m_logs.push_back(std::move(entry));
    }

    while (m_logs.size() > MAX_LOGS)
        m_logs.pop_front();

    m_cacheDirty = true;
    if (!m_userScrolledUp)
        m_scrollToBottom = true;
}

void ConsolePanel::GetCountSnapshot(int &infoCount, int &warnCount, int &errorCount) const
{
    infoCount = 0;
    warnCount = 0;
    errorCount = 0;

    auto accumulate = [&infoCount, &warnCount, &errorCount](LogLevel level) {
        switch (level) {
        case LOG_WARN:
            ++warnCount;
            break;
        case LOG_ERROR:
        case LOG_FATAL:
            ++errorCount;
            break;
        default:
            ++infoCount;
            break;
        }
    };

    std::lock_guard<std::mutex> lock(m_logMutex);
    for (const auto &log : m_logs)
        accumulate(log.level);
    for (const auto &log : m_pendingLogs)
        accumulate(log.level);
}

// ════════════════════════════════════════════════════════════════════
// Filter cache
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::DetectFilterChange()
{
    bool changed = (showInfo != m_prevShowInfo || showWarnings != m_prevShowWarnings ||
                    showErrors != m_prevShowErrors || collapse != m_prevCollapse);
    if (changed) {
        m_prevShowInfo = showInfo;
        m_prevShowWarnings = showWarnings;
        m_prevShowErrors = showErrors;
        m_prevCollapse = collapse;
        m_filterDirty = true;
    }
}

void ConsolePanel::EnsureCache()
{
    DetectFilterChange();
    if (!m_cacheDirty && !m_filterDirty)
        return;

    // Rebuild counts
    int ic = 0, wc = 0, ec = 0;
    GetCountSnapshot(ic, wc, ec);
    m_cachedInfoCount = ic;
    m_cachedWarnCount = wc;
    m_cachedErrorCount = ec;

    // Remember selected entry UID so we can restore selection after rebuild
    uint64_t selectedUid = 0;
    if (m_selectedIndex >= 0 && m_selectedIndex < static_cast<int>(m_visible.size()))
        selectedUid = m_visible[m_selectedIndex].uid;

    // Rebuild visible list
    m_visible.clear();
    // collapse_map: key = (level, message) → index in m_visible
    std::unordered_map<std::string, size_t> collapseMap;

    for (size_t i = 0; i < m_logs.size(); ++i) {
        const auto &log = m_logs[i];

        // Apply filters
        if (log.level <= LOG_INFO && !showInfo)
            continue;
        if (log.level == LOG_WARN && !showWarnings)
            continue;
        if ((log.level == LOG_ERROR || log.level == LOG_FATAL) && !showErrors)
            continue;

        if (collapse) {
            // Build collapse key: level + message
            std::string key = std::to_string(static_cast<int>(log.level)) + "|" + log.message;
            auto it = collapseMap.find(key);
            if (it != collapseMap.end()) {
                m_visible[it->second].count++;
                m_visible[it->second].uid = log.uid;
                continue;
            }
            collapseMap[key] = m_visible.size();
        }

        VisibleEntry ve;
        ve.logIndex = i;
        ve.count = 1;
        ve.uid = log.uid;
        m_visible.push_back(ve);
    }

    // Restore selection by UID — prevents click/refresh race
    if (selectedUid > 0) {
        m_selectedIndex = -1;
        for (int idx = 0; idx < static_cast<int>(m_visible.size()); ++idx) {
            if (m_visible[idx].uid == selectedUid) {
                m_selectedIndex = idx;
                break;
            }
        }
    }

    m_cacheDirty = false;
    m_filterDirty = false;
}

// ════════════════════════════════════════════════════════════════════
// Toolbar
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::RenderToolbar(InxGUIContext * /*ctx*/)
{
    // Push console toolbar compact spacing (3 style vars)
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                        ImVec2(EditorTheme::CONSOLE_FRAME_PAD_X, EditorTheme::CONSOLE_FRAME_PAD_Y));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                        ImVec2(EditorTheme::CONSOLE_ITEM_SPC_X, EditorTheme::CONSOLE_ITEM_SPC_Y));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, EditorTheme::TOOLBAR_FRAME_BRD);

    if (ImGui::Button("Clear"))
        Clear();

    ImGui::SameLine();
    ImGui::Checkbox("Collapse", &collapse);

    ImGui::SameLine();
    ImGui::Checkbox("Clear on Play", &clearOnPlay);

    ImGui::SameLine();
    ImGui::Checkbox("Error Pause", &errorPause);

    // Right-aligned filter toggles
    float winW = ImGui::GetWindowWidth();
    float filterW = 240.0f;
    float leftW = 350.0f;
    float filterX = (std::max)(winW - filterW, leftW);
    if (winW >= leftW + filterW)
        ImGui::SameLine(filterX);

    // Info filter
    {
        char label[64];
        if (m_cachedInfoCount > 0)
            snprintf(label, sizeof(label), "Log %d###ConsoleFilterInfo", m_cachedInfoCount);
        else
            snprintf(label, sizeof(label), "Log###ConsoleFilterInfo");
        ImGui::Checkbox(label, &showInfo);
    }

    // Warning filter (yellow text)
    ImGui::SameLine();
    {
        ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::LOG_WARNING);
        char label[64];
        if (m_cachedWarnCount > 0)
            snprintf(label, sizeof(label), "Warn %d###ConsoleFilterWarn", m_cachedWarnCount);
        else
            snprintf(label, sizeof(label), "Warn###ConsoleFilterWarn");
        ImGui::Checkbox(label, &showWarnings);
        ImGui::PopStyleColor();
    }

    // Error filter (red text)
    ImGui::SameLine();
    {
        ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::LOG_ERROR);
        char label[64];
        if (m_cachedErrorCount > 0)
            snprintf(label, sizeof(label), "Error %d###ConsoleFilterError", m_cachedErrorCount);
        else
            snprintf(label, sizeof(label), "Error###ConsoleFilterError");
        ImGui::Checkbox(label, &showErrors);
        ImGui::PopStyleColor();
    }

    ImGui::PopStyleVar(3);
}

// ════════════════════════════════════════════════════════════════════
// Body (log list + detail pane)
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::RenderBody(InxGUIContext * /*ctx*/)
{
    float availH = ImGui::GetContentRegionAvail().y;

    // Sentinel -2: "select last visible entry" (from SelectLatestEntry)
    if (m_selectedIndex == -2 && !m_visible.empty()) {
        m_selectedIndex = static_cast<int>(m_visible.size()) - 1;
        m_scrollToBottom = true;
    } else if (m_selectedIndex == -2) {
        m_selectedIndex = -1;
    }

    bool hasDetail = (m_selectedIndex >= 0 && m_selectedIndex < static_cast<int>(m_visible.size()));

    // Clamp selection
    if (m_selectedIndex >= static_cast<int>(m_visible.size())) {
        m_selectedIndex = -1;
        hasDetail = false;
    }

    float splitterH = 3.0f;
    float listH;
    if (hasDetail) {
        m_detailHeight = (std::max)(40.0f, (std::min)(m_detailHeight, availH - 60.0f));
        listH = (std::max)(availH - m_detailHeight - splitterH, 40.0f);
    } else {
        listH = 0.0f; // 0 = use remaining space
    }

    int total = static_cast<int>(m_visible.size());
    float rowH = m_rowHeight;

    // ── Log list (virtual-scrolled) ──
    ImGui::PushStyleColor(ImGuiCol_Border, EditorTheme::BORDER_TRANSPARENT);
    if (ImGui::BeginChild("##ConsoleLogList", ImVec2(0, listH), ImGuiChildFlags_Borders)) {
        float scrollY = ImGui::GetScrollY();
        float viewportH = ImGui::GetContentRegionAvail().y;
        int firstVis = (rowH > 0.0f) ? (std::max)(static_cast<int>(scrollY / rowH), 0) : 0;
        int lastVis = (total > 0) ? (std::min)(firstVis + static_cast<int>(viewportH / rowH) + 2, total - 1) : -1;

        // Top spacer
        if (firstVis > 0) {
            float w = ImGui::GetContentRegionAvail().x;
            ImGui::Dummy(ImVec2(w, firstVis * rowH));
        }

        // Render visible rows
        for (int idx = (std::max)(firstVis, 0); idx <= lastVis; ++idx) {
            if (!m_rowHeightMeasured) {
                float y0 = ImGui::GetCursorPosY();
                RenderRow(idx, m_visible[idx]);
                float y1 = ImGui::GetCursorPosY();
                float measured = y1 - y0;
                if (measured > 1.0f) {
                    m_rowHeight = measured;
                    rowH = measured;
                    m_rowHeightMeasured = true;
                }
            } else {
                RenderRow(idx, m_visible[idx]);
            }
        }

        // Bottom spacer
        int remaining = total - (lastVis + 1);
        if (remaining > 0) {
            float w = ImGui::GetContentRegionAvail().x;
            ImGui::Dummy(ImVec2(w, remaining * rowH));
        }

        // Ctrl+C: copy selected entry
        if (m_selectedIndex >= 0 && m_selectedIndex < total) {
            if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_C)) {
                const auto &ve = m_visible[m_selectedIndex];
                const auto &log = m_logs[ve.logIndex];
                std::string copyText = log.message;
                if (!log.stackTrace.empty())
                    copyText += "\n" + log.stackTrace;
                ImGui::SetClipboardText(copyText.c_str());
            }
        }

        // Smart auto-scroll
        scrollY = ImGui::GetScrollY();
        float scrollMax = ImGui::GetScrollMaxY();
        if (scrollMax > 0) {
            bool atBottom = (scrollMax - scrollY) < 20.0f;
            m_userScrolledUp = !atBottom;
        }

        if (m_scrollToBottom && !m_visible.empty()) {
            ImGui::SetScrollHereY(1.0f);
            m_scrollToBottom = false;
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(); // Border

    // ── Draggable splitter ──
    if (hasDetail) {
        float availW = ImGui::GetContentRegionAvail().x;
        ImGui::PushStyleColor(ImGuiCol_Button, EditorTheme::BTN_GHOST);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, EditorTheme::SPLITTER_HOVER);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, EditorTheme::SPLITTER_ACTIVE);
        ImGui::InvisibleButton("##ConsoleSplitter", ImVec2(availW, splitterH));
        if (ImGui::IsItemActive()) {
            float dy = ImGui::GetMouseDragDelta(0).y;
            if (std::abs(dy) > 0.5f) {
                m_detailHeight = (std::max)(40.0f, m_detailHeight - dy);
                ImGui::ResetMouseDragDelta(0);
            }
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        } else if (ImGui::IsItemHovered()) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        }
        ImGui::PopStyleColor(3);
    }

    // ── Detail pane ──
    if (hasDetail && m_selectedIndex >= 0 && m_selectedIndex < static_cast<int>(m_visible.size())) {
        const auto &ve = m_visible[m_selectedIndex];
        const auto &log = m_logs[ve.logIndex];
        const ImVec4 &clr = LevelColor(log.level);

        std::string detailText = "[" + log.timestamp + "]  " + log.message;
        if (!log.stackTrace.empty())
            detailText += "\n\n" + log.stackTrace;

        ImGui::PushStyleColor(ImGuiCol_Text, clr);
        ImGui::PushStyleColor(ImGuiCol_WindowBg, EditorTheme::ROW_NONE);
        ImGui::PushStyleColor(ImGuiCol_FrameBg, EditorTheme::ROW_NONE);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);

        // Read-only multiline input — supports text selection & Ctrl+C
        ImGui::InputTextMultiline("##ConsoleDetail", const_cast<char *>(detailText.c_str()), detailText.size() + 1,
                                  ImVec2(-1, -1), ImGuiInputTextFlags_ReadOnly);

        ImGui::PopStyleVar();
        ImGui::PopStyleColor(3);
    }
}

// ════════════════════════════════════════════════════════════════════
// Single row
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::RenderRow(int visIdx, const VisibleEntry &ve)
{
    const auto &log = m_logs[ve.logIndex];
    const ImVec4 &clr = LevelColor(log.level);
    bool isSel = (visIdx == m_selectedIndex);

    // Row background
    if (isSel)
        ImGui::PushStyleColor(ImGuiCol_Header, EditorTheme::SELECTION_BG);
    else if (visIdx % 2 == 1)
        ImGui::PushStyleColor(ImGuiCol_Header, EditorTheme::ROW_ALT);
    else
        ImGui::PushStyleColor(ImGuiCol_Header, EditorTheme::ROW_NONE);

    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, EditorTheme::SELECTION_BG);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, EditorTheme::SELECTION_BG);
    ImGui::PushStyleColor(ImGuiCol_Text, clr);

    // Unique ID to avoid ImGui ID conflicts
    char label[512];
    snprintf(label, sizeof(label), "%s##clog_%llu_%d", log.firstLine.c_str(), static_cast<unsigned long long>(ve.uid),
             visIdx);

    if (ImGui::Selectable(label, isSel, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
        m_selectedIndex = visIdx;
        // Double-click: navigate to source
        if (ImGui::IsMouseDoubleClicked(0) && onDoubleClickEntry && !log.sourceFile.empty()) {
            onDoubleClickEntry(log.sourceFile, log.sourceLine);
        }
    }

    // Collapse count badge
    if (ve.count > 1) {
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 20.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, EditorTheme::LOG_BADGE);
        ImGui::Text("%d", ve.count);
        ImGui::PopStyleColor();
    }

    ImGui::PopStyleColor(4);
}

// ════════════════════════════════════════════════════════════════════
// Utilities
// ════════════════════════════════════════════════════════════════════

const ImVec4 &ConsolePanel::LevelColor(LogLevel lv) const
{
    switch (lv) {
    case LOG_ERROR:
    case LOG_FATAL:
        return EditorTheme::LOG_ERROR;
    case LOG_WARN:
        return EditorTheme::LOG_WARNING;
    default:
        return EditorTheme::LOG_INFO;
    }
}

std::string ConsolePanel::CurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(ms.count()));
    return buf;
}

bool ConsolePanel::IsInternalNoise(const std::string &msg)
{
    if (msg.find("DEAR IMGUI") != std::string::npos)
        return true;
    if (msg.find("PushID") != std::string::npos)
        return true;
    if (msg.find("conflicting ID") != std::string::npos)
        return true;
    return false;
}

} // namespace infernux
