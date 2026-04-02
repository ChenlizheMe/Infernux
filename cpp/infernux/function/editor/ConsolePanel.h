#pragma once

#include "EditorPanel.h"
#include "EditorTheme.h"
#include <core/log/InxLog.h>

#include <imgui.h>

#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernux
{

/// C++ native Console panel — replaces the Python ConsolePanel.
/// Subscribes to INXLOG sinks to receive all engine log messages directly.
/// Also exposes LogFromPython() so Python Debug.log() messages appear here.
class ConsolePanel : public EditorPanel
{
  public:
    ConsolePanel();
    ~ConsolePanel() override;

    // ── Public API (called from pybind11 or other C++ systems) ──

    /// Log a message originating from Python's Debug.log() system.
    void LogFromPython(LogLevel level, const std::string &message, const std::string &stackTrace = "",
                       const std::string &sourceFile = "", int sourceLine = 0);

    /// Clear all log entries.
    void Clear();

    /// Query counts for status bar integration.
    int GetInfoCount() const;
    int GetWarningCount() const;
    int GetErrorCount() const;

    /// Select the last visible entry and request window focus.
    /// Called from status bar click.
    void SelectLatestEntry();

    /// Filter state — exposed for pybind11 property access.
    bool showInfo = true;
    bool showWarnings = true;
    bool showErrors = true;
    bool collapse = false;
    bool clearOnPlay = true;
    bool errorPause = false;
    bool autoScroll = true;

  protected:
    void OnRenderContent(InxGUIContext *ctx) override;

  private:
    // ── Internal log entry ──
    struct LogEntry
    {
        std::string message;
        std::string firstLine; // cached first line for display
        std::string timestamp;
        std::string stackTrace;
        std::string sourceFile;
        int sourceLine = 0;
        LogLevel level = LOG_INFO;
        uint64_t uid = 0;
    };

    // ── Visible entry after filtering/collapsing ──
    struct VisibleEntry
    {
        size_t logIndex; // index into m_logs
        int count;       // collapse count
        uint64_t uid;
    };

    // ── INXLOG sink ──
    size_t m_sinkId = 0;
    void OnLogMessage(LogLevel level, const char *file, int line, const std::string &message, bool internalOnly);

    // ── Log storage ──
    static constexpr size_t MAX_LOGS = 2000;
    std::deque<LogEntry> m_logs;
    uint64_t m_nextUid = 1;
    mutable std::mutex m_logMutex; // protects m_logs + m_pendingLogs
    std::vector<LogEntry> m_pendingLogs; // accumulated off-main-thread, flushed in OnRender

    // ── Filter cache ──
    bool m_cacheDirty = true;
    bool m_filterDirty = true;
    bool m_prevShowInfo = true;
    bool m_prevShowWarnings = true;
    bool m_prevShowErrors = true;
    bool m_prevCollapse = false;
    std::vector<VisibleEntry> m_visible;
    int m_cachedInfoCount = 0;
    int m_cachedWarnCount = 0;
    int m_cachedErrorCount = 0;

    // ── Selection & scroll ──
    int m_selectedIndex = -1;
    bool m_userScrolledUp = false;
    bool m_scrollToBottom = false;
    float m_rowHeight = 22.0f;
    bool m_rowHeightMeasured = false;
    float m_detailHeight = 90.0f;

    // ── Helpers ──
    void FlushPendingLogs();
    void GetCountSnapshot(int &infoCount, int &warnCount, int &errorCount) const;
    void EnsureCache();
    void DetectFilterChange();
    void RenderToolbar(InxGUIContext *ctx);
    void RenderBody(InxGUIContext *ctx);
    void RenderRow(int visIdx, const VisibleEntry &ve);
    const ImVec4 &LevelColor(LogLevel lv) const;
    static std::string CurrentTimestamp();
    static bool IsInternalNoise(const std::string &msg);
};

} // namespace infernux
