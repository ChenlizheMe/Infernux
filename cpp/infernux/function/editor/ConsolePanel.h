#pragma once

#include "EditorPanel.h"
#include "EditorTheme.h"
#include "interaction/EditorSearchModel.h"
#include <core/log/InxLog.h>
#include <function/scene/ObjectHandle.h>

#include <imgui.h>

#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infernux
{

/// C++ native Console panel — replaces the Python ConsolePanel.
/// Subscribes to INXLOG sinks to receive all engine log messages directly.
/// Also exposes LogFromPython() so Python Debug.log() messages appear here.
class ConsolePanel : public EditorPanel
{
  public:
    explicit ConsolePanel(const std::string &projectRoot = "");
    ~ConsolePanel() override;
    std::function<std::string(const std::string &)> translate = [](const std::string &key) { return key; };

    // ── Public API (called from pybind11 or other C++ systems) ──

    /// Log a message originating from Python's Debug.log() system.
    void LogFromPython(LogLevel level, const std::string &message, const std::string &stackTrace = "",
                       const std::string &sourceFile = "", int sourceLine = 0, const ObjectHandle &contextHandle = {});
    void SetDiagnostic(const std::string &owner, const std::string &key, LogLevel level, const std::string &message,
                       const std::string &stackTrace = "", const std::string &sourceFile = "", int sourceLine = 0,
                       const ObjectHandle &contextHandle = {}, int occurrences = 1);
    bool ResolveDiagnostic(const std::string &owner, const std::string &key);

    [[nodiscard]] const std::string &GetProjectRoot() const
    {
        return m_projectRoot;
    }
    uint64_t GetEntryContextObjectId(uint64_t uid);

    /// Clear all log entries.
    void Clear();

    /// Remove diagnostics owned by one source file without disturbing other
    /// Console history. Returns the number of removed entries.
    size_t RemoveEntriesFromSource(const std::string &sourceFile, const std::string &messagePrefix = "");

    /// Query counts for status bar integration.
    int GetInfoCount() const;
    int GetWarningCount() const;
    int GetErrorCount() const;

    std::unordered_map<std::string, double> ConsumeSubTimings() override;

    /// Select the latest entry and request window focus.
    /// Called from status bar click.
    void SelectLatestEntry();
    void SelectEntry(uint64_t uid);
    /// Project the authoritative editor selection without publishing a second
    /// selection event back to Python.
    void SetSelectionSnapshot(uint64_t uid);

    /// Authoritative status-bar snapshot. The latest entry and counts do not
    /// depend on the Console window's current filters.
    void GetStatusBarSnapshot(std::string &outMsg, std::string &outLevel, int &outInfoCount, int &outWarnCount,
                              int &outErrorCount, uint64_t &outUid);
    [[nodiscard]] uint64_t GetRevision() const noexcept;
    [[nodiscard]] uint64_t GetSelectedUid() const noexcept;
    [[nodiscard]] bool HasSelectedEntry() const noexcept;
    [[nodiscard]] std::vector<uint64_t> GetSelectedUids() const;

    /// Snapshot the entries currently visible in the native Console view.
    /// This is intentionally a bounded, filtered view so external tools do
    /// not copy the full log buffer every frame.
    struct VisibleLogSnapshot
    {
        std::string message;
        std::string timestamp;
        std::string stackTrace;
        std::string sourceFile;
        int sourceLine = 0;
        LogLevel level = LOG_INFO;
        uint64_t uid = 0;
        uint64_t latestUid = 0;
        int count = 1;
        std::string diagnosticOwner;
        std::string diagnosticKey;
        int occurrences = 1;
    };

    std::vector<VisibleLogSnapshot> GetVisibleLogSnapshot(size_t limit);
    bool CopySelectedEntry();
    void RenderSelectionInspector(InxGUIContext *ctx);
    [[nodiscard]] bool HasViewOption(const std::string &option) const noexcept;
    [[nodiscard]] bool GetViewOption(const std::string &option) const noexcept;
    void SetViewOption(const std::string &option, bool enabled);
    [[nodiscard]] std::string GetSearchQuery() const;
    void SetSearchQuery(const std::string &query);
    void RequestSearchFocus()
    {
        m_focusSearchNextFrame = true;
    }

    std::function<void()> onErrorPause;
    std::function<void()> onRequestFocus;
    std::function<void(uint64_t, bool)> onSelectionChanged;

    /// Filter state — exposed for pybind11 property access.
    bool showInfo = true;
    bool showWarnings = true;
    bool showErrors = true;
    bool collapse = false;
    bool clearOnPlay = true;
    bool errorPause = false;
    bool autoScroll = true;

  protected:
    void PreRender(InxGUIContext *ctx) override;
    void OnRenderContent(InxGUIContext *ctx) override;

  private:
    // ── Internal log entry ──
    struct LogEntry
    {
        std::string message;
        std::string firstLine; // cached first line for display
        std::string displayLine;
        std::string sourceLabel;
        std::string timestamp;
        std::string stackTrace;
        std::string sourceFile;
        int sourceLine = 0;
        LogLevel level = LOG_INFO;
        uint64_t uid = 0;
        ObjectHandle contextHandle;
        std::string diagnosticOwner;
        std::string diagnosticKey;
        int occurrences = 1;
    };

    // ── Visible entry after filtering/collapsing ──
    struct VisibleEntry
    {
        size_t logIndex; // index into m_logs
        int count;       // collapse count
        uint64_t uid;    // stable UID of the first entry in this group
        uint64_t latestUid;
    };

    // ── INXLOG sink ──
    size_t m_sinkId = 0;
    DiagnosticSubscription m_diagnosticSubscription;
    const std::string m_projectRoot;
    void OnLogMessage(LogLevel level, const char *file, int line, const std::string &message, bool internalOnly);
    void CacheDisplayFields(LogEntry &entry) const;
    void UpdateCounts(LogLevel level, int delta);
    void ApplyDiagnosticChanges(std::unordered_map<uint64_t, std::optional<LogEntry>> &changes,
                                std::vector<LogEntry> &incoming);

    // ── Log storage ──
    static constexpr size_t MAX_LOGS = 100000u;
    std::deque<LogEntry> m_logs;
    uint64_t m_nextUid = 1;
    mutable std::mutex m_logMutex;       // protects m_logs + m_pendingLogs
    std::vector<LogEntry> m_pendingLogs; // accumulated off-main-thread, flushed in OnRender
    std::unordered_map<std::string, LogEntry> m_diagnostics;
    std::unordered_map<uint64_t, std::optional<LogEntry>> m_pendingDiagnosticChanges;
    std::atomic<int> m_infoCount{0};
    std::atomic<int> m_warnCount{0};
    std::atomic<int> m_errorCount{0};
    std::atomic<uint64_t> m_revision{1};

    // ── Filter cache ──
    bool m_cacheDirty = true;
    bool m_filterDirty = true;
    bool m_prevShowInfo = true;
    bool m_prevShowWarnings = true;
    bool m_prevShowErrors = true;
    bool m_prevCollapse = false;
    EditorSearchModel m_searchModel;
    std::vector<VisibleEntry> m_visible;
    std::unordered_map<std::string, size_t> m_collapseLookup;
    int m_cachedInfoCount = 0;
    int m_cachedWarnCount = 0;
    int m_cachedErrorCount = 0;

    // ── Selection projection & scroll ──
    // The global diagnostic selection projects into the Inspector.
    uint64_t m_selectedUid = 0;
    std::vector<uint64_t> m_selectedUids;
    std::unordered_set<uint64_t> m_selectedUidLookup;
    uint64_t m_requestedUid = 0;
    bool m_followTail = true;
    bool m_scrollToBottom = false;
    bool m_resetScrollToTop = false;
    std::array<char, 256> m_search{};
    std::string m_searchEditStart;
    bool m_focusSearchNextFrame = false;
    float m_rowHeight = 48.0f;
    bool m_rowHeightMeasured = false;
    float m_lastDpiScale = 0.0f;
    std::vector<uint64_t> m_inspectorUids;
    uint64_t m_inspectorPrimaryUid = 0;
    bool m_inspectorDirty = false;
    struct InspectorStackLine
    {
        std::string text;
        std::string sourceFile;
        int sourceLine = 0;
    };
    struct InspectorEntry
    {
        uint64_t uid;
        LogLevel level;
        std::string timestamp, message, sourceFile;
        int sourceLine;
        std::vector<InspectorStackLine> stack;
    };
    std::vector<InspectorEntry> m_inspectorEntries;
    std::string m_inspectorText;

    double m_subFlush = 0.0;
    double m_subCache = 0.0;
    double m_subToolbar = 0.0;
    double m_subBody = 0.0;

    // ── Helpers ──
    void FlushPendingLogs();
    void GetCountSnapshot(int &infoCount, int &warnCount, int &errorCount) const;
    void EnsureCache();
    void DetectFilterChange();
    bool MatchesCurrentFilters(const LogEntry &entry) const;
    std::string CollapseKey(const LogEntry &entry) const;
    int FindVisibleIndexByUid(uint64_t uid) const;
    [[nodiscard]] bool IsUidSelected(uint64_t uid) const noexcept;
    void ReplaceLocalSelection(uint64_t uid);
    void ToggleLocalSelection(uint64_t uid);
    void PruneLocalSelection();
    void SelectUid(uint64_t uid, bool focusWindow, bool publishSelection = true, bool recordHistory = true);
    void PublishSelection(uint64_t uid, bool recordHistory);
    void RenderToolbar(InxGUIContext *ctx);
    void RenderBody(InxGUIContext *ctx);
    void UpdateSelectionInspector();
    static InspectorStackLine ParseInspectorStackLine(std::string text);
    void RenderRow(InxGUIContext *ctx, int visIdx, const VisibleEntry &ve, bool selected);
    ImVec4 LevelColor(LogLevel lv) const;
    static std::string CurrentTimestamp();
    static bool IsInternalNoise(const std::string &msg);
};

} // namespace infernux
