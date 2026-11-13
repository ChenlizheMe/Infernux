#include "ConsolePanel.h"
#include "EditorThemeRegistry.h"

#include <function/scene/GameObject.h>
#include <function/scene/Scene.h>
#include <function/scene/SceneManager.h>
#include <platform/filesystem/InxPath.h>

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <imgui_internal.h>
#include <iomanip>
#include <sstream>

namespace infernux
{

// ════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ════════════════════════════════════════════════════════════════════

ConsolePanel::ConsolePanel(const std::string &projectRoot)
    : EditorPanel("Console", "console"), m_projectRoot(projectRoot)
{
    // Subscribe to INXLOG — receives ALL C++ log messages.
    m_sinkId = InxLog::GetInstance().AddSink(
        [this](LogLevel level, const char *file, int line, const std::string &message, bool internalOnly) {
            OnLogMessage(level, file, line, message, internalOnly);
        });
    m_diagnosticSubscription = InxLog::GetInstance().SubscribeDiagnostics([this](const OwnedDiagnosticEvent &event) {
        if (event.resolved)
            ResolveDiagnostic(event.owner, event.key);
        else
            SetDiagnostic(event.owner, event.key, event.level, event.message, "", event.sourceFile, event.sourceLine);
    });
}

ConsolePanel::~ConsolePanel()
{
    m_diagnosticSubscription.Reset();
    InxLog::GetInstance().RemoveSink(m_sinkId);
}

void ConsolePanel::CacheDisplayFields(LogEntry &entry) const
{
    entry.displayLine = entry.message.substr(0, entry.message.find('\n'));
    if (!entry.sourceFile.empty()) {
        entry.sourceLabel = FromFsPath(ToFsPath(NormalizePortablePath(entry.sourceFile)).filename());
        if (entry.sourceLine > 0)
            entry.sourceLabel += ":" + std::to_string(entry.sourceLine);
    }
}

std::unordered_map<std::string, double> ConsolePanel::ConsumeSubTimings()
{
    std::unordered_map<std::string, double> result{
        {"flush", m_subFlush}, {"cache", m_subCache}, {"toolbar", m_subToolbar}, {"body", m_subBody}};
    m_subFlush = m_subCache = m_subToolbar = m_subBody = 0.0;
    return result;
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
    entry.firstLine =
        FormatDiagnosticMessage(level, entry.sourceFile, line, entry.message.substr(0, nl), m_projectRoot);
    CacheDisplayFields(entry);

    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        entry.uid = m_nextUid++;
        switch (entry.level) {
        case LOG_WARN:
            ++m_warnCount;
            break;
        case LOG_ERROR:
        case LOG_FATAL:
            ++m_errorCount;
            break;
        default:
            ++m_infoCount;
            break;
        }
        m_pendingLogs.push_back(std::move(entry));
        m_revision.fetch_add(1, std::memory_order_release);
    }
}

// ════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::LogFromPython(LogLevel level, const std::string &message, const std::string &stackTrace,
                                 const std::string &sourceFile, int sourceLine, const ObjectHandle &contextHandle)
{
    LogEntry entry;
    entry.level = level;
    entry.message = message;
    entry.stackTrace = stackTrace;
    entry.sourceFile = sourceFile;
    entry.sourceLine = sourceLine;
    entry.contextHandle = contextHandle;
    entry.timestamp = CurrentTimestamp();

    auto nl = entry.message.find('\n');
    entry.firstLine =
        FormatDiagnosticMessage(level, sourceFile, sourceLine, entry.message.substr(0, nl), m_projectRoot);
    CacheDisplayFields(entry);

    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        entry.uid = m_nextUid++;
        switch (entry.level) {
        case LOG_WARN:
            ++m_warnCount;
            break;
        case LOG_ERROR:
        case LOG_FATAL:
            ++m_errorCount;
            break;
        default:
            ++m_infoCount;
            break;
        }
        m_pendingLogs.push_back(std::move(entry));
        m_revision.fetch_add(1, std::memory_order_release);
    }
}

void ConsolePanel::UpdateCounts(LogLevel level, int delta)
{
    if (level == LOG_WARN)
        m_warnCount += delta;
    else if (level == LOG_ERROR || level == LOG_FATAL)
        m_errorCount += delta;
    else
        m_infoCount += delta;
}

void ConsolePanel::SetDiagnostic(const std::string &owner, const std::string &key, LogLevel level,
                                 const std::string &message, const std::string &stackTrace,
                                 const std::string &sourceFile, int sourceLine, const ObjectHandle &contextHandle,
                                 int occurrences)
{
    const std::string identity = owner + '\0' + key;
    std::lock_guard lock(m_logMutex);
    auto found = m_diagnostics.find(identity);
    if (found != m_diagnostics.end()) {
        const auto &previous = found->second;
        if (previous.level == level && previous.message == message && previous.stackTrace == stackTrace &&
            previous.sourceFile == sourceFile && previous.sourceLine == sourceLine &&
            previous.contextHandle == contextHandle && previous.occurrences == occurrences)
            return;
        UpdateCounts(previous.level, -1);
    }
    LogEntry entry;
    entry.level = level;
    entry.message = message;
    entry.stackTrace = stackTrace;
    entry.sourceFile = sourceFile;
    entry.sourceLine = sourceLine;
    entry.contextHandle = contextHandle;
    entry.diagnosticOwner = owner;
    entry.diagnosticKey = key;
    entry.occurrences = occurrences;
    entry.uid = found == m_diagnostics.end() ? m_nextUid++ : found->second.uid;
    entry.timestamp = found == m_diagnostics.end() ? CurrentTimestamp() : found->second.timestamp;
    entry.firstLine =
        FormatDiagnosticMessage(level, sourceFile, sourceLine, message.substr(0, message.find('\n')), m_projectRoot);
    CacheDisplayFields(entry);
    UpdateCounts(level, 1);
    m_pendingDiagnosticChanges[entry.uid] = entry;
    m_diagnostics[identity] = std::move(entry);
    m_revision.fetch_add(1, std::memory_order_release);
}

bool ConsolePanel::ResolveDiagnostic(const std::string &owner, const std::string &key)
{
    std::lock_guard lock(m_logMutex);
    const auto found = m_diagnostics.find(owner + '\0' + key);
    if (found == m_diagnostics.end())
        return false;
    UpdateCounts(found->second.level, -1);
    m_pendingDiagnosticChanges[found->second.uid] = std::nullopt;
    m_diagnostics.erase(found);
    m_revision.fetch_add(1, std::memory_order_release);
    return true;
}

uint64_t ConsolePanel::GetEntryContextObjectId(uint64_t uid)
{
    const auto entry =
        std::find_if(m_logs.begin(), m_logs.end(), [uid](const LogEntry &log) { return log.uid == uid; });
    if (entry == m_logs.end() || !entry->contextHandle.IsValid())
        return 0;
    Scene *scene = SceneManager::Instance().GetSceneByWorldId(entry->contextHandle.worldId);
    GameObject *object = scene ? scene->ResolveGameObject(entry->contextHandle) : nullptr;
    return object ? object->GetID() : 0;
}

void ConsolePanel::Clear()
{
    const bool selectionChanged = m_selectedUid != 0 || !m_selectedUids.empty();
    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        m_logs.clear();
        m_pendingLogs.clear();
        m_diagnostics.clear();
        m_pendingDiagnosticChanges.clear();
        m_inspectorDirty = true;
        m_infoCount = 0;
        m_warnCount = 0;
        m_errorCount = 0;
        m_selectedUid = 0;
        m_selectedUids.clear();
        m_selectedUidLookup.clear();
        m_requestedUid = 0;
        m_followTail = true;
        m_scrollToBottom = false;
        m_resetScrollToTop = true;
        m_cacheDirty = true;
        m_cachedInfoCount = 0;
        m_cachedWarnCount = 0;
        m_cachedErrorCount = 0;
        m_visible.clear();
        m_collapseLookup.clear();
        m_revision.fetch_add(1, std::memory_order_release);
    }
    if (selectionChanged)
        PublishSelection(0, false);
}

size_t ConsolePanel::RemoveEntriesFromSource(const std::string &sourceFile, const std::string &messagePrefix)
{
    FlushPendingLogs();
    const std::string sourceKey = FilesystemPathKey(sourceFile);
    if (sourceKey.empty())
        return 0;

    size_t removed = 0;
    bool selectionChanged = false;
    uint64_t replacementSelection = 0;
    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        const auto matchesSource = [&sourceKey, &messagePrefix](const LogEntry &entry) {
            return !entry.sourceFile.empty() && FilesystemPathKey(entry.sourceFile) == sourceKey &&
                   (messagePrefix.empty() || entry.message.compare(0, messagePrefix.size(), messagePrefix) == 0);
        };
        const auto selected = std::find_if(m_logs.begin(), m_logs.end(), [this, &matchesSource](const LogEntry &entry) {
            return entry.uid == m_selectedUid && matchesSource(entry);
        });
        selectionChanged = selected != m_logs.end();

        const size_t logCountBefore = m_logs.size();
        m_logs.erase(std::remove_if(m_logs.begin(), m_logs.end(), matchesSource), m_logs.end());
        removed += logCountBefore - m_logs.size();
        const size_t pendingCountBefore = m_pendingLogs.size();
        m_pendingLogs.erase(std::remove_if(m_pendingLogs.begin(), m_pendingLogs.end(), matchesSource),
                            m_pendingLogs.end());
        removed += pendingCountBefore - m_pendingLogs.size();
        for (auto entry = m_diagnostics.begin(); entry != m_diagnostics.end();) {
            if (matchesSource(entry->second)) {
                m_pendingDiagnosticChanges.erase(entry->second.uid);
                entry = m_diagnostics.erase(entry);
            } else {
                ++entry;
            }
        }
        if (removed == 0)
            return 0;

        int infoCount = 0;
        int warnCount = 0;
        int errorCount = 0;
        const auto countEntry = [&infoCount, &warnCount, &errorCount](const LogEntry &entry) {
            if (entry.level == LOG_WARN)
                ++warnCount;
            else if (entry.level == LOG_ERROR || entry.level == LOG_FATAL)
                ++errorCount;
            else
                ++infoCount;
        };
        for (const LogEntry &entry : m_logs)
            if (entry.diagnosticOwner.empty())
                countEntry(entry);
        for (const LogEntry &entry : m_pendingLogs)
            countEntry(entry);
        for (const auto &[identity, entry] : m_diagnostics)
            countEntry(entry);
        m_infoCount = infoCount;
        m_warnCount = warnCount;
        m_errorCount = errorCount;

        if (selectionChanged) {
            m_selectedUid = 0;
            m_requestedUid = 0;
        }
        PruneLocalSelection();
        if (selectionChanged && !m_selectedUids.empty()) {
            replacementSelection = m_selectedUids.back();
            m_selectedUid = replacementSelection;
            m_requestedUid = replacementSelection;
        }
        m_resetScrollToTop = true;
        m_cacheDirty = true;
        m_inspectorDirty = true;
        m_filterDirty = true;
        m_visible.clear();
        m_collapseLookup.clear();
        m_revision.fetch_add(1, std::memory_order_release);
    }
    if (selectionChanged)
        PublishSelection(replacementSelection, false);
    return removed;
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
    FlushPendingLogs();
    if (m_logs.empty()) {
        m_isOpen = true;
        if (onRequestFocus)
            onRequestFocus();
        return;
    }
    SelectUid(m_logs.back().uid, true);
}

void ConsolePanel::SelectEntry(uint64_t uid)
{
    FlushPendingLogs();
    if (uid == 0) {
        SelectLatestEntry();
        return;
    }
    SelectUid(uid, true);
}

void ConsolePanel::SetSelectionSnapshot(uint64_t uid)
{
    // A Ctrl selection publishes its most recently toggled entry as the
    // global primary. Preserve the Console-owned set when that projection
    // comes back; an unrelated external projection replaces it.
    const bool projectedLocalPrimary = uid != 0 && IsUidSelected(uid);
    if (!projectedLocalPrimary)
        ReplaceLocalSelection(uid);
    m_selectedUid = uid;
    m_requestedUid = uid;
    if (uid != 0) {
        m_followTail = false;
        m_scrollToBottom = false;
    }
}

void ConsolePanel::GetStatusBarSnapshot(std::string &outMsg, std::string &outLevel, int &outInfoCount,
                                        int &outWarnCount, int &outErrorCount, uint64_t &outUid)
{
    // The status bar is always rendered, even when the Console window is
    // closed. Flushing here keeps the native queue bounded and gives the
    // displayed message a stable UID that the subsequent click can select.
    FlushPendingLogs();
    outMsg.clear();
    outLevel = "info";
    outUid = 0;
    GetCountSnapshot(outInfoCount, outWarnCount, outErrorCount);
    LogEntry pendingLatest;
    bool hasPending = false;
    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        if (!m_pendingLogs.empty()) {
            pendingLatest = m_pendingLogs.back();
            hasPending = true;
        }
    }
    if (!hasPending && m_logs.empty())
        return;
    const LogEntry &log = hasPending ? pendingLatest : m_logs.back();
    outMsg = log.firstLine;
    outUid = log.uid;
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

uint64_t ConsolePanel::GetRevision() const noexcept
{
    return m_revision.load(std::memory_order_acquire);
}

uint64_t ConsolePanel::GetSelectedUid() const noexcept
{
    return m_selectedUid;
}

bool ConsolePanel::HasSelectedEntry() const noexcept
{
    return !m_selectedUids.empty();
}

std::vector<uint64_t> ConsolePanel::GetSelectedUids() const
{
    return m_selectedUids;
}

std::vector<ConsolePanel::VisibleLogSnapshot> ConsolePanel::GetVisibleLogSnapshot(size_t limit)
{
    FlushPendingLogs();
    EnsureCache();

    std::vector<VisibleLogSnapshot> result;
    if (limit == 0 || m_visible.empty())
        return result;

    const size_t first = m_visible.size() > limit ? m_visible.size() - limit : 0;
    result.reserve(m_visible.size() - first);
    for (size_t visibleIndex = first; visibleIndex < m_visible.size(); ++visibleIndex) {
        const VisibleEntry &visible = m_visible[visibleIndex];
        if (visible.logIndex >= m_logs.size())
            continue;
        const LogEntry &entry = m_logs[visible.logIndex];
        VisibleLogSnapshot snapshot;
        snapshot.message = entry.message;
        snapshot.timestamp = entry.timestamp;
        snapshot.stackTrace = entry.stackTrace;
        snapshot.sourceFile = entry.sourceFile;
        snapshot.sourceLine = entry.sourceLine;
        snapshot.level = entry.level;
        snapshot.uid = visible.uid;
        snapshot.latestUid = visible.latestUid;
        snapshot.count = visible.count;
        snapshot.diagnosticOwner = entry.diagnosticOwner;
        snapshot.diagnosticKey = entry.diagnosticKey;
        snapshot.occurrences = entry.occurrences;
        result.push_back(std::move(snapshot));
    }
    return result;
}

bool ConsolePanel::CopySelectedEntry()
{
    UpdateSelectionInspector();
    if (m_inspectorText.empty())
        return false;
    ImGui::SetClipboardText(m_inspectorText.c_str());
    return true;
}

void ConsolePanel::UpdateSelectionInspector()
{
    FlushPendingLogs();
    if (!m_inspectorDirty && m_inspectorUids == m_selectedUids && m_inspectorPrimaryUid == m_selectedUid)
        return;
    m_inspectorDirty = false;
    m_inspectorUids = m_selectedUids;
    m_inspectorPrimaryUid = m_selectedUid;
    m_inspectorText.clear();
    m_inspectorEntries.clear();
    // Only selected diagnostic updates invalidate this projection. Unrelated
    // messages do not rebuild a selected stack, and filters do not own selection.
    for (const auto &log : m_logs) {
        if (!IsUidSelected(log.uid))
            continue;
        if (!m_inspectorText.empty())
            m_inspectorText += "\n\n----------------------------------------\n\n";
        m_inspectorText +=
            FormatDiagnosticMessage(log.level, log.sourceFile, log.sourceLine, log.message, m_projectRoot);
        if (!log.stackTrace.empty())
            m_inspectorText += "\n\n" + log.stackTrace;
        InspectorEntry view{log.uid, log.level, log.timestamp, log.message, log.sourceFile, log.sourceLine, {}};
        std::istringstream stack(log.stackTrace);
        for (std::string line; std::getline(stack, line);)
            view.stack.push_back(ParseInspectorStackLine(std::move(line)));
        m_inspectorEntries.push_back(std::move(view));
    }
}

ConsolePanel::InspectorStackLine ConsolePanel::ParseInspectorStackLine(std::string text)
{
    InspectorStackLine line{std::move(text), {}, 0};
    const auto begin = line.text.find("File \"");
    if (begin == std::string::npos)
        return line;
    const auto end = line.text.find("\", line ", begin + 6);
    if (end == std::string::npos)
        return line;
    const char *number = line.text.data() + end + 8;
    const auto parsed = std::from_chars(number, line.text.data() + line.text.size(), line.sourceLine);
    if (parsed.ec != std::errc{})
        return line;
    line.sourceFile = line.text.substr(begin + 6, end - begin - 6);
    line.text = line.sourceFile + ":" + std::to_string(line.sourceLine) +
                std::string(parsed.ptr, line.text.c_str() + line.text.size());
    return line;
}

void ConsolePanel::RenderSelectionInspector(InxGUIContext *ctx)
{
    UpdateSelectionInspector();
    if (m_inspectorEntries.empty()) {
        ctx->EmptyState("##diagnostic_empty", "glyphs/info", translate("console.empty"),
                        translate("console.empty_hint"));
        return;
    }
    const float dpi = ctx->GetDpiScale();
    const float smallSize = EditorThemeRegistry::Float("EDITOR_SMALL_FONT_SIZE") * dpi;
    const auto muted = EditorThemeRegistry::Color("ROLE_TEXT_DIM");
    const auto surface = EditorThemeRegistry::Color("ROLE_BG_SURFACE");
    ImGui::BeginGroup();
    const float available = ImGui::GetContentRegionAvail().x;
    const std::string copyLabel = translate("console.copy_details");
    const float copyWidth = ImGui::CalcTextSize(copyLabel.c_str()).x + 20 * dpi;
    const ImVec2 titleOrigin = ImGui::GetCursorScreenPos();
    ctx->PushUIFont(EditorThemeRegistry::Float("EDITOR_TITLE_FONT_SIZE"));
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(translate("console.details").c_str());
    ctx->PopFont();
    if (available > ImGui::GetItemRectSize().x + copyWidth + 12 * dpi) {
        ImGui::SameLine();
        ImGui::SetCursorScreenPos(ImVec2(titleOrigin.x + available - copyWidth, titleOrigin.y));
    }
    if (ImGui::Button(copyLabel.c_str(), ImVec2(copyWidth, 28 * dpi)))
        CopySelectedEntry();
    ctx->RecordSemanticItem("button", copyLabel, true, "console.detail.copy");
    ImGui::Dummy(ImVec2(0, 4 * dpi));
    ImGui::Separator();

    // Source links use the same wrapped text bounds as the text itself. A
    // path never creates a horizontal scroll region or a one-line clip.
    auto sourceLink = [&](const std::string &label, const std::string &path, int line, const std::string &id) {
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImGui::PushStyleColor(ImGuiCol_Text, muted);
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(label.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
        const ImVec2 size = ImGui::GetItemRectSize();
        const ImVec2 next = ImGui::GetCursorScreenPos();
        const bool isFile = !path.empty() && path.front() != '<';
        if (isFile) {
            ImGui::SetCursorScreenPos(origin);
            const bool open = ImGui::InvisibleButton(id.c_str(), size, ImGuiButtonFlags_EnableNav);
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                ImGui::GetWindowDrawList()->AddLine(ImVec2(origin.x, origin.y + size.y),
                                                    ImVec2(origin.x + size.x, origin.y + size.y),
                                                    ImGui::GetColorU32(muted));
            }
            ctx->RecordSemanticItem("source_link", label, true, id, std::nullopt, std::nullopt, path);
            if (open)
                ExecuteEditorCommand("console.open_source", "pointer", path + "\t" + std::to_string(line));
            ImGui::SetCursorScreenPos(next);
        } else {
            ctx->RecordSemanticItem("source_label", label, true, id);
        }
    };

    for (size_t index = 0; index < m_inspectorEntries.size(); ++index) {
        const auto &entry = m_inspectorEntries[index];
        ImGui::PushID(static_cast<int>(entry.uid));
        const std::string id = "console.detail." + std::to_string(entry.uid);
        const bool error = entry.level == LOG_ERROR || entry.level == LOG_FATAL;
        const bool warning = entry.level == LOG_WARN;
        const auto statusColor = EditorThemeRegistry::Color(error ? "LOG_ERROR" : warning ? "LOG_WARNING" : "LOG_INFO");
        ctx->BeginFramedGroup(12, 12, 0, {surface.x, surface.y, surface.z, surface.w}, {0, 0, 0, 0});
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(14 * dpi, 16 * dpi));
        const auto glyph = ctx->EditorIcon(error ? "glyphs/error" : warning ? "glyphs/warn" : "glyphs/info");
        if (glyph)
            ImGui::GetWindowDrawList()->AddImage(ImTextureRef(static_cast<ImTextureID>(glyph)), origin,
                                                 ImVec2(origin.x + 14 * dpi, origin.y + 14 * dpi), ImVec2(0, 0),
                                                 ImVec2(1, 1), ImGui::GetColorU32(statusColor));
        ImGui::SameLine(0, 8 * dpi);
        ctx->PushUIFont(EditorThemeRegistry::Float("EDITOR_SMALL_FONT_SIZE"));
        ImGui::TextColored(statusColor, "%s",
                           translate(error     ? "console.level_error"
                                     : warning ? "console.level_warning"
                                               : "console.level_info")
                               .c_str());
        ctx->PopFont();
        ImGui::SameLine(0, 12 * dpi);
        ctx->PushMonoFont();
        ImGui::PushFont(ImGui::GetFont(), smallSize);
        ImGui::TextColored(muted, "%s", entry.timestamp.c_str());
        ImGui::PopFont();
        ctx->PopFont();
        ImGui::PushTextWrapPos(0);
        ImGui::TextUnformatted(entry.message.c_str());
        ImGui::PopTextWrapPos();
        ctx->RecordSemanticItem("diagnostic_message", entry.message, true, id + ".message");
        if (!entry.sourceFile.empty()) {
            ImGui::Dummy(ImVec2(0, 2 * dpi));
            ctx->PushMonoFont();
            ImGui::PushFont(ImGui::GetFont(), smallSize);
            const std::string label =
                entry.sourceFile + (entry.sourceLine > 0 ? ":" + std::to_string(entry.sourceLine) : "");
            sourceLink(label, entry.sourceFile, entry.sourceLine,
                       index == 0 ? "console.detail.source" : id + ".source");
            ImGui::PopFont();
            ctx->PopFont();
        }
        ctx->EndFramedGroup();
        if (!entry.stack.empty()) {
            ctx->PushMonoFont();
            ImGui::PushFont(ImGui::GetFont(), smallSize);
            ImGui::PushStyleColor(ImGuiCol_Text, muted);
            ImGui::SeparatorText(translate("console.stack_trace").c_str());
            ImGui::PopStyleColor();
            ImGui::Indent(12 * dpi);
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 4 * dpi));
            for (size_t row = 0; row < entry.stack.size(); ++row) {
                const auto &line = entry.stack[row];
                if (!line.sourceFile.empty()) {
                    sourceLink(line.text, line.sourceFile, line.sourceLine, id + ".stack." + std::to_string(row));
                } else {
                    ImGui::PushTextWrapPos(0);
                    ImGui::TextUnformatted(line.text.c_str());
                    ImGui::PopTextWrapPos();
                    ctx->RecordSemanticItem("diagnostic_code", line.text, true, id + ".stack." + std::to_string(row));
                }
            }
            ImGui::PopStyleVar();
            ImGui::Unindent(12 * dpi);
            ImGui::PopFont();
            ctx->PopFont();
        }
        if (index + 1 < m_inspectorEntries.size())
            ImGui::Separator();
        ImGui::PopID();
    }
    ImGui::EndGroup();
    ctx->RecordSemanticItem("console_detail", translate("console.details"), true, "console.detail", std::nullopt,
                            static_cast<double>(m_inspectorUids.size()), m_inspectorText);
}

bool ConsolePanel::HasViewOption(const std::string &option) const noexcept
{
    return option == "show_info" || option == "show_warnings" || option == "show_errors" || option == "collapse" ||
           option == "clear_on_play" || option == "error_pause" || option == "follow";
}

bool ConsolePanel::GetViewOption(const std::string &option) const noexcept
{
    if (option == "show_info")
        return showInfo;
    if (option == "show_warnings")
        return showWarnings;
    if (option == "show_errors")
        return showErrors;
    if (option == "collapse")
        return collapse;
    if (option == "clear_on_play")
        return clearOnPlay;
    if (option == "error_pause")
        return errorPause;
    if (option == "follow")
        return autoScroll && m_followTail;
    return false;
}

void ConsolePanel::SetViewOption(const std::string &option, bool enabled)
{
    if (option == "show_info")
        showInfo = enabled;
    else if (option == "show_warnings")
        showWarnings = enabled;
    else if (option == "show_errors")
        showErrors = enabled;
    else if (option == "collapse")
        collapse = enabled;
    else if (option == "clear_on_play")
        clearOnPlay = enabled;
    else if (option == "error_pause")
        errorPause = enabled;
    else if (option == "follow") {
        autoScroll = enabled;
        m_followTail = enabled;
        if (enabled) {
            m_requestedUid = 0;
            m_scrollToBottom = true;
        }
    }
}

std::string ConsolePanel::GetSearchQuery() const
{
    return std::string(m_search.data());
}

void ConsolePanel::SetSearchQuery(const std::string &query)
{
    std::snprintf(m_search.data(), m_search.size(), "%s", query.c_str());
    if (m_searchModel.SetQuery(m_search.data()))
        m_filterDirty = true;
    m_followTail = false;
    m_resetScrollToTop = true;
}

// ════════════════════════════════════════════════════════════════════
// Render
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::OnRenderContent(InxGUIContext *ctx)
{
    const float dpi = ctx->GetDpiScale();
    if (std::abs(dpi - m_lastDpiScale) >= 0.01f) {
        m_lastDpiScale = dpi;
        m_rowHeightMeasured = false;
        m_rowHeight =
            (EditorThemeRegistry::Float("CONSOLE_ROW_HEIGHT") + EditorThemeRegistry::Vec2("CONTENT_SPACING").y) * dpi;
    }
    const auto toolbarStart = std::chrono::steady_clock::now();
    RenderToolbar(ctx);
    // Toolbar commands execute synchronously and may change filters or clear
    // the model. Rebuild here so the body uses the new list coordinates in
    // this very frame instead of briefly drawing the previous viewport.
    EnsureCache();
    const auto bodyStart = std::chrono::steady_clock::now();
    RenderBody(ctx);
    const auto bodyEnd = std::chrono::steady_clock::now();
    m_subToolbar += std::chrono::duration<double, std::milli>(bodyStart - toolbarStart).count();
    m_subBody += std::chrono::duration<double, std::milli>(bodyEnd - bodyStart).count();
}

void ConsolePanel::PreRender(InxGUIContext * /*ctx*/)
{
    const auto flushStart = std::chrono::steady_clock::now();
    FlushPendingLogs();
    const auto cacheStart = std::chrono::steady_clock::now();
    EnsureCache();
    const auto cacheEnd = std::chrono::steady_clock::now();
    m_subFlush += std::chrono::duration<double, std::milli>(cacheStart - flushStart).count();
    m_subCache += std::chrono::duration<double, std::milli>(cacheEnd - cacheStart).count();
}

// ════════════════════════════════════════════════════════════════════
// Flush pending logs (main thread only)
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::ApplyDiagnosticChanges(std::unordered_map<uint64_t, std::optional<LogEntry>> &changes,
                                          std::vector<LogEntry> &incoming)
{
    if (changes.empty())
        return;
    const auto previousSelection = m_selectedUids;
    for (auto &[uid, candidate] : changes) {
        const auto current =
            std::find_if(m_logs.begin(), m_logs.end(), [uid](const LogEntry &entry) { return entry.uid == uid; });
        if (IsUidSelected(uid))
            m_inspectorDirty = true;
        if (candidate) {
            if (current == m_logs.end())
                incoming.push_back(std::move(*candidate));
            else
                *current = std::move(*candidate);
        } else if (current != m_logs.end()) {
            m_logs.erase(current);
        }
    }
    std::sort(incoming.begin(), incoming.end(), [](const LogEntry &a, const LogEntry &b) { return a.uid < b.uid; });
    PruneLocalSelection();
    if (previousSelection != m_selectedUids) {
        if (!IsUidSelected(m_selectedUid))
            m_selectedUid = m_selectedUids.empty() ? 0 : m_selectedUids.back();
        m_requestedUid = m_selectedUid;
        PublishSelection(m_selectedUid, false);
    }
    m_cacheDirty = true;
}

void ConsolePanel::FlushPendingLogs()
{
    std::vector<LogEntry> incoming;
    std::unordered_map<uint64_t, std::optional<LogEntry>> diagnosticChanges;
    {
        std::lock_guard<std::mutex> lock(m_logMutex);
        if (m_pendingLogs.empty() && m_pendingDiagnosticChanges.empty())
            return;
        incoming.swap(m_pendingLogs);
        diagnosticChanges.swap(m_pendingDiagnosticChanges);
    }
    ApplyDiagnosticChanges(diagnosticChanges, incoming);

    DetectFilterChange();
    const bool appendToVisible = !m_cacheDirty && !m_filterDirty;
    bool receivedError = false;
    for (auto &entry : incoming) {
        const size_t logIndex = m_logs.size();
        if (appendToVisible && MatchesCurrentFilters(entry)) {
            if (collapse) {
                const std::string key = CollapseKey(entry);
                const auto it = m_collapseLookup.find(key);
                if (it != m_collapseLookup.end()) {
                    VisibleEntry &visible = m_visible[it->second];
                    visible.count += entry.occurrences;
                    visible.latestUid = entry.uid;
                } else {
                    m_collapseLookup.emplace(key, m_visible.size());
                    m_visible.push_back({logIndex, entry.occurrences, entry.uid, entry.uid});
                }
            } else {
                m_visible.push_back({logIndex, entry.occurrences, entry.uid, entry.uid});
            }
        }
        switch (entry.level) {
        case LOG_WARN:
            ++m_cachedWarnCount;
            break;
        case LOG_ERROR:
        case LOG_FATAL:
            ++m_cachedErrorCount;
            receivedError = true;
            break;
        default:
            ++m_cachedInfoCount;
            break;
        }
        m_logs.push_back(std::move(entry));
    }

    bool trimmed = false;
    while (m_logs.size() > MAX_LOGS) {
        const auto oldest = std::find_if(m_logs.begin(), m_logs.end(),
                                         [](const LogEntry &entry) { return entry.diagnosticOwner.empty(); });
        if (oldest == m_logs.end())
            break;
        UpdateCounts(oldest->level, -1);
        m_logs.erase(oldest);
        trimmed = true;
    }

    bool trimmedSelectionChanged = false;
    if (trimmed && m_selectedUid != 0) {
        const auto selected = std::find_if(m_logs.begin(), m_logs.end(),
                                           [this](const LogEntry &entry) { return entry.uid == m_selectedUid; });
        if (selected == m_logs.end()) {
            trimmedSelectionChanged = true;
            m_selectedUid = 0;
            m_requestedUid = 0;
        }
    }
    if (trimmed) {
        PruneLocalSelection();
        if (m_selectedUid == 0 && !m_selectedUids.empty()) {
            m_selectedUid = m_selectedUids.back();
            m_requestedUid = m_selectedUid;
        }
        if (trimmedSelectionChanged)
            PublishSelection(m_selectedUid, false);
    }

    // The common non-collapse path can extend the visible cache in O(new logs).
    // Filtering/collapse changes and deque trimming still use the full rebuild.
    if (!appendToVisible || trimmed)
        m_cacheDirty = true;
    if (autoScroll && m_followTail)
        m_scrollToBottom = true;
    if (receivedError && errorPause && onErrorPause)
        onErrorPause();
}

void ConsolePanel::GetCountSnapshot(int &infoCount, int &warnCount, int &errorCount) const
{
    infoCount = m_infoCount.load(std::memory_order_relaxed);
    warnCount = m_warnCount.load(std::memory_order_relaxed);
    errorCount = m_errorCount.load(std::memory_order_relaxed);
}

// ════════════════════════════════════════════════════════════════════
// Filter cache
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::DetectFilterChange()
{
    const bool searchChanged = m_searchModel.SetQuery(m_search.data());
    bool changed = (showInfo != m_prevShowInfo || showWarnings != m_prevShowWarnings ||
                    showErrors != m_prevShowErrors || collapse != m_prevCollapse || searchChanged);
    if (changed) {
        m_prevShowInfo = showInfo;
        m_prevShowWarnings = showWarnings;
        m_prevShowErrors = showErrors;
        m_prevCollapse = collapse;
        m_filterDirty = true;
        // A rebuilt virtual list has a different coordinate system. Reset in
        // the same frame instead of letting ImGui clamp the old offset over a
        // visible sequence of frames.
        m_resetScrollToTop = true;
        m_scrollToBottom = false;
        m_followTail = false;
    }
}

bool ConsolePanel::MatchesCurrentFilters(const LogEntry &entry) const
{
    bool severityMatches = showInfo;
    if (entry.level == LOG_WARN)
        severityMatches = showWarnings;
    else if (entry.level == LOG_ERROR || entry.level == LOG_FATAL)
        severityMatches = showErrors;
    if (!severityMatches)
        return false;

    return m_searchModel.Matches(entry.message) || m_searchModel.Matches(entry.stackTrace) ||
           m_searchModel.Matches(entry.sourceFile);
}

std::string ConsolePanel::CollapseKey(const LogEntry &entry) const
{
    if (!entry.diagnosticOwner.empty())
        return "diagnostic:" + entry.diagnosticOwner + '\0' + entry.diagnosticKey;
    return std::to_string(static_cast<int>(entry.level)) + "|" + entry.sourceFile + ":" +
           std::to_string(entry.sourceLine) + "|" + std::to_string(entry.contextHandle.id) + ":" +
           std::to_string(entry.contextHandle.generation) + ":" + std::to_string(entry.contextHandle.worldId) + "|" +
           entry.message;
}

int ConsolePanel::FindVisibleIndexByUid(uint64_t uid) const
{
    if (uid == 0)
        return -1;
    for (int index = 0; index < static_cast<int>(m_visible.size()); ++index) {
        const VisibleEntry &entry = m_visible[index];
        if (entry.uid == uid || entry.latestUid == uid)
            return index;
    }
    if (!collapse)
        return -1;

    const auto target =
        std::find_if(m_logs.begin(), m_logs.end(), [uid](const LogEntry &entry) { return entry.uid == uid; });
    if (target == m_logs.end())
        return -1;
    const std::string targetKey = CollapseKey(*target);
    const auto group = m_collapseLookup.find(targetKey);
    return group == m_collapseLookup.end() ? -1 : static_cast<int>(group->second);
}

bool ConsolePanel::IsUidSelected(uint64_t uid) const noexcept
{
    return uid != 0 && m_selectedUidLookup.find(uid) != m_selectedUidLookup.end();
}

void ConsolePanel::ReplaceLocalSelection(uint64_t uid)
{
    m_selectedUids.clear();
    m_selectedUidLookup.clear();
    if (uid != 0) {
        m_selectedUids.push_back(uid);
        m_selectedUidLookup.insert(uid);
    }
}

void ConsolePanel::ToggleLocalSelection(uint64_t uid)
{
    if (uid == 0)
        return;
    if (m_selectedUidLookup.erase(uid) > 0) {
        m_selectedUids.erase(std::remove(m_selectedUids.begin(), m_selectedUids.end(), uid), m_selectedUids.end());
        return;
    }
    m_selectedUidLookup.insert(uid);
    m_selectedUids.push_back(uid);
}

void ConsolePanel::PruneLocalSelection()
{
    if (m_selectedUids.empty())
        return;
    std::unordered_set<uint64_t> existing;
    existing.reserve(m_selectedUids.size());
    for (const LogEntry &entry : m_logs) {
        if (IsUidSelected(entry.uid))
            existing.insert(entry.uid);
    }
    m_selectedUids.erase(std::remove_if(m_selectedUids.begin(), m_selectedUids.end(),
                                        [this, &existing](uint64_t uid) {
                                            if (existing.find(uid) == existing.end()) {
                                                m_selectedUidLookup.erase(uid);
                                                return true;
                                            }
                                            return false;
                                        }),
                         m_selectedUids.end());
}

void ConsolePanel::SelectUid(uint64_t uid, bool focusWindow, bool publishSelection, bool recordHistory)
{
    const bool selectionChanged = m_selectedUid != uid;
    m_isOpen = true;
    m_requestedUid = uid;
    m_followTail = false;
    m_scrollToBottom = false;
    m_search[0] = '\0';
    ReplaceLocalSelection(uid);

    const auto target =
        std::find_if(m_logs.begin(), m_logs.end(), [uid](const LogEntry &entry) { return entry.uid == uid; });
    if (target != m_logs.end()) {
        if (target->level == LOG_WARN)
            showWarnings = true;
        else if (target->level == LOG_ERROR || target->level == LOG_FATAL)
            showErrors = true;
        else
            showInfo = true;
    }
    if (focusWindow) {
        if (onRequestFocus)
            onRequestFocus();
    }
    if (!selectionChanged)
        return;
    if (publishSelection)
        PublishSelection(uid, recordHistory);
    else
        SetSelectionSnapshot(uid);
}

void ConsolePanel::PublishSelection(uint64_t uid, bool recordHistory)
{
    if (onSelectionChanged)
        onSelectionChanged(uid, recordHistory);
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

    // Rebuild visible list
    m_visible.clear();
    m_collapseLookup.clear();

    for (size_t i = 0; i < m_logs.size(); ++i) {
        const auto &log = m_logs[i];

        // Apply filters
        if (!MatchesCurrentFilters(log))
            continue;

        if (collapse) {
            // Build collapse key: level + message
            const std::string key = CollapseKey(log);
            auto it = m_collapseLookup.find(key);
            if (it != m_collapseLookup.end()) {
                m_visible[it->second].count += log.occurrences;
                m_visible[it->second].latestUid = log.uid;
                continue;
            }
            m_collapseLookup[key] = m_visible.size();
        }

        VisibleEntry ve;
        ve.logIndex = i;
        ve.count = log.occurrences;
        ve.uid = log.uid;
        ve.latestUid = log.uid;
        m_visible.push_back(ve);
    }

    m_cacheDirty = false;
    m_filterDirty = false;
}

// ════════════════════════════════════════════════════════════════════
// Toolbar
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::RenderToolbar(InxGUIContext *ctx)
{
    const float dpi = ctx->GetDpiScale();
    const float available = ImGui::GetContentRegionAvail().x;
    const float gap = 4.0f * dpi;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f * dpi, 4.5f * dpi));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(gap, gap));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    float used = 0;
    auto nextControl = [&](float width) {
        if (used > 0 && used + gap + width <= available) {
            ImGui::SameLine(0, gap);
            used += gap + width;
        } else {
            used = width;
        }
    };
    const bool canClear = CanExecuteEditorCommand("console.clear");
    nextControl(54.0f * dpi);
    ImGui::BeginDisabled(!canClear);
    if (ImGui::Button(translate("console.clear").c_str(), ImVec2(54.0f * dpi, 24.0f * dpi)))
        ExecuteEditorCommand("console.clear", "pointer");
    ImGui::EndDisabled();
    ctx->RecordSemanticItem("console_action", translate("console.clear"), canClear, "console.clear");

    auto option = [&](const char *labelKey, const char *key, const char *semanticId) {
        const std::string label = translate(labelKey);
        const bool active = GetViewOption(key);
        const float width = ImGui::CalcTextSize(label.c_str()).x + 18.0f * dpi;
        nextControl(width);
        ImGui::PushStyleColor(ImGuiCol_Button, EditorThemeRegistry::Color(active ? "ROLE_BG_RAISED" : "BTN_GHOST"));
        if (ImGui::Button((label + "##" + key).c_str(), ImVec2(width, 24.0f * dpi)))
            ExecuteEditorCommand("console.set_option", "pointer", std::string(key) + "\t" + (active ? "0" : "1"));
        if (active) {
            const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
            ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(lo.x, hi.y - 2 * dpi), hi,
                                                      ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_ACCENT")));
        }
        ctx->RecordSemanticItem("checkbox", label, true, semanticId, GetViewOption(key));
        ImGui::PopStyleColor();
    };
    option("console.collapse", "collapse", "console.collapse");
    option("console.clear_on_play", "clear_on_play", "console.clear_on_play");
    option("console.error_pause", "error_pause", "console.error_pause");
    option("console.follow", "follow", "console.follow");

    const float segmentWidth = 60.0f * dpi;
    const float severityWidth = segmentWidth * 3 + gap * 2;
    const bool stacked = available < severityWidth + 140.0f * dpi;
    ImGui::SetNextItemWidth(stacked ? -1.0f : available - severityWidth - gap);
    if (m_focusSearchNextFrame) {
        ImGui::SetKeyboardFocusHere();
        m_focusSearchNextFrame = false;
    }
    const std::string searchHint = translate("console.search_hint");
    ImGui::InputTextWithHint("##ConsoleSearch", searchHint.c_str(), m_search.data(), m_search.size());
    if (ImGui::IsItemActivated())
        m_searchEditStart = m_search.data();
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        const std::string committed = m_search.data();
        if (committed != m_searchEditStart &&
            !ExecuteEditorCommand("console.set_search", "inline_edit", m_searchEditStart + "\n" + committed))
            SetSearchQuery(m_searchEditStart);
        m_searchEditStart.clear();
    }
    ctx->RecordSemanticItem("text_input", searchHint, true, "console.search", std::nullopt, std::nullopt,
                            std::string(m_search.data()));
    if (ImGui::IsItemEdited())
        m_followTail = false;

    auto severity = [&](const char *id, const char *key, const char *labelKey, int count, const char *glyph,
                        const char *colorKey) {
        const bool enabled = GetViewOption(key);
        const ImVec4 color = EditorThemeRegistry::Color(enabled ? colorKey : "TEXT_DISABLED");
        ImGui::PushStyleColor(ImGuiCol_Button, EditorThemeRegistry::Color(enabled ? "ROLE_BG_RAISED" : "BTN_GHOST"));
        if (ImGui::Button((std::string("##") + id).c_str(), ImVec2(segmentWidth, 22.0f * dpi)))
            ExecuteEditorCommand("console.set_option", "pointer", std::string(key) + "\t" + (enabled ? "0" : "1"));
        const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
        const auto texture = ctx->EditorIcon(glyph);
        if (texture)
            ImGui::GetWindowDrawList()->AddImage((ImTextureID)(intptr_t)texture, ImVec2(lo.x + 8 * dpi, lo.y + 5 * dpi),
                                                 ImVec2(lo.x + 20 * dpi, lo.y + 17 * dpi), ImVec2(0, 0), ImVec2(1, 1),
                                                 ImGui::GetColorU32(color));
        const std::string text = count > 999 ? "999+" : std::to_string(count);
        ctx->PushMonoFont();
        ImGui::GetWindowDrawList()->AddText(ImVec2(lo.x + 27 * dpi, (lo.y + hi.y - ImGui::GetFontSize()) * 0.5f),
                                            ImGui::GetColorU32(color), text.c_str());
        ctx->PopFont();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", translate(labelKey).c_str());
        ctx->RecordSemanticItem("console_filter", translate(labelKey), true, std::string("console.filter.") + id,
                                GetViewOption(key), static_cast<double>(count));
        ImGui::PopStyleColor();
    };
    if (!stacked)
        ImGui::SameLine(0, gap);
    severity("ConsoleFilterInfo", "show_info", "console.filter_info", m_cachedInfoCount, "glyphs/info", "LOG_INFO");
    ImGui::SameLine(0, gap);
    severity("ConsoleFilterWarn", "show_warnings", "console.filter_warning", m_cachedWarnCount, "glyphs/warn",
             "LOG_WARNING");
    ImGui::SameLine(0, gap);
    severity("ConsoleFilterError", "show_errors", "console.filter_error", m_cachedErrorCount, "glyphs/error",
             "LOG_ERROR");
    ImGui::PopStyleVar(3);
}

void ConsolePanel::RenderBody(InxGUIContext *ctx)
{
    const float dpi = ctx->GetDpiScale();
    int total = static_cast<int>(m_visible.size());
    float rowH = m_rowHeight;

    // ── Log list (virtual-scrolled) ──
    ImGui::PushStyleColor(ImGuiCol_Border, EditorThemeRegistry::Color("BORDER_TRANSPARENT"));
    if (ImGui::BeginChild("##ConsoleLogList", ImVec2(0, 0), ImGuiChildFlags_AlwaysUseWindowPadding)) {
        if (m_visible.empty())
            ctx->EmptyState("##console_empty", "glyphs/info", translate("console.empty"),
                            translate("console.empty_hint"));
        float scrollY = ImGui::GetScrollY();
        if (m_resetScrollToTop) {
            ImGui::SetScrollY(0.0f);
            // SetScrollY records a target that ImGui applies while completing
            // the child window. The virtualizer must use that same target in
            // this frame; otherwise it renders rows from the stale offset once
            // and the filtered list visibly appears to crawl back to the top.
            scrollY = 0.0f;
            m_resetScrollToTop = false;
        }
        // Freeze follow on mouse-down, before Selectable resolves on release.
        // New messages continue entering the model without moving the target
        // row out from under the pointer.
        if (ImGui::IsWindowHovered() &&
            (ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::GetIO().MouseWheel != 0.0f)) {
            m_followTail = false;
            m_scrollToBottom = false;
        }

        if (m_requestedUid > 0) {
            const int selectedIndex = FindVisibleIndexByUid(m_requestedUid);
            if (selectedIndex >= 0) {
                const float targetY = selectedIndex * rowH;
                const float viewH = ImGui::GetContentRegionAvail().y;
                if (targetY < scrollY || targetY + rowH > scrollY + viewH) {
                    scrollY = (std::max)(0.0f, targetY - viewH * 0.35f);
                    ImGui::SetScrollY(scrollY);
                }
            }
            m_requestedUid = 0;
        }

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
                RenderRow(ctx, idx, m_visible[idx], IsUidSelected(m_visible[idx].uid));
                float y1 = ImGui::GetCursorPosY();
                float measured = y1 - y0;
                if (measured > 1.0f) {
                    m_rowHeight = measured;
                    rowH = measured;
                    m_rowHeightMeasured = true;
                }
            } else {
                RenderRow(ctx, idx, m_visible[idx], IsUidSelected(m_visible[idx].uid));
            }
        }

        // Bottom spacer
        int remaining = total - (lastVis + 1);
        if (remaining > 0) {
            float w = ImGui::GetContentRegionAvail().x;
            ImGui::Dummy(ImVec2(w, remaining * rowH));
        }

        if (m_scrollToBottom && !m_visible.empty()) {
            ImGui::SetScrollHereY(1.0f);
            m_scrollToBottom = false;
        }

        // Wheel/scrollbar interaction owns follow state. Appending messages
        // never changes it, so selecting a row remains stable under log floods.
        if (ImGui::IsWindowHovered() &&
            (ImGui::GetIO().MouseWheel != 0.0f || ImGui::IsMouseDragging(ImGuiMouseButton_Left))) {
            scrollY = ImGui::GetScrollY();
            const float scrollMax = ImGui::GetScrollMaxY();
            const bool atBottom = scrollMax <= 0.0f || (scrollMax - scrollY) < 20.0f * dpi;
            m_followTail = autoScroll && atBottom && m_selectedUid == 0;
        }
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(); // Border
}

// ════════════════════════════════════════════════════════════════════
// Single row
// ════════════════════════════════════════════════════════════════════

void ConsolePanel::RenderRow(InxGUIContext *ctx, int visIdx, const VisibleEntry &ve, bool isSel)
{
    const float dpi = ctx->GetDpiScale();
    const auto &log = m_logs[ve.logIndex];
    const ImVec4 color = LevelColor(log.level);
    ImGui::PushStyleColor(ImGuiCol_Header, EditorThemeRegistry::Color(isSel        ? "ROW_SELECTED"
                                                                      : visIdx % 2 ? "ROW_ALT"
                                                                                   : "ROW_NONE"));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, EditorThemeRegistry::Color("ROLE_BG_HOVER"));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, EditorThemeRegistry::Color("ROW_SELECTED"));
    const std::string id = "##clog_" + std::to_string(ve.uid);
    if (ImGui::Selectable(id.c_str(), isSel,
                          ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick,
                          ImVec2(0, EditorThemeRegistry::Float("CONSOLE_ROW_HEIGHT") * dpi))) {
        m_requestedUid = 0;
        m_followTail = false;
        m_scrollToBottom = false;
        if (ImGui::IsMouseDoubleClicked(0) && !log.sourceFile.empty())
            ExecuteEditorCommand("console.open_source", "pointer",
                                 log.sourceFile + "\t" + std::to_string((std::max)(log.sourceLine, 0)));
        if (ImGui::GetIO().KeyCtrl) {
            ToggleLocalSelection(ve.uid);
            const uint64_t primary =
                IsUidSelected(ve.uid) ? ve.uid : (m_selectedUids.empty() ? 0 : m_selectedUids.back());
            if (primary != m_selectedUid)
                PublishSelection(primary, true);
        } else {
            const bool changed = m_selectedUid != ve.uid;
            if (m_selectedUids.size() != 1 || !IsUidSelected(ve.uid))
                ReplaceLocalSelection(ve.uid);
            if (changed)
                PublishSelection(ve.uid, true);
        }
    }
    const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    auto *draw = ImGui::GetWindowDrawList();
    if (isSel)
        draw->AddRectFilled(lo, ImVec2(lo.x + 2 * dpi, hi.y),
                            ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_ACCENT")));
    const bool error = log.level == LOG_ERROR || log.level == LOG_FATAL;
    const bool warning = log.level == LOG_WARN;
    const auto glyph = ctx->EditorIcon(error ? "glyphs/error" : warning ? "glyphs/warn" : "glyphs/info");
    if (glyph)
        draw->AddImage((ImTextureID)(intptr_t)glyph, ImVec2(lo.x + 8 * dpi, lo.y + 8 * dpi),
                       ImVec2(lo.x + 20 * dpi, lo.y + 20 * dpi), ImVec2(0, 0), ImVec2(1, 1),
                       ImGui::GetColorU32(EditorThemeRegistry::Color(error     ? "LOG_ERROR"
                                                                     : warning ? "LOG_WARNING"
                                                                               : "LOG_INFO")));
    ctx->PushMonoFont();
    const float smallSize = EditorThemeRegistry::Float("EDITOR_SMALL_FONT_SIZE") * dpi;
    ImFont *mono = ImGui::GetFont();
    const ImU32 muted = ImGui::GetColorU32(EditorThemeRegistry::Color("TEXT_DISABLED"));
    const float timeWidth = mono->CalcTextSizeA(smallSize, FLT_MAX, 0, log.timestamp.c_str()).x;
    draw->AddText(mono, smallSize, ImVec2(lo.x + 28 * dpi, lo.y + 8 * dpi), muted, log.timestamp.c_str());
    const float textX = lo.x + 36 * dpi + timeWidth;
    const std::string count = ve.count > 1 ? std::to_string(ve.count) : "";
    const float countWidth = count.empty() ? 0 : mono->CalcTextSizeA(smallSize, FLT_MAX, 0, count.c_str()).x + 12 * dpi;
    const float textRight = (std::max)(textX, hi.x - 12 * dpi - countWidth);
    if (!log.sourceLabel.empty()) {
        const ImVec4 clip(textX, lo.y + 27 * dpi, textRight, hi.y - 4 * dpi);
        draw->AddText(mono, smallSize, ImVec2(textX, lo.y + 28 * dpi), muted, log.sourceLabel.c_str(), nullptr, 0,
                      &clip);
    }
    if (!count.empty()) {
        const ImVec2 badgeLo(hi.x - countWidth - 8 * dpi, lo.y + 6 * dpi);
        draw->AddRectFilled(badgeLo, ImVec2(hi.x - 8 * dpi, lo.y + 25 * dpi),
                            ImGui::GetColorU32(EditorThemeRegistry::Color("ROLE_BG_RAISED")));
        draw->AddText(mono, smallSize, ImVec2(badgeLo.x + 6 * dpi, lo.y + 10 * dpi), muted, count.c_str());
    }
    ctx->PopFont();
    const ImVec2 messageMin(textX, lo.y + 5 * dpi), messageMax(textRight, lo.y + 23 * dpi);
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::RenderTextEllipsis(draw, messageMin, messageMax, messageMax.x, log.displayLine.c_str(), nullptr, nullptr);
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered() && !log.sourceFile.empty())
        ImGui::SetTooltip("%s", log.sourceFile.c_str());
    ctx->RecordSemanticItem("console_entry", log.firstLine, true, "console.entry." + std::to_string(ve.uid), isSel,
                            static_cast<double>(ve.count));
    ImGui::PopStyleColor(3);
}

ImVec4 ConsolePanel::LevelColor(LogLevel lv) const
{
    switch (lv) {
    case LOG_ERROR:
    case LOG_FATAL:
        return EditorThemeRegistry::Color("ERROR_TEXT");
    case LOG_WARN:
        return EditorThemeRegistry::Color("CONSOLE_MESSAGE_WARNING");
    default:
        return EditorThemeRegistry::Color("ROLE_TEXT");
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
