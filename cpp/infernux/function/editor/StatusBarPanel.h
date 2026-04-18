#pragma once

#include "EditorTheme.h"
#include <function/renderer/gui/InxGUIContext.h>
#include <function/renderer/gui/InxGUIRenderable.h>

#include <imgui.h>

#include <functional>
#include <mutex>
#include <string>

namespace infernux
{

class ConsolePanel;

/// C++ native status bar — fixed bar at the bottom of the editor window.
/// Not dockable; inherits InxGUIRenderable directly.
///
/// Reads log counts from ConsolePanel.
/// Engine-status text/progress is fed from Python via SetEngineStatus().
class StatusBarPanel : public InxGUIRenderable
{
  public:
    StatusBarPanel();
    ~StatusBarPanel() override = default;

    // ── Public API ───────────────────────────────────────────────────

    /// Wire to C++ ConsolePanel for count queries and "select latest" action.
    void SetConsolePanel(ConsolePanel *panel);

    /// Show a new log message in the left zone (called from Python listener).
    void SetLatestMessage(const std::string &message, const std::string &level);

    /// Clear the latest message and locally-tracked counts.
    void ClearCounts();

    /// Update engine-status indicator (called from Python every frame).
    void SetEngineStatus(const std::string &text, float progress);

    /// Warning / error counts tracked independently of ConsolePanel to match
    /// the Python StatusBarPanel's listener-based counting.
    void IncrementWarnCount();
    void IncrementErrorCount();

    // ── InxGUIRenderable ─────────────────────────────────────────────
    void OnRender(InxGUIContext *ctx) override;

  private:
    void RenderContent(InxGUIContext *ctx, float dispW);
    void RenderEngineStatus(InxGUIContext *ctx, float dispW, float leftZoneW, const std::string &text, float progress);

    const ImVec4 &LevelColor() const;
    const ImVec4 &LevelColorForString(const std::string &level) const;

    ConsolePanel *m_console = nullptr;

    // Latest message state
    std::string m_latestMsg;
    std::string m_latestLevel{"info"};

    // Counts (incremented via Python listener)
    int m_warnCount = 0;
    int m_errorCount = 0;

    // Engine status
    std::string m_statusText;
    float m_statusProgress = -1.0f;
};

} // namespace infernux
