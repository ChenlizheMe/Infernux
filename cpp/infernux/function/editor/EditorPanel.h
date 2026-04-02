#pragma once

#include <function/renderer/gui/InxGUIContext.h>
#include <function/renderer/gui/InxGUIRenderable.h>

#include <imgui.h>
#include <string>

namespace infernux
{

/// Base class for C++ editor panels.
/// Provides Begin/End window management so subclasses only implement
/// OnRenderContent().
class EditorPanel : public InxGUIRenderable
{
  public:
    explicit EditorPanel(const std::string &title, const std::string &windowId = "")
        : m_title(title), m_windowId(windowId.empty() ? title : windowId)
    {
    }

    ~EditorPanel() override = default;

    /// True when the panel window is open (visible & docked or floating).
    [[nodiscard]] bool IsOpen() const
    {
        return m_isOpen;
    }

    void SetOpen(bool open)
    {
        m_isOpen = open;
    }

    [[nodiscard]] const std::string &GetWindowId() const
    {
        return m_windowId;
    }

    /// InxGUIRenderable override — wraps content in ImGui::Begin/End.
    void OnRender(InxGUIContext *ctx) override
    {
        if (!m_isOpen)
            return;

        PreRender(ctx);

        // Use ###id for stable ImGui window identity
        std::string label = m_title + "###" + m_windowId;

        bool visible = ImGui::Begin(label.c_str(), &m_isOpen, GetWindowFlags());

        if (visible)
        {
            OnRenderContent(ctx);
        }

        ImGui::End();
    }

  protected:
    /// Override to draw the panel body.  Called between Begin/End.
    virtual void OnRenderContent(InxGUIContext *ctx) = 0;

    /// Override to supply custom ImGui window flags.
    virtual ImGuiWindowFlags GetWindowFlags() const
    {
        return ImGuiWindowFlags_None;
    }

    /// Override for per-frame work before window begins.
    virtual void PreRender(InxGUIContext * /*ctx*/)
    {
    }

    std::string m_title;
    std::string m_windowId;
    bool m_isOpen = true;
};

} // namespace infernux
