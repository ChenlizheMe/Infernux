#pragma once

#include <memory>

namespace infernux
{
class RenderWorldFrame;
class InxScreenUIRenderer;

/// Immutable UI publication identity. Device storage remains private to its renderer.
class CompletedUISnapshot
{
  public:
    ~CompletedUISnapshot();
    [[nodiscard]] const std::shared_ptr<const RenderWorldFrame> &World() const noexcept
    {
        return m_world;
    }

  private:
    friend class InxScreenUIRenderer;
    CompletedUISnapshot();
    struct Data;
    std::shared_ptr<const RenderWorldFrame> m_world;
    std::unique_ptr<Data> m_data;
};
using CompletedUIFrame = std::shared_ptr<const CompletedUISnapshot>;
} // namespace infernux
