#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

// The producer and retained UI packets share an allocation lease. Dropping
// the final CPU reference schedules destruction on the engine's existing
// submission retirement queue. Shutdown may invalidate GPU storage while CPU
// packet references survive; their later destructors do not touch the device.
class ImGuiTextureAllocationLease final
{
  public:
    using Retire = std::function<void(std::function<void()>)>;
    ImGuiTextureAllocationLease(Retire retire, std::function<void()> destroy)
        : m_retire(std::move(retire)), m_destroy(std::move(destroy))
    {
        if (!m_retire || !m_destroy)
            throw std::invalid_argument("UI texture allocation requires a submission retirement owner");
    }
    ~ImGuiTextureAllocationLease()
    {
        Release();
    }
    ImGuiTextureAllocationLease(const ImGuiTextureAllocationLease &) = delete;
    ImGuiTextureAllocationLease &operator=(const ImGuiTextureAllocationLease &) = delete;
    [[nodiscard]] bool IsAlive() const noexcept
    {
        return !!m_destroy;
    }
    void Release()
    {
        auto destroy = std::move(m_destroy);
        auto retire = std::move(m_retire);
        if (destroy)
            retire(std::move(destroy));
    }

  private:
    Retire m_retire;
    std::function<void()> m_destroy;
};
