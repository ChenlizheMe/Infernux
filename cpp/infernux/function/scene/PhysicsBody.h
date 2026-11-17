#pragma once

#include "Component.h"

namespace infernux
{
class PhysicsWorld;

/// Author and result lifetime shared by components with a physical backend.
class PhysicsBody : public Component
{
  public:
    [[nodiscard]] static ComponentTypeConstraints GetTypeConstraints()
    {
        ComponentTypeConstraints constraints;
        constraints.satisfiedTypes = {"PhysicsBody", "native:PhysicsBody"};
        return constraints;
    }

    [[nodiscard]] bool IsGpuEnabled() const noexcept
    {
        return m_gpu;
    }
    void SetGpuEnabled(bool gpu);
    [[nodiscard]] bool AllowsCrossBackendInteraction() const noexcept
    {
        return m_crossBackendInteraction;
    }
    void SetCrossBackendInteraction(bool enabled) noexcept
    {
        m_crossBackendInteraction = enabled;
    }
    void OnDisable() override;

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    bool DeserializeDocument(const nlohmann::json &document) override;

  protected:
    virtual void ValidateGpuSelection(bool gpu) const = 0;
    virtual void OnGpuSelectionChanged() = 0;

    void CopyPhysicsBodyAuthoringTo(PhysicsBody &clone) const noexcept;
    void InvalidatePhysicsResults() noexcept
    {
        ++m_resultEpoch;
    }
    [[nodiscard]] uint64_t GetPhysicsResultEpoch() const noexcept
    {
        return m_resultEpoch;
    }

  private:
    friend class PhysicsWorld;
    bool m_gpu = false;
    bool m_crossBackendInteraction = false;
    uint64_t m_resultEpoch = 0;
};
} // namespace infernux
