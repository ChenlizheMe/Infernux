#pragma once

#include "Component.h"

namespace infernux
{
/// Author and result lifetime shared by components in their Scene physics domain.
class PhysicsBody : public Component
{
  public:
    [[nodiscard]] static ComponentTypeConstraints GetTypeConstraints()
    {
        ComponentTypeConstraints constraints;
        constraints.satisfiedTypes = {"PhysicsBody", "native:PhysicsBody"};
        return constraints;
    }

    void OnDisable() override;

    [[nodiscard]] nlohmann::json SerializeDocument() const override;
    bool DeserializeDocument(const nlohmann::json &document) override;

  protected:
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
    uint64_t m_resultEpoch = 0;
};
} // namespace infernux
