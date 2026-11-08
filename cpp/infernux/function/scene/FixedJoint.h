#pragma once
#include "Joint.h"

namespace infernux
{
/// Weld two bodies in their current relative pose, or weld this body to the world.
class FixedJoint final : public Joint
{
  public:
    const char *GetTypeName() const override
    {
        return "FixedJoint";
    }
    static SemanticTypeDescriptor DescribeSemanticType();
    static void ValidateSerializedDocument(const nlohmann::json &document);
    bool DeserializeDocument(const nlohmann::json &document) override;
    std::unique_ptr<Component> Clone() const override;

  private:
    uint64_t CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId) override;
    void UpdateConstraintSettings() override;
};
} // namespace infernux
