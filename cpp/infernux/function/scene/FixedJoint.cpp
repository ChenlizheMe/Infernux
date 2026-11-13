#include "FixedJoint.h"
#include "ComponentDocumentValidation.h"
#include "ComponentFactory.h"
#include "Transform.h"
#include "TransformECSStore.h"
#include "physics/PhysicsWorld.h"

namespace infernux
{
namespace
{
const bool registered = ComponentFactory::Register(
    "FixedJoint", [] { return std::make_unique<FixedJoint>(); }, FixedJoint::ValidateSerializedDocument,
    FixedJoint::GetTypeConstraints(), FixedJoint::DescribeSemanticType);
} // namespace

SemanticTypeDescriptor FixedJoint::DescribeSemanticType()
{
    return DescribeJointType("FixedJoint", "infernux.component.fixed_joint", "Fixed Joint");
}

uint64_t FixedJoint::CreateConstraint(PhysicsWorld &world, uint32_t bodyId, uint32_t connectedBodyId)
{
    TransformECSStore::PhysicsPoseReadScope authored(TransformECSStore::Instance());
    const auto *transform = GetTransform();
    const auto localBakedAnchor = glm::conjugate(transform->GetWorldRotation()) * transform->TransformVector(m_anchor);
    return world.CreateFixedConstraint(*this, bodyId, connectedBodyId, localBakedAnchor, m_anchor, m_enableCollision);
}

void FixedJoint::UpdateConstraintSettings()
{
    if (!m_constraintId)
        RebuildConstraint();
    else
        PhysicsWorld::Instance().SetConstraintCollisionEnabled(m_constraintId, m_enableCollision);
}

void FixedJoint::ValidateSerializedDocument(const nlohmann::json &document)
{
    component_document_validation::ValidateComponentDocument(
        document, "FixedJoint", {"anchor", "connected_body_component_id", "enable_collision"});
    ValidateJointDocumentFields(document, "FixedJoint");
}

bool FixedJoint::DeserializeDocument(const nlohmann::json &document)
{
    ValidateSerializedDocument(document);
    if (!Component::DeserializeDocument(document))
        return false;
    ReadJointDocumentFields(document);
    ReleaseConstraint();
    RebuildConstraint();
    return true;
}

std::unique_ptr<Component> FixedJoint::Clone() const
{
    auto clone = std::make_unique<FixedJoint>();
    CopyJointFieldsTo(*clone);
    return clone;
}
} // namespace infernux
