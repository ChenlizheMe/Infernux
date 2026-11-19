#include "PhysicsBody.h"

namespace infernux
{
void PhysicsBody::OnDisable()
{
    InvalidatePhysicsResults();
}

nlohmann::json PhysicsBody::SerializeDocument() const
{
    auto document = Component::SerializeDocument();
    return document;
}

bool PhysicsBody::DeserializeDocument(const nlohmann::json &document)
{
    if (!Component::DeserializeDocument(document))
        return false;
    return true;
}

void PhysicsBody::CopyPhysicsBodyAuthoringTo(PhysicsBody &clone) const noexcept
{
    clone.m_enabled = m_enabled;
    clone.m_executionOrder = m_executionOrder;
}
} // namespace infernux
