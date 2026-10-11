#include "PhysicsTargetReference.h"

#include <function/scene/Collider.h>
#include <function/scene/GameObject.h>
#include <function/scene/Transform.h>

namespace infernux
{
namespace
{
Component *ResolveLifetime(uint64_t id, uint64_t generation)
{
    Component *current = Component::FindByComponentId(id);
    // A move between Scenes changes worldId, not native lifetime. Never resolve
    // by body ID or accept a replacement that reused the serialized component ID.
    return current && current->GetLifetimeGeneration() == generation ? current : nullptr;
}
} // namespace

PhysicsTargetReference::PhysicsTargetReference(const Collider *collider)
{
    if (!collider)
        return;
    m_resolve = &ResolveLifetime;
    m_colliderId = collider->GetComponentID();
    m_colliderGeneration = collider->GetLifetimeGeneration();
    if (GameObject *owner = collider->GetGameObject()) {
        m_ownerTransformId = owner->GetTransform()->GetComponentID();
        m_ownerTransformGeneration = owner->GetTransform()->GetLifetimeGeneration();
    }
}

Collider *PhysicsTargetReference::GetCollider() const
{
    return m_resolve ? static_cast<Collider *>(m_resolve(m_colliderId, m_colliderGeneration)) : nullptr;
}

GameObject *PhysicsTargetReference::GetGameObject() const
{
    Component *transform = m_resolve ? m_resolve(m_ownerTransformId, m_ownerTransformGeneration) : nullptr;
    return transform ? transform->GetGameObject() : nullptr;
}
} // namespace infernux
