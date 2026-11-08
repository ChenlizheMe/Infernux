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

PhysicsTargetReference::PhysicsTargetReference(const Component *component)
{
    if (!component)
        return;
    m_resolve = &ResolveLifetime;
    m_componentId = component->GetComponentID();
    m_componentGeneration = component->GetLifetimeGeneration();
    auto *target = component->GetGameObject() ? component->GetGameObject()->GetTransform() : nullptr;
    if (target) {
        m_ownerTransformId = target->GetComponentID();
        m_ownerTransformGeneration = target->GetLifetimeGeneration();
    }
}

Component *PhysicsTargetReference::GetComponent() const
{
    return m_resolve ? m_resolve(m_componentId, m_componentGeneration) : nullptr;
}

Collider *PhysicsTargetReference::GetCollider() const
{
    return dynamic_cast<Collider *>(GetComponent());
}

GameObject *PhysicsTargetReference::GetGameObject() const
{
    Component *transform = m_resolve ? m_resolve(m_ownerTransformId, m_ownerTransformGeneration) : nullptr;
    return transform ? transform->GetGameObject() : nullptr;
}
} // namespace infernux
