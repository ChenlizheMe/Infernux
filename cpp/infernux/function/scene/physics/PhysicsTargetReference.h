#pragma once

#include <cstdint>

namespace infernux
{
class Collider;
class Component;
class GameObject;

/// The target of a retained physics value, independent of body rebuilds and
/// Scene membership. Capture on the owning thread before publishing a query
/// epoch or delivering a callback; workers only copy this immutable identity.
class PhysicsTargetReference
{
  public:
    PhysicsTargetReference() = default;
    explicit PhysicsTargetReference(const Collider *collider);

    [[nodiscard]] Collider *GetCollider() const;
    [[nodiscard]] GameObject *GetGameObject() const;

  private:
    // Native embedded hosts and the extension can own different static runtime
    // registries. Resolve against the registry that published this value.
    using Resolver = Component *(*)(uint64_t, uint64_t);
    Resolver m_resolve = nullptr;
    uint64_t m_colliderId = 0;
    uint64_t m_colliderGeneration = 0;
    // A GameObject owns its intrinsic Transform for its entire lifetime. Its
    // registry identity lets the owner remain resolvable after collider removal.
    uint64_t m_ownerTransformId = 0;
    uint64_t m_ownerTransformGeneration = 0;
};
} // namespace infernux
